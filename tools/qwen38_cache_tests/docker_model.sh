#!/usr/bin/env bash
# Existing local checkpoint or HF cache only; runs conversion or canary.
set -euo pipefail
image="${SGLANG_IMAGE:-}"
device="${QWEN38_GPU_DEVICE:-0}"
checkpoint=""
hf_cache="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
pack=""
timeout_seconds="${QWEN38_TEST_TIMEOUT_SECONDS:-0}"
idle_seconds="${QWEN38_IDLE_SECONDS:-60}"
stall_action="${QWEN38_STALL_ACTION:-kill}"
confirm_seconds="${QWEN38_CONFIRM_SECONDS:-30}"
results=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --hf-cache) hf_cache="$2"; shift 2 ;;
    --pack) pack="$2"; shift 2 ;;
    --results) results="$2"; shift 2 ;;
    --stall-action) stall_action="$2"; shift 2 ;;
    --confirm-seconds) confirm_seconds="$2"; shift 2 ;;
    --idle-seconds) idle_seconds="$2"; shift 2 ;;
    --timeout) timeout_seconds="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Usage: $0 --image IMAGE --device GPU --checkpoint DIR_OR_REPO_ID [--hf-cache HUB_DIR] --pack DIR --results DIR -- convert|canary [args]" >&2; exit 2 ;;
  esac
done
: "${image:?select a local image}"
: "${checkpoint:?checkpoint directory or cached model ID required}"
: "${pack:?runtime pack directory required}"
: "${results:?results directory required}"
mode="${1:?select convert or canary}"
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$pack" "$results"
pack="$(cd "$pack" && pwd)"
results="$(cd "$results" && pwd)"
# HF offline caches resolve snapshot entries through relative symlinks into the
# repo-level blobs directory; binding only a snapshots directory leaves them
# dangling inside the container. Bind the models--<name> tree instead and map
# the checkpoint path below it.
if [[ -d "$checkpoint" ]]; then
checkpoint="$(cd "$checkpoint" && pwd)"
bind_source="$checkpoint"
container_checkpoint="/workspace/checkpoint"
snapshot_parent="$(dirname "$checkpoint")"
repo_dir="$(dirname "$snapshot_parent")"
if [[ "$(basename "$snapshot_parent")" == "snapshots" && -d "$repo_dir/blobs" ]]; then
  bind_source="$repo_dir"
  container_checkpoint="/workspace/checkpoint-root/$(basename "$repo_dir")/snapshots/$(basename "$checkpoint")"
  # Distinguish a self-contained repo cache from a deduplicated layout where
  # repo blobs entries are symlinks into an outer shared pool: only then must
  # the outer pool directory be part of the mounted tree.
  case "$(find -H "$repo_dir/blobs" -maxdepth 1 -type l -print -quit 2>/dev/null)" in
    "") ;;
    *) if [[ -d "$(dirname "$repo_dir")/blobs" ]]; then
         # Keep the computed relative path; only widen the mounted tree so the
         # outer pool stays reachable through the repo-level symlinks.
         bind_source="$(dirname "$repo_dir")"
       fi;;
  esac
fi
else
  if [[ ! -d "$hf_cache" ]]; then
    echo "HF hub cache directory not found: $hf_cache (set --hf-cache or HF_HUB_CACHE); no download attempted" >&2
    exit 2
  fi
  hf_cache="$(cd "$hf_cache" && pwd)"
  container_checkpoint="$checkpoint"
fi
case "$mode" in
  convert) command=(-m sglang.srt.model_loader.qwen38_nvfp4_pack.convert --checkpoint "$container_checkpoint" --output /workspace/pack) ;;
  canary) command=(tools/qwen38_cache_tests/full_model_canary.py --checkpoint "$container_checkpoint" --manifest /workspace/pack/manifest.json --output /workspace/results) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
if [[ ! -d "$checkpoint" ]]; then
  HF_BINDS=(-v "$hf_cache:/workspace/hf-cache:ro")
elif [[ "$container_checkpoint" == "/workspace/checkpoint" ]]; then
  HF_BINDS=(-v "$checkpoint:/workspace/checkpoint:ro")
else
  bind_target="/workspace/checkpoint-root"
  if [[ "$bind_source" == "$repo_dir" ]]; then
    bind_target="$bind_target/$(basename "$repo_dir")"
  fi
  HF_BINDS=(-v "$bind_source:$bind_target:ro")
fi
# Model containers also expose the conventional production cache root so
# HF-style environment lookups work alongside the explicit checkpoint binds.
PROD_MOUNTS=()
if [[ -n "${QWEN38_PROD_MOUNT:-}" && -d "${QWEN38_PROD_MOUNT:-}" || -d "/home/jay/workspace/prod" ]]; then
  PROD_MOUNTS=(-v "${QWEN38_PROD_MOUNT:-/home/jay/workspace/prod}:/prod:ro")
fi
docker image inspect "$image" --format 'Using local image {{.Id}}'
exec python3 "$repo_root/tools/qwen38_cache_tests/docker_supervisor.py" \
  --timeout "$timeout_seconds" --idle-seconds "$idle_seconds" --stall-action "$stall_action" --confirm-seconds "$confirm_seconds" --gpu-device "$device" --results "$results" -- \
  docker run --rm --pull never --network none \
  --gpus "device=$device" --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" \
  ${HF_BINDS[@]} ${PROD_MOUNTS[@]} \
  -v "$pack:/workspace/pack" -v "$results:/workspace/results" \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_CACHE=/workspace/hf-cache \
  --entrypoint python3 "$image" "${command[@]}" "$@"
