#!/usr/bin/env bash
# Explicit local paths/image; runs only the requested conversion or canary.
set -euo pipefail
image="${SGLANG_IMAGE:-}"
device="${QWEN38_GPU_DEVICE:-0}"
checkpoint=""
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
    --pack) pack="$2"; shift 2 ;;
    --results) results="$2"; shift 2 ;;
    --stall-action) stall_action="$2"; shift 2 ;;
    --confirm-seconds) confirm_seconds="$2"; shift 2 ;;
    --idle-seconds) idle_seconds="$2"; shift 2 ;;
    --timeout) timeout_seconds="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Usage: $0 --image IMAGE --device GPU --checkpoint DIR --pack DIR --results DIR -- convert|canary [args]" >&2; exit 2 ;;
  esac
done
: "${image:?select a local image}"
: "${checkpoint:?local checkpoint required}"
: "${pack:?runtime pack directory required}"
: "${results:?results directory required}"
mode="${1:?select convert or canary}"
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="$(cd "$checkpoint" && pwd)"
mkdir -p "$pack" "$results"
pack="$(cd "$pack" && pwd)"
results="$(cd "$results" && pwd)"
# HF offline caches resolve snapshot entries through relative symlinks into the
# repo-level blobs directory; binding only a snapshots directory leaves them
# dangling inside the container. Bind the models--<name> tree instead and map
# the checkpoint path below it.
bind_source="$checkpoint"
container_checkpoint="/workspace/checkpoint"
snapshot_parent="$(dirname "$checkpoint")"
if [[ "$(basename "$snapshot_parent")" == "snapshots" && -d "$(dirname "$snapshot_parent")/blobs" ]]; then
  bind_source="$(dirname "$snapshot_parent")"
  container_checkpoint="/workspace/checkpoint-root/snapshots/$(basename "$checkpoint")"
fi
case "$mode" in
  convert) command=(-m sglang.srt.model_loader.qwen38_nvfp4_pack.convert --checkpoint "$container_checkpoint" --output /workspace/pack) ;;
  canary) command=(tools/qwen38_cache_tests/full_model_canary.py --checkpoint "$container_checkpoint" --manifest /workspace/pack/manifest.json --output /workspace/results) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
HF_BINDS=(-v "$bind_source:/workspace/checkpoint:ro")
if [[ "$container_checkpoint" != "/workspace/checkpoint" ]]; then
  HF_BINDS=(-v "$bind_source:/workspace/checkpoint-root:ro")
fi
docker image inspect "$image" --format 'Using local image {{.Id}}'
exec python3 "$repo_root/tools/qwen38_cache_tests/docker_supervisor.py" \
  --timeout "$timeout_seconds" --idle-seconds "$idle_seconds" --stall-action "$stall_action" --confirm-seconds "$confirm_seconds" --gpu-device "$device" --results "$results" -- \
  docker run --rm --pull never --network none \
  --gpus "device=$device" --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" \
  ${HF_BINDS[@]} \
  -v "$pack:/workspace/pack" -v "$results:/workspace/results" \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 "$image" "${command[@]}" "$@"
