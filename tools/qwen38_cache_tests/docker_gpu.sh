#!/usr/bin/env bash
set -euo pipefail
image="${SGLANG_IMAGE:-}"
device="${QWEN38_GPU_DEVICE:-0}"
timeout_seconds="${QWEN38_TEST_TIMEOUT_SECONDS:-300}"
idle_seconds="${QWEN38_IDLE_SECONDS:-60}"
stall_action="${QWEN38_STALL_ACTION:-kill}"
confirm_seconds="${QWEN38_CONFIRM_SECONDS:-30}"
results="${QWEN38_RESULTS_DIR:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --results) results="$2"; shift 2 ;;
    --stall-action) stall_action="$2"; shift 2 ;;
    --confirm-seconds) confirm_seconds="$2"; shift 2 ;;
    --idle-seconds) idle_seconds="$2"; shift 2 ;;
    --timeout) timeout_seconds="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Usage: $0 --image LOCAL_IMAGE --device GPU --results DIR [-- test-args]" >&2; exit 2 ;;
  esac
done
: "${image:?select local image}"
: "${results:?select results directory}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$results"
results="$(cd "$results" && pwd)"
docker image inspect "$image" --format 'Testing local image {{.Id}}'
exec python3 "$repo_root/tools/qwen38_cache_tests/docker_supervisor.py" \
  --timeout "$timeout_seconds" --idle-seconds "$idle_seconds" --stall-action "$stall_action" --confirm-seconds "$confirm_seconds" --gpu-device "$device" --results "$results" -- \
  docker run --rm --pull never --network none \
  --gpus "device=$device" --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/workspace/results" \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 "$image" tools/qwen38_cache_tests/run_gpu.py "$@"
