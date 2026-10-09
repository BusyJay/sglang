#!/usr/bin/env bash
# Use an explicitly selected local image; no image pulls, GPU or HF network.
set -euo pipefail
image="${1:-${SGLANG_IMAGE:-}}"
if [[ -z "$image" ]]; then
  echo "Usage: $0 <local-image> (or set SGLANG_IMAGE)" >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
docker image inspect "$image" --format 'Testing local image {{.Id}}'
exec docker run --rm --pull never --network none \
  --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" \
  -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 "$image" \
  tools/qwen38_cache_tests/run_container_cpu.py
