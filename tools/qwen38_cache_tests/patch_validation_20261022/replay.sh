#!/usr/bin/env bash
# Run a fresh named variant on this host; never replace saved measurements.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
baseline_root=${QWEN38_BASELINE_ROOT:-/tmp/optimization-validation-20261022/baseline}
image=${QWEN38_IMAGE:-swallow-sglang:nightly-dev-20261018-b234b5c3}
if [ "$#" -eq 0 ] || [ -e "$here/$1/performance.json" ]; then
  echo 'Pass a fresh output name, followed by run_variant.py arguments.' >&2
  exit 2
fi
if [ ! -d "$baseline_root/python" ]; then
  echo 'Set QWEN38_BASELINE_ROOT to a checkout of 9321a3f4504534811676c4d1f39d62341089e610.' >&2
  exit 2
fi
docker run --rm --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$here:/results" -v "$baseline_root:/results/baseline:ro" \
  -v /home/user/workspace/prod/huggingface/hub:/hf-cache:ro \
  -v /home/user/workspace/prod/cache:/root/.cache \
  -v /home/user/.cargo/registry:/root/.cargo/registry \
  -e SGLANG_RUST_BUILD_MODE=auto -e RUSTUP_TOOLCHAIN=stable -e MAX_JOBS=2 -e CARGO_BUILD_JOBS=2 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/sglang/python \
  -w /workspace/sglang --entrypoint python3 "$image" /results/run_variant.py "$@"
