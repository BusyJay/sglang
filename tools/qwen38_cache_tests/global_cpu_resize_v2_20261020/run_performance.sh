#!/usr/bin/env bash
# Compare CPU hybrid, global GPU miss handling, and the legacy GPU path.
# CPU hybrid, native-library path and core-list overrides are intentionally absent.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
results="${QWEN38_VERIFY_RESULTS:-$repo_root/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/performance-20261021}"
mkdir -p "$results"
docker run --rm --name qwen38-performance --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/results" \
  -v /home/jay/workspace/prod/huggingface/hub:/hf-cache:ro \
  -v /home/jay/workspace/prod/cache:/root/.cache \
  -v /home/jay/.cargo/registry:/root/.cargo/registry -e SGLANG_RUST_BUILD_MODE=auto -e MAX_JOBS=2 -e CARGO_BUILD_JOBS=2 -e RUSTUP_TOOLCHAIN=stable \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1 -e SGLANG_QWEN38_FILL_CTA=adaptive \
  -e SGLANG_QWEN38_STARTUP_CACHE=auto -e SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup \
  -e SGLANG_QWEN38_HOST_TOKEN_EMBEDDING=1 -e SGLANG_QWEN38_BORROW_PREFILL_ARENA=1 -e SGLANG_QWEN38_PLE_PACKED_CACHE=1 \
  -e SGLANG_QWEN38_CACHE_W4A16=1 -e SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x \
  -e SGLANG_QWEN38_CACHE_PREFILL_W4A16=0 -e SGLANG_QWEN38_FP32_NORM_FUSION=1 \
  -e SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=1 \
  --entrypoint python3 swallow-sglang:nightly-dev-20261018-b234b5c3 \
  tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py \
  --reference /workspace/sglang/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/performance-command.json \
  --output /results "$@"

python3 "$repo_root/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/summarize_performance.py" "$results" --output "$results/performance-summary.json"
