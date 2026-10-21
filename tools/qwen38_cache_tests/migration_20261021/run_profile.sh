#!/usr/bin/env bash
# Capture warmed CPU/GPU serving work on RTX 5090 D.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
results="${QWEN38_VERIFY_RESULTS:-$repo_root/tools/qwen38_cache_tests/migration_20261021/nsight-final}"
mkdir -p "$results"
docker run --rm --cap-add SYS_PTRACE --security-opt seccomp=unconfined --name qwen38-nsight --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v /tmp/qwen38-profile-deps:/profiledeps:ro -v /usr/lib/nsight-systems:/usr/lib/nsight-systems:ro \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/results" \
  -v /home/jay/workspace/prod/huggingface/hub:/hf-cache:ro \
  -v /home/jay/workspace/prod/cache:/root/.cache \
  -v /home/jay/.cargo/registry:/root/.cargo/registry -e SGLANG_RUST_BUILD_MODE=auto -e MAX_JOBS=2 -e CARGO_BUILD_JOBS=2 -e RUSTUP_TOOLCHAIN=stable \
  -w /workspace/sglang -e PYTHONPATH=/profiledeps:/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e QWEN38_REFERENCE="${QWEN38_REFERENCE:-/workspace/sglang/tools/qwen38_cache_tests/migration_20261021/performance-command-compact.json}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_ENABLE_QWEN38_CPU_HYBRID=1 \
  -e SGLANG_QWEN38_CPU_LIBRARY=/workspace/sglang/tools/qwen38_cache_tests/migration_20261021/cpu.so \
  -e SGLANG_QWEN38_CPU_CORES=2,3,4,5 \
  -e SGLANG_QWEN38_CPU_COST_CALIBRATION="${QWEN38_CALIBRATION:-/workspace/sglang/tools/qwen38_cache_tests/migration_20261021/calibration-compact.json}" \
  -e SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1 -e SGLANG_QWEN38_FILL_CTA=adaptive \
  -e SGLANG_QWEN38_STARTUP_CACHE=auto -e SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup \
  -e SGLANG_QWEN38_HOST_TOKEN_EMBEDDING=1 -e SGLANG_QWEN38_BORROW_PREFILL_ARENA=1 -e SGLANG_QWEN38_PLE_PACKED_CACHE=1 \
  -e SGLANG_QWEN38_CACHE_W4A16=1 -e SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x \
  -e SGLANG_QWEN38_CACHE_PREFILL_W4A16=0 -e SGLANG_QWEN38_FP32_NORM_FUSION=1 \
  -e SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=1 \
  --entrypoint python3 swallow-sglang:nightly-dev-20261018-b234b5c3 \
  tools/qwen38_cache_tests/migration_20261021/profile_nsight.py --output /results --modes default "$@"
