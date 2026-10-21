#!/usr/bin/env bash
# Replay salted long/short traffic on a separate local test container.
# Use the locally measured CPU/GPU cost table.
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
results="${QWEN38_VERIFY_RESULTS:-$repo_root/tools/qwen38_cache_tests/migration_20261021/lifecycle-replay}"
mkdir -p "$results"
docker run --rm --name qwen38-hybrid-verify --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/results" \
  -v /home/user/workspace/prod/huggingface/hub:/hf-cache:ro \
  -v /home/user/workspace/prod/cache:/root/.cache \
  -e QWEN38_CANARY_SALT_MAX_NEW_TOKENS=128 \
  -e QWEN38_CANARY_FLAGS='{"kv_cache_stages":[80000,160000,220000],"preallocate_gdn_proj":true,"async_pack_prefetch":true,"prefill_direct_cache":false}' -v /home/user/.cargo/registry:/root/.cargo/registry -e SGLANG_RUST_BUILD_MODE=auto -e MAX_JOBS=2 -e CARGO_BUILD_JOBS=2 -e RUSTUP_TOOLCHAIN=stable \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
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
  tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/model_canary.py \
  --checkpoint /hf-cache/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300 --revision b9ce18a6887e18c089fdea8509a33e63429a3300 \
  --cache-dir /hf-cache --manifest /root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json \
  --output /results --runtime-mode threaded --adaptive-cache --total-slots 6388 --ple-backend pinned \
  --chunk 8192 --context-length 220000 --contexts 1024 170000 1024 170000 1024 --context-max-new-tokens 128 --context-ignore-eos \
  --decode-graph --mtp-steps 2 --mem-fraction-static 0.90 --max-mamba-cache-size 10 \
  --mamba-full-memory-ratio 0.22 --mamba-ssm-dtype bfloat16 \
  --startup-timeout 600 --request-timeout 600 --cache-stats-log-interval 128 \
  --probe 'What is 17 plus 25? Reply with just the number.|42'

python3 "$repo_root/tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/verify_lifecycle.py" "$results" --output "$results/validation-summary.json"
