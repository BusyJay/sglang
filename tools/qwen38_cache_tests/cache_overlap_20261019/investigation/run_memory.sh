#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
results="$repo_root/tools/qwen38_cache_tests/cache_overlap_20261019"
docker run --rm --name qwen38-overlap-mem-diag-20261019 --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/results" \
  -v /home/user/workspace/prod/huggingface/hub:/hf-cache:ro \
  -v /home/user/workspace/prod/cache:/root/.cache \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1 -e SGLANG_QWEN38_FILL_CTA=adaptive \
  -e SGLANG_QWEN38_STARTUP_CACHE=auto -e SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup \
  -e SGLANG_QWEN38_HOST_TOKEN_EMBEDDING=1 -e SGLANG_QWEN38_BORROW_PREFILL_ARENA=1 -e SGLANG_QWEN38_PLE_PACKED_CACHE=1 \
  -e SGLANG_QWEN38_CACHE_W4A16=1 -e SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x \
  -e SGLANG_QWEN38_CACHE_PREFILL_W4A16=0 -e SGLANG_QWEN38_FP32_NORM_FUSION=1 \
  -e SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=1 \
  -e SGLANG_QWEN38_OOM_MEMORY_DIR=/results/investigation/oom-snapshots \
  -e SGLANG_QWEN38_OOM_MEMORY_ENTRIES=200000 \
  -e SGLANG_MEM_PROFILE_MAX_ENTRIES=200000 \
  --entrypoint python3 swallow-sglang:nightly-dev-20261018-b234b5c3 \
  tools/qwen38_cache_tests/cache_overlap_20261019/investigation/memory_canary.py \
  --checkpoint /hf-cache/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300 --revision b9ce18a6887e18c089fdea8509a33e63429a3300 \
  --cache-dir /hf-cache --manifest /root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json \
  --output /results/investigation/model-memory --runtime-mode threaded --adaptive-cache --total-slots 6388 --ple-backend pinned \
  --chunk 8192 --context-length 220000 --contexts 1024 8192 74000 --context-max-new-tokens 8 \
  --decode-graph --mtp-steps 2 --mem-fraction-static 0.90 --max-mamba-cache-size 10 \
  --mamba-full-memory-ratio 0.22 --mamba-ssm-dtype bfloat16 \
  --startup-timeout 600 --request-timeout 600 \
  --probe 'What is 17 plus 25? Reply with just the number.|42'
