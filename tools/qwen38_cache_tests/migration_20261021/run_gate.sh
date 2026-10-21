#!/bin/bash
set -euo pipefail
label=$1
cases=$2
overlay=${3:-/home/jay/workspace/sglang/test/manual/kernels/moe/bench_qwen38_hybrid_gate.py}
docker run --rm --name qwen38-migration-gate --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
-v /home/jay/workspace/sglang:/workspace/sglang:ro -v /home/jay/workspace/prod/cache:/root/.cache \
-v /home/jay/workspace/sglang/tools/qwen38_cache_tests/migration_20261021:/results \
-v "$overlay:/workspace/sglang/test/manual/kernels/moe/bench_qwen38_hybrid_gate.py:ro" \
-w /workspace/sglang -e PYTHONPATH=python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
-e SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup \
-e SGLANG_QWEN38_FILL_CTA=adaptive -e SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1 -e SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=1 \
--entrypoint python3 swallow-sglang:nightly-dev-20261018-b234b5c3 test/manual/kernels/moe/bench_qwen38_hybrid_gate.py \
--manifest /root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json \
--library /results/cpu.so --cases "/results/$cases" --cores 2,3,4,5 --source-mode shared_thp \
--epilogue "${4:-merge}" --samples 11 --warmup 4 --margin 0.05 --output "/results/$label.json"
