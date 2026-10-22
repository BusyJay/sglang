#!/bin/bash
# Run on the target SM120 host; paths/configuration are explicit arguments.
set -euo pipefail
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 REPO IMAGE MANIFEST CPU_LIBRARY CORES SOURCE_MODE OUTPUT_DIR" >&2
    exit 2
fi
repo=$(realpath "$1")
image=$2
manifest=$(realpath "$3")
library=$(realpath "$4")
payload=$(python3 - "$manifest" <<'PYTHON'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
payload = Path(json.loads(manifest.read_text())["payload"])
print((payload if payload.is_absolute() else manifest.parent / payload).resolve())
PYTHON
)
cores=$5
source_mode=$6
mkdir -p "$7"
output=$(realpath "$7")
# Preserve caller tuning while retaining the original comparison defaults.
for shape in m3 m12; do
    docker run --rm --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
        -v "$repo:/workspace/sglang:ro" -v "$payload:$payload:ro" \
        -v "$(dirname "$manifest"):$(dirname "$manifest"):ro" \
        -v "$library:$library:ro" -v "$output:/results" \
        -w /workspace/sglang -e PYTHONPATH=python -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONUNBUFFERED=1 -e SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-fill-gate \
        -e "SGLANG_QWEN38_FILL_CTA=${SGLANG_QWEN38_FILL_CTA:-adaptive}" \
        -e "SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=${SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH:-1}" \
        -e "SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=${SGLANG_QWEN38_CACHE_W4A16_SPLIT_K:-1}" \
        --entrypoint python3 "$image" test/manual/kernels/moe/bench_qwen38_hybrid_gate.py \
        --manifest "$manifest" --library "$library" \
        --cases "/workspace/sglang/tools/qwen38_cache_tests/bottleneck_review_20261022/fill_cases/gate-cases-$shape.json" \
        --cores "$cores" --source-mode "$source_mode" --epilogue merge \
        --samples 11 --warmup 4 --margin 0.05 --output "/results/gate-fill-$shape.json"
done
docker run --rm --pull never --network none -v "$repo:/workspace/sglang:ro" \
    -v "$output:/results" -w /workspace/sglang --entrypoint python3 "$image" \
    tools/qwen38_cache_tests/plan_20261022/merge_calibration.py \
    --prefix /results/gate-fill --output /results/calibration.json
