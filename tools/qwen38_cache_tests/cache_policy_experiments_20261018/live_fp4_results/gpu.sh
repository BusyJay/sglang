#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/sglang/python:/workspace/sglang
export SGLANG_QWEN38_FILL_CTA=adaptive
export SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1
export SGLANG_QWEN38_SOURCE_MODE=shared_thp
export SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /workspace/sglang
exec python "$@"
