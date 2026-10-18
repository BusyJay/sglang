#!/usr/bin/env bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace/sglang/python:/workspace/sglang
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/sglang
exec python -m unittest test.registered.unit.layers.quantization.test_qwen38_cache_cost_replay -v
