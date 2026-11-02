#!/bin/sh
cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1
export SGLANG_MLX_GLM_DECODE_EVAL_GROUP=2 SGLANG_MLX_GLM_VERIFY_EVAL_GROUP=2
export MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
exec /Users/jay/workspace/prod/common/sglang-metal/bin/python -m sglang.launch_server \
 --model-path /Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
 --reasoning-parser glm45 --mem-fraction-static 0.78 --host 127.0.0.1 --port 30007 \
 --context-length 8192 --max-total-tokens 17024 --chunked-prefill-size 256 \
 --max-running-requests 2 --mlx-enable-sampling --random-seed 1234 \
 --enable-deterministic-inference --disable-cuda-graph --disable-overlap-schedule
