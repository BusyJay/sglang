cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=1024
/Users/jay/workspace/prod/common/sglang-metal/bin/python -u benchmark/mlx/remote_validation_20261103/macro_strided.py > benchmark/mlx/remote_validation_20261103/macro_strided.log 2>&1
rc=$?
tail -n 12 benchmark/mlx/remote_validation_20261103/macro_strided.log
exit "$rc"
