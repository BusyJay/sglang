cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
export MTL_CAPTURE_ENABLED=1
/Users/jay/workspace/prod/common/sglang-metal/bin/python -u benchmark/mlx/remote_validation_20261103/m5_geometry.py > benchmark/mlx/remote_validation_20261103/m5_geometry.log 2>&1
rc=$?; tail -n 18 benchmark/mlx/remote_validation_20261103/m5_geometry.log
exit "$rc"
