cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=1024
/Users/jay/workspace/prod/common/sglang-metal/bin/python -u benchmark/mlx/remote_validation_20261103/fullmodel_mechanism.py --modes fp8,current,reuse2,fused --prompts 1 --rounds 2 --output benchmark/mlx/remote_validation_20261103/fullmodel_mechanism.json > benchmark/mlx/remote_validation_20261103/fullmodel_mechanism.log 2>&1
rc=$?
tail -n 18 benchmark/mlx/remote_validation_20261103/fullmodel_mechanism.log
exit "$rc"
