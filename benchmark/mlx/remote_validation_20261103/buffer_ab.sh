cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 MLX_MAX_OPS_PER_BUFFER=200
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
for mb in 8192 1024 32768; do
MLX_MAX_MB_PER_BUFFER=$mb $P -u benchmark/mlx/remote_validation_20261103/buffer_ab.py --modes baseline,fp8,combined,nvfp4 --prompts 1 --rounds 2 --output benchmark/mlx/remote_validation_20261103/buffer_${mb}.json > benchmark/mlx/remote_validation_20261103/buffer_${mb}.log 2>&1
rc=$?; tail -n 9 benchmark/mlx/remote_validation_20261103/buffer_${mb}.log
[ "$rc" = 0 ] || exit "$rc"
done
