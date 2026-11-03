cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=1024
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
for mode in ar dflash; do
$P -u benchmark/mlx/remote_validation_20261103/http_final.py --mode "$mode" --count 3 --tokens 300 --output benchmark/mlx/remote_validation_20261103/http_final_${mode}.json > benchmark/mlx/remote_validation_20261103/http_final_${mode}.log 2>&1
rc=$?
tail -n 12 benchmark/mlx/remote_validation_20261103/http_final_${mode}.log
[ "$rc" = 0 ] || exit "$rc"
done
