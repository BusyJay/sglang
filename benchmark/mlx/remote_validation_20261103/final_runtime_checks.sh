cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
for test in test_glm5_next_mlx test_dflash_serving test_dflash2_mlx test_mlx_runner_pool_contract; do
$P test/registered/unit/hardware_backend/mlx/${test}.py -f > benchmark/mlx/remote_validation_20261103/final_${test}.log 2>&1
rc=$?
tail -n 8 benchmark/mlx/remote_validation_20261103/final_${test}.log
[ "$rc" = 0 ] || exit "$rc"
done
