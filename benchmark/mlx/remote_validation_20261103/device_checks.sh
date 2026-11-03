cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200
P=/Users/jay/workspace/prod/common/sglang-metal/bin/python
$P test/registered/kernels/ops/gemm/test_mlx_block_fp8.py -f > benchmark/mlx/remote_validation_20261103/test_fp8.log 2>&1
rc=$?; tail -n 10 benchmark/mlx/remote_validation_20261103/test_fp8.log
[ "$rc" = 0 ] || exit "$rc"
$P test/registered/kernels/ops/gemm/test_mlx_nvfp4_gather.py -f > benchmark/mlx/remote_validation_20261103/test_nvfp4.log 2>&1
rc=$?; tail -n 10 benchmark/mlx/remote_validation_20261103/test_nvfp4.log
[ "$rc" = 0 ] || exit "$rc"
$P test/registered/unit/hardware_backend/mlx/test_mlx_startup_warmup.py -f > benchmark/mlx/remote_validation_20261103/test_warmup.log 2>&1
rc=$?; tail -n 10 benchmark/mlx/remote_validation_20261103/test_warmup.log
exit "$rc"
