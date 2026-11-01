cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python MLX_ENABLE_TF32=0
py=/Users/jay/workspace/prod/common/sglang-metal/bin/python
for mode in baseline candidate; do
 flag=""
 if [ "$mode" = baseline ]; then flag="--baseline"; fi
 "$py" benchmark/mlx/dflash2_20261102/verify_host_ab.py $flag --tokens 128 --repeats 1 --output "benchmark/mlx/dflash2_20261102/verify_host_ab_${mode}.json" > "benchmark/mlx/dflash2_20261102/verify_host_ab_${mode}.log" 2>&1
 rc=$?
 tail -6 "benchmark/mlx/dflash2_20261102/verify_host_ab_${mode}.log"
 if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
