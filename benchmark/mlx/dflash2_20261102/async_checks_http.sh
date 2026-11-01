cd /Users/jay/workspace/sglang || exit 1
export PYTHONPATH=python MLX_ENABLE_TF32=0
py=/Users/jay/workspace/prod/common/sglang-metal/bin/python
for f in test/registered/unit/hardware_backend/mlx/test_auxiliary_state_acquire.py test/registered/unit/hardware_backend/mlx/test_dflash_serving.py test/registered/unit/hardware_backend/mlx/test_glm5_next_mlx.py; do
 "$py" "$f" || exit $?
done
for count in 3 5; do
 "$py" benchmark/mlx/dflash2_20261102/http_probe.py --mode dflash --count "$count" --tokens 256 --output "benchmark/mlx/dflash2_20261102/http_verify_final_d${count}.json" > "benchmark/mlx/dflash2_20261102/http_verify_final_d${count}.log" 2>&1
 rc=$?
 tail -9 "benchmark/mlx/dflash2_20261102/http_verify_final_d${count}.log"
 if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
