cd /Users/jay/workspace/sglang && export PYTHONPATH=python MLX_ENABLE_TF32=0
PY_REVIEW=/Users/jay/workspace/prod/common/sglang-metal/bin/python
OUT_REVIEW=benchmark/mlx/review_optimization_20261101/mechanism_revision
$PY_REVIEW test/registered/kernels/ops/gemm/test_mlx_affine_rows.py TestMlxAffineRows.test_four_bit_masks_preserve_native_rounding_and_duplicate_picks -f > "$OUT_REVIEW/final_gate_check.log" 2>&1 &&
$PY_REVIEW test/registered/kernels/ops/gemm/test_mlx_dense_rows.py -f > "$OUT_REVIEW/final_head_check.log" 2>&1 &&
$PY_REVIEW test/registered/unit/hardware_backend/mlx/test_glm5_next_mlx.py TestGlm5NextMlx.test_token_only_serving_switches_contract_at_consumption_boundary TestGlm5NextMlx.test_token_only_target_retains_logits_and_cache_contract TestGlm5NextMlx.test_fixed_groups_refresh_replaced_layer_container -f > "$OUT_REVIEW/final_cache_check.log" 2>&1 &&
$PY_REVIEW "$OUT_REVIEW/final_compare.py" > "$OUT_REVIEW/final_compare.log" 2>&1 &&
$PY_REVIEW "$OUT_REVIEW/final_http.py" --mode candidate --group 300 --repeat 1 --port 30022 > "$OUT_REVIEW/final_http.log" 2>&1
