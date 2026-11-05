"""Grouped W4A8 uses original NVFP4 scales and explicit expert ownership."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=15, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxNVFP4Prefill(CustomTestCase):
    def test_token_and_per_pick_rows_with_small_and_dense_experts(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_nvfp4_prefill import (
            mlx_nvfp4_prefill_a8,
            mlx_nvfp4_prefill_gate_up_a8,
            mlx_nvfp4_prefill_groups,
        )
        from sglang.kernels.ops.quantization.mlx_fp8_cache import (
            mlx_fp8_cache_pack,
            mlx_fp8_cache_unpack,
        )

        for dtype in (mx.bfloat16, mx.float32):
            for rows in (1, 8, 64):
                for per_pick in (False, True):
                    mx.random.seed(rows)
                    e, n, k, picks = 16, 128, 512, 4
                    global_scale = mx.array(0.75, mx.float32)
                    w, scales = mx.quantize(
                        (mx.random.normal((e, n, k)) * 0.1).astype(mx.bfloat16),
                        mode="nvfp4",
                        global_scale=global_scale,
                    )
                    ids = mx.random.randint(0, e, (1, rows, picks)).astype(mx.uint32)
                    x = (
                        mx.random.normal(
                            (1, rows, picks, 1, k) if per_pick else (1, rows, k)
                        )
                        * 0.1
                    ).astype(dtype)
                    groups = mlx_nvfp4_prefill_groups(ids, e)
                    actual = mlx_nvfp4_prefill_a8(
                        x,
                        w.view(mx.uint8),
                        scales,
                        mx.full((e,), 0.75, mx.float32),
                        ids,
                        groups,
                    )
                    qx = mlx_fp8_cache_unpack(mlx_fp8_cache_pack(x), k).astype(dtype)
                    expected = mx.gather_qmm(
                        qx if per_pick else qx[..., None, None, :],
                        w,
                        scales,
                        rhs_indices=ids,
                        mode="nvfp4",
                        group_size=16,
                        bits=4,
                        global_scale=mx.full((e,), 0.75, mx.float32),
                    )
                    if not per_pick:
                        g, u = mlx_nvfp4_prefill_gate_up_a8(
                            x,
                            (w.view(mx.uint8), scales, mx.full((e,), 0.75, mx.float32)),
                            (w.view(mx.uint8), scales, mx.full((e,), 0.75, mx.float32)),
                            ids,
                            groups,
                        )
                        self.assertTrue(mx.array_equal(g, actual).item())
                        self.assertTrue(mx.array_equal(u, actual).item())
                    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
                    # Different FP32 accumulation schedules are allowed; bound
                    # their error well below one BF16 output rounding interval.
                    relative = mx.linalg.norm(delta) / mx.linalg.norm(
                        expected.astype(mx.float32)
                    )
                    self.assertLess(relative.item(), 1e-4)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, dtype)
                    for expert, count in enumerate(groups[1].tolist()):
                        owners = groups[0][expert, :count].tolist()
                        self.assertEqual(
                            owners,
                            [
                                i
                                for i, v in enumerate(ids.reshape(-1).tolist())
                                if v == expert
                            ],
                        )


if __name__ == "__main__":
    unittest.main()
