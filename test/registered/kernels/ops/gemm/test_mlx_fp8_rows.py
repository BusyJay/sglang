# SPDX-License-Identifier: Apache-2.0
import unittest


from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(
    __import__("sys").platform == "darwin", "Requires MLX Metal device"
)
class TestMlxFP8Rows(CustomTestCase):
    def test_dynamic_ue8m0_and_batch_boundaries(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_fp8_rows import (
            mlx_fp8_dynamic_32,
            mlx_fp8_rows,
        )

        x = mx.full((1, 5, 256), 448.0, mx.bfloat16)
        q, s = mlx_fp8_dynamic_32(x)
        self.assertTrue(mx.all(q == 126).item())
        self.assertTrue(mx.all(s == 1.0).item())
        zero, scale = mlx_fp8_dynamic_32(mx.zeros_like(x))
        self.assertTrue(mx.all(zero == 0).item())
        self.assertTrue(mx.all(scale == 1.0).item())
        w = mx.full((128, 256), 56, mx.uint8)
        scales = mx.full((1, 2), 0.5, mx.float32)
        y = mlx_fp8_rows(x, w, scales, block_size=128, quantized_input=(q, s))
        self.assertTrue(mx.all(y == 57344).item())
        parts = []
        for j in range(5):
            v = x[:, j : j + 1]
            parts.append(
                mlx_fp8_rows(
                    v, w, scales, block_size=128, quantized_input=mlx_fp8_dynamic_32(v)
                )
            )
        self.assertTrue(mx.array_equal(y, mx.concatenate(parts, 1)).item())

    def test_e4m3_signed_subnormal_and_block_scale(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_rows

        w = mx.zeros((4, 128), mx.uint8)
        w[:, 0] = mx.array([1, 129, 56, 184], mx.uint8)
        x = mx.zeros((1, 4, 128), mx.float32)
        x[:, :, 0] = 1
        y = mlx_fp8_rows(x, w, mx.full((4, 4), 127, mx.uint8))
        self.assertTrue(
            mx.array_equal(
                y,
                mx.broadcast_to(
                    mx.array([1 / 512, -1 / 512, 1, -1], mx.float32), (1, 4, 4)
                ),
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
