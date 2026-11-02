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

    def test_block_fp8_output_tile_keeps_batch_rounding_and_tail(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_fp8_rows import (
            mlx_fp8_dynamic_32,
            mlx_fp8_rows,
        )

        mx.random.seed(453)
        # Real MLA output geometry selects the two-output SIMD tile.
        weights = mx.random.randint(0, 127, (4096, 512)).astype(mx.uint8)
        scales = mx.full((32, 4), 0.001, mx.float32)
        x = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
        y = mlx_fp8_rows(
            x, weights, scales, block_size=128, quantized_input=mlx_fp8_dynamic_32(x)
        )
        expected = mx.concatenate(
            [
                mlx_fp8_rows(
                    x[:, j : j + 1],
                    weights,
                    scales,
                    block_size=128,
                    quantized_input=mlx_fp8_dynamic_32(x[:, j : j + 1]),
                )
                for j in range(7)
            ],
            axis=1,
        )
        self.assertTrue(mx.array_equal(y, expected).item())

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

    def test_mxfp8_wide_order_unpadded_rows_and_views(self):
        """Changing lane grouping or scale placement must not change native rounding."""
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_mxfp8_rows import mlx_mxfp8_rows

        mx.random.seed(20261103)
        for dtype in (mx.bfloat16, mx.float32):
            for n, k in ((3, 32), (7, 544), (64, 1024)):
                w, scales = mx.quantize(mx.random.normal((n, k)), mode="mxfp8")
                for count in (1, 3, 4, 6, 9):
                    x = mx.random.normal((count, k * 2)).astype(dtype)[:, ::2]
                    parts = []
                    for j in range(0, count, 4):
                        tile = x[j : j + 4]
                        real = tile.shape[0]
                        tile = mx.concatenate([tile, mx.zeros((4 - real, k), dtype)])
                        parts.append(
                            mx.quantized_matmul(
                                tile, w, scales, mode="mxfp8", group_size=32, bits=8
                            )[:real]
                        )
                    expected = mx.concatenate(parts)
                    bits = mx.uint16 if dtype == mx.bfloat16 else mx.uint32
                    for packed in (w, w.view(mx.uint8)):
                        actual = mlx_mxfp8_rows(x, packed, scales)
                        self.assertTrue(
                            mx.array_equal(
                                actual.view(bits), expected.view(bits)
                            ).item()
                        )

    def test_mxfp8_scale_extremes_and_signed_codes(self):
        """UE8M0 zero is subnormal and E4M3 signed subnormals retain their sign."""
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_mxfp8_rows import mlx_mxfp8_rows

        x = mx.ones((3, 32), mx.float32)
        weights = mx.zeros((3, 32), mx.uint8)
        weights[:, 0] = mx.array([1, 129, 56], mx.uint8)
        scales = mx.array([[127], [127], [0]], mx.uint8)
        actual = mlx_mxfp8_rows(x, weights, scales)
        expected = mx.quantized_matmul(
            mx.concatenate([x, mx.zeros((1, 32), mx.float32)]),
            weights.view(mx.uint32),
            scales,
            mode="mxfp8",
            group_size=32,
            bits=8,
        )[:3]
        self.assertTrue(
            mx.array_equal(actual.view(mx.uint32), expected.view(mx.uint32)).item()
        )


if __name__ == "__main__":
    unittest.main()
