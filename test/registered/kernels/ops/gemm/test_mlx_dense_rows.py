# SPDX-License-Identifier: Apache-2.0
"""Dense shared-weight rows preserve each native GEMV's arithmetic."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxDenseRows(CustomTestCase):
    def test_shared_loads_match_singletons_and_tail_rows(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows

        mx.random.seed(901)
        for dtype in (mx.float16, mx.bfloat16):
            for n, k in (
                (258, 256),
                (1026, 256),
                (4096, 512),
                (8192, 1024),
                (4096, 4096),
            ):
                weight = (mx.random.normal((n, k)) * 0.02).astype(dtype)
                for count in (1, 2, 3, 5, 9, 17):
                    with self.subTest(dtype=dtype, n=n, k=k, count=count):
                        x = mx.random.normal((1, count, k)).astype(dtype)
                        result = mlx_dense_rows(x, weight)
                        self.assertIsNotNone(result)
                        expected = mx.concatenate(
                            [x[:, j : j + 1] @ weight.T for j in range(count)], 1
                        )
                        self.assertTrue(mx.array_equal(result, expected).item())

    def test_strided_inputs_and_unsupported_reductions(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows

        mx.random.seed(902)
        weight = mx.random.normal((256, 258)).astype(mx.bfloat16)[:, :256]
        x = mx.random.normal((1, 6, 512)).astype(mx.bfloat16)[:, ::2, ::2]
        actual = mlx_dense_rows(x, weight)
        expected = mx.concatenate([x[:, j : j + 1] @ weight.T for j in range(3)], 1)
        self.assertTrue(mx.array_equal(actual, expected).item())
        for value, matrix in (
            (x, weight[:, :128]),
            (mx.zeros((1, 18, 256), mx.bfloat16), weight),
            (mx.zeros((2, 3, 256), mx.bfloat16), weight),
            (mx.zeros((1, 2, 256), mx.bfloat16), weight[:4]),
        ):
            with self.subTest(
                shape=value.shape, matrix=matrix.shape, dtype=value.dtype
            ):
                self.assertIsNone(mlx_dense_rows(value, matrix))

    def test_mixed_projection_and_latent_singleton_reductions(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import (
            mlx_dense_rows,
            mlx_headed_dense_rows,
        )

        mx.random.seed(711)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            x = mx.random.normal((1, 3, 1536)).astype(dtype)
            w = mx.random.normal((4096, 1536)).astype(mx.bfloat16)
            expected = mx.concatenate([x[:, j : j + 1] @ w.T for j in range(3)], 1)
            actual = mlx_dense_rows(x, w)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(mx.array_equal(actual, expected).item())
            for k, n in ((256, 512), (512, 256)):
                x = mx.random.normal((1, 8, 3, k)).astype(dtype)
                w = mx.random.normal((8, k, n)).astype(dtype)
                expected = mx.concatenate([x[:, :, j : j + 1] @ w for j in range(3)], 2)
                actual = mlx_headed_dense_rows(x, w, transposed=True)
                self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
