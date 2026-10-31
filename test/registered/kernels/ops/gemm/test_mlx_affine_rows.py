# SPDX-License-Identifier: Apache-2.0
"""Batched affine rows retain native singleton QMV arithmetic."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxAffineRows(CustomTestCase):
    def test_dense_and_routed_rows_match_native_singletons(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        mx.random.seed(123)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for count in (1, 3, 17):
                    with self.subTest(dtype=dtype, bits=bits, count=count):
                        x = mx.random.normal((1, count, 512)).astype(dtype)
                        w, s, b = mx.quantize(
                            (mx.random.normal((8, 256, 512)) * 0.02).astype(dtype),
                            group_size=64,
                            bits=bits,
                        )
                        indices = mx.array([[[1, 3, 7]] * count], mx.uint32)
                        for per_expert in (False, True):
                            act = (
                                mx.repeat(x[:, :, None], 3, axis=2) if per_expert else x
                            )
                            if per_expert:
                                act = (
                                    act
                                    + mx.array([0, 0.125, -0.25], dtype)[
                                        None, None, :, None
                                    ]
                                )
                            actual = mlx_affine_rows(
                                act,
                                w,
                                s,
                                b,
                                bits=bits,
                                group_size=64,
                                indices=indices,
                            )
                            expected = []
                            for j in range(count):
                                row = act[:, j : j + 1]
                                row = (
                                    row[..., None, :]
                                    if per_expert
                                    else row[:, :, None, None, :]
                                )
                                expected.append(
                                    mx.gather_qmm(
                                        row,
                                        w,
                                        s,
                                        b,
                                        rhs_indices=indices[:, j : j + 1],
                                        transpose=True,
                                        group_size=64,
                                        bits=bits,
                                    ).squeeze(-2)
                                )
                            self.assertTrue(
                                mx.array_equal(
                                    actual, mx.concatenate(expected, 1)
                                ).item()
                            )
                        actual = mlx_affine_rows(
                            x, w[0], s[0], b[0], bits=bits, group_size=64
                        )
                        expected = mx.concatenate(
                            [
                                mx.quantized_matmul(
                                    x[:, j : j + 1],
                                    w[0],
                                    s[0],
                                    b[0],
                                    transpose=True,
                                    group_size=64,
                                    bits=bits,
                                )
                                for j in range(count)
                            ],
                            1,
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_group_sizes_and_long_reductions_match_native(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        mx.random.seed(617)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for group, dim in ((32, 512), (64, 1536), (128, 4096)):
                    with self.subTest(dtype=dtype, bits=bits, group=group, dim=dim):
                        x = mx.random.normal((1, 3, dim)).astype(dtype)
                        w, s, b = mx.quantize(
                            (mx.random.normal((32, dim)) * 0.02).astype(dtype),
                            group_size=group,
                            bits=bits,
                        )
                        actual = mlx_affine_rows(
                            x, w, s, b, bits=bits, group_size=group
                        )
                        expected = mx.concatenate(
                            [
                                mx.quantized_matmul(
                                    x[:, j : j + 1],
                                    w,
                                    s,
                                    b,
                                    transpose=True,
                                    group_size=group,
                                    bits=bits,
                                )
                                for j in range(3)
                            ],
                            1,
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_unsupported_tiles_are_explicit(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        w, s, b = mx.quantize(mx.ones((16, 512), mx.float16), group_size=64, bits=4)
        for x in (
            mx.ones((2, 2, 512), mx.float16),
            mx.ones((1, 18, 512), mx.float16),
            mx.ones((1, 2, 512), mx.float32),
        ):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(mlx_affine_rows(x, w, s, b, bits=4, group_size=64))
