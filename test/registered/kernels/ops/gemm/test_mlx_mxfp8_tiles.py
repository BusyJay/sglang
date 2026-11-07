# SPDX-License-Identifier: Apache-2.0
"""MXFP8 eight-vector tiles preserve native per-vector accumulation."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxMxfp8Tiles(CustomTestCase):
    def test_indexed_rows_preserve_full_projection_values_and_padding(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_mxfp8_rows import (
            mlx_mxfp8_indexed_rows,
            mlx_mxfp8_rows,
        )

        mx.random.seed(803)
        for dtype in (mx.bfloat16, mx.float32):
            weight = mx.random.randint(0, 127, (137, 512), dtype=mx.uint8)
            scales = mx.random.randint(119, 128, (137, 16), dtype=mx.uint8)
            ids = mx.array([136, 0xFFFFFFFF, 65, 137, 65, 0, 1, 31, 2], mx.uint32)
            for rows in (1, 3, 8, 17):
                x = mx.random.normal((1, rows, 512)).astype(dtype)
                full = mlx_mxfp8_rows(x, weight, scales)
                actual = mlx_mxfp8_indexed_rows(x, weight, scales, ids)
                expected = mx.take(full, mx.minimum(ids, 136), axis=-1)
                expected = mx.where(ids < 137, expected, -mx.inf)
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_partial_tiles_match_four_vector_reference(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_mxfp8_rows import mlx_mxfp8_rows

        for dtype in (mx.bfloat16, mx.float32):
            for rows in (1, 4, 5, 6, 7, 8, 9, 17):
                with self.subTest(dtype=dtype, rows=rows):
                    mx.random.seed(rows)
                    x = mx.random.normal((1, rows, 512)).astype(dtype)
                    weight = (
                        mx.random.randint(0, 127, (128, 512), dtype=mx.uint8)
                        | (mx.random.randint(0, 2, (128, 512), dtype=mx.uint8) * 128)
                    ).astype(mx.uint8)
                    scales = mx.random.randint(119, 128, (128, 16), dtype=mx.uint8)
                    expected = mlx_mxfp8_rows(
                        x, weight, scales, tilegroups=min(rows, 4)
                    )
                    actual = mlx_mxfp8_rows(x, weight, scales)
                    mx.eval(expected, actual)
                    self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
