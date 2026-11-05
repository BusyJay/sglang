# SPDX-License-Identifier: Apache-2.0
"""MXFP8 eight-vector tiles preserve native per-vector accumulation."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxMxfp8Tiles(CustomTestCase):
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
