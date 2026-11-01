# SPDX-License-Identifier: Apache-2.0
"""Exact MLX top16 ordering, rounded values, and tile boundaries."""

import importlib.util
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxTopk(CustomTestCase):
    def check_topk(self, data, dtype, k=16):
        import mlx.core as mx

        from sglang.kernels.ops.sampling.mlx_topk import mlx_topk

        logits = mx.array(data, dtype=dtype)
        values, ids = mlx_topk(logits, k)
        mx.eval(values, ids)
        # BF16 ties are defined on the rounded input, not the FP32 generator.
        rounded = np.asarray(logits.astype(mx.float32))
        expected_ids = np.argsort(-rounded, axis=-1, kind="stable")[..., :k]
        expected_values = np.take_along_axis(rounded, expected_ids, axis=-1)
        self.assertEqual(values.dtype, dtype)
        self.assertEqual(ids.dtype, mx.uint32)
        self.assertEqual(values.shape, data.shape[:-1] + (k,))
        np.testing.assert_array_equal(np.asarray(ids), expected_ids)
        np.testing.assert_array_equal(
            np.asarray(values.astype(mx.float32)), expected_values
        )

    def test_target_random_and_rounded_ties(self):
        import mlx.core as mx

        rng = np.random.default_rng(20261102)
        for dtype in (mx.float32, mx.bfloat16):
            for rows in (1, 7, 14):
                with self.subTest(dtype=dtype, rows=rows):
                    self.check_topk(
                        rng.normal(-5, 2, (rows, 154880)).astype(np.float32), dtype
                    )
            self.check_topk(np.zeros((2, 7, 154880), np.float32), dtype)

    def test_tiles_tails_and_concentrated_winners(self):
        import mlx.core as mx

        rng = np.random.default_rng(8)
        for dtype in (mx.float32, mx.bfloat16):
            for width in (16, 17, 255, 2047, 2048, 2049, 4097, 154881):
                with self.subTest(dtype=dtype, width=width):
                    data = rng.normal(-20, 1, (3, width)).astype(np.float32)
                    data[0, -16:] = 10
                    data[1, :16] = np.arange(16)
                    data[2, :] = -np.inf
                    self.check_topk(data, dtype)
            data = np.full((2, 4097), -np.inf, np.float32)
            data[:, [0, 31, 2047, 2048, 4096]] = 3
            self.check_topk(data, dtype)

    def test_special_values_signed_zero_and_one_dimensional(self):
        import mlx.core as mx

        for dtype in (mx.float32, mx.bfloat16):
            data = np.full((3, 4097), -np.inf, np.float32)
            data[0, [15, 2048, 4096]] = np.inf
            data[1, :] = np.nan
            data[1, [0, 17, 2049]] = -np.inf
            data[2, :40:2] = -0.0
            data[2, 1:40:2] = 0.0
            self.check_topk(data, dtype)
            self.check_topk(data[0], dtype)

    def test_strided_inputs_and_recursive_candidate_merge(self):
        import mlx.core as mx

        from sglang.kernels.ops.sampling.mlx_topk import mlx_topk

        rng = np.random.default_rng(4)
        source = mx.array(rng.normal(size=(4, 8194)).astype(np.float32))
        logits = source[::2, ::2]
        values, ids = mlx_topk(logits)
        expected = np.argsort(-np.asarray(logits), kind="stable")[:, :16]
        np.testing.assert_array_equal(np.asarray(ids), expected)
        np.testing.assert_array_equal(
            np.asarray(values), np.take_along_axis(np.asarray(logits), expected, -1)
        )
        self.check_topk(rng.normal(size=(1, 262145)).astype(np.float32), mx.float32)

    def test_other_k_and_invalid_arguments(self):
        import mlx.core as mx

        from sglang.kernels.ops.sampling.mlx_topk import mlx_topk

        data = np.zeros((2, 33), np.float32)
        for k in (1, 7, 33):
            self.check_topk(data, mx.float32, k)
        for k in (0, 34, True, 1.5):
            with self.assertRaises(ValueError):
                mlx_topk(mx.array(data), k)
        with self.assertRaises(ValueError):
            mlx_topk(mx.array(1.0))
        with self.assertRaises(TypeError):
            mlx_topk(mx.zeros((1, 32), mx.int32))


if __name__ == "__main__":
    unittest.main()
