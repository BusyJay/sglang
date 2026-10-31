# SPDX-License-Identifier: Apache-2.0
"""Indexed MLA preserves legal KV sets and singleton dtype/reduction boundaries."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxRagged(CustomTestCase):
    def test_segmented_kv_preserves_native_reduction_and_visibility(self):
        """Index slots cross sealed/append boundaries without reading poisoned tails."""
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(817)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            q = mx.random.normal((1, 64, 3, 512)).astype(dtype)
            kv = mx.random.normal((1, 513, 512)).astype(dtype)
            lengths = [0, 17, 257]
            indices = (mx.arange(3 * 257).reshape(1, 3, 257) * 7 % 513).astype(mx.int32)
            for row, length in enumerate(lengths):
                indices[:, row, length:] = -9999
            expected = mlx_ragged_latent_attention(
                q, kv, indices, lengths, scale=128**-0.5
            )
            for split in (0, 1, 255, 256, 511, 513):
                actual = mlx_ragged_latent_attention(
                    q,
                    kv[:, :split],
                    indices,
                    lengths,
                    scale=128**-0.5,
                    tail=kv[:, split:],
                )
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_empty_segments_and_zero_visibility_preserve_attention(self):
        """An empty prefix must compile and attend to its tail at unchanged indices."""
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(818)
        for dtype, kv_dtype in (
            (mx.float16, mx.float16),
            (mx.bfloat16, mx.bfloat16),
            (mx.float32, mx.float32),
            (mx.float32, mx.bfloat16),
        ):
            kv = mx.random.normal((1, 17, 512)).astype(kv_dtype)
            empty = kv[:, :0]
            for rows in (1, 3):
                q = mx.random.normal((1, 64, rows, 512)).astype(dtype)
                lengths = [17] if rows == 1 else [0, 1, 17]
                indices = mx.broadcast_to(mx.arange(17, dtype=mx.int32), (1, rows, 17))
                for row, length in enumerate(lengths):
                    indices[:, row, length:] = -9999
                expected = mlx_ragged_latent_attention(
                    q, kv, indices, lengths, scale=128**-0.5
                )
                for split in (0, 1, 16, 17):
                    with self.subTest(dtype=dtype, rows=rows, split=split):
                        actual = mlx_ragged_latent_attention(
                            q,
                            kv[:, :split],
                            indices,
                            lengths,
                            scale=128**-0.5,
                            tail=kv[:, split:],
                        )
                        self.assertIsNotNone(actual)
                        self.assertTrue(mx.array_equal(actual, expected).item())
                for width in (0, 17):
                    padding = mx.full((1, rows, width), -9999, mx.int32)
                    for prefix, tail in ((empty, empty), (empty, kv), (kv, empty)):
                        actual = mlx_ragged_latent_attention(
                            q,
                            prefix,
                            padding,
                            [0] * rows,
                            scale=128**-0.5,
                            tail=tail,
                        )
                        self.assertIsNotNone(actual)
                        self.assertTrue(mx.array_equal(actual, mx.zeros_like(q)).item())

    def test_sparse_rows_empty_sets_and_poisoned_tails(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(714)
        for dtype, kv_dtype in (
            (mx.float16, mx.float16),
            (mx.bfloat16, mx.bfloat16),
            (mx.float32, mx.float32),
            (mx.float32, mx.bfloat16),
        ):
            for lengths in (
                [0, 1, 2],
                [1, 3, 7],
                [2, 3, 4],
                [5, 6, 7],
                [6, 7, 8],
                [8, 9, 15],
                [17, 18, 19],
                [31, 32, 33],
                [127, 128, 129],
                [255, 256, 257],
                [512, 513, 514],
                [1024, 1025, 1026],
                [2048, 2049, 2050],
                [4094, 4095, 4096],
                [0, 2048, 2050],
            ):
                width = max(lengths)
                q = mx.random.normal((1, 64, 3, 512)).astype(dtype)
                kv = mx.random.normal((1, 4096, 512)).astype(kv_dtype)
                rows = [
                    mx.random.permutation(4096)[:n].astype(mx.int32) for n in lengths
                ]
                indices = mx.stack(
                    [
                        mx.pad(row, (0, width - row.size), constant_values=-123456)
                        for row in rows
                    ]
                )[None]
                actual = mlx_ragged_latent_attention(
                    q, kv, indices, lengths, scale=256**-0.5
                )
                expected = []
                for j, (row, length) in enumerate(zip(rows, lengths)):
                    if not length:
                        expected.append(mx.zeros_like(q[:, :, j : j + 1]))
                    else:
                        selected = mx.take(kv, row, axis=1)[:, None]
                        expected.append(
                            mx.fast.scaled_dot_product_attention(
                                q[:, :, j : j + 1], selected, selected, scale=256**-0.5
                            )
                        )
                with self.subTest(dtype=dtype, lengths=lengths):
                    self.assertTrue(
                        mx.array_equal(actual, mx.concatenate(expected, 2)).item()
                    )


if __name__ == "__main__":
    unittest.main()
