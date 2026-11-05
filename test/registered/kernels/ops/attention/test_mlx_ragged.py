# SPDX-License-Identifier: Apache-2.0
"""Indexed MLA preserves legal KV sets and singleton dtype/reduction boundaries."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxRagged(CustomTestCase):
    def test_implicit_positions_match_tables_with_runtime_visibility(self):
        """Removing index tables must keep per-row causality and segmented addresses."""
        import mlx.core as mx
        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(930)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            q = mx.random.normal((1, 64, 4, 512)).astype(dtype)
            kv = mx.random.normal((1, 513, 512)).astype(dtype)
            traced = []

            def forward(query, left, right, lengths, split):
                traced.append(True)
                return mlx_ragged_latent_attention(
                    query,
                    left,
                    None,
                    lengths,
                    scale=128**-0.5,
                    tail=right,
                    split=split,
                    implicit_width=1024,
                )

            graph = mx.compile(forward)
            for split in (0, 1, 255, 256, 513):
                prefix, tail = mx.zeros_like(kv), mx.zeros_like(kv)
                prefix[:, :split] = kv[:, :split]
                tail[:, : 513 - split] = kv[:, split:]
                for end in (1, 31, 32, 255, 256, 257, 511, 512, 513):
                    lengths = mx.array(
                        [0, min(1, end), max(0, end - 1), end], mx.uint32
                    )
                    positions = mx.broadcast_to(mx.arange(1024), (1, 4, 1024))
                    expected = mlx_ragged_latent_attention(
                        q,
                        kv,
                        positions,
                        lengths,
                        scale=128**-0.5,
                    )
                    actual = graph(q, prefix, tail, lengths, mx.array(split, mx.uint32))
                    self.assertTrue(
                        mx.array_equal(actual, expected).item(), (dtype, split, end)
                    )
            self.assertEqual(len(traced), 1)

    def test_width_buckets_above_4096_preserve_native_rows(self):
        """A padded index bucket must preserve short and looped long reductions."""
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(823)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            q = mx.random.normal((1, 64, 3, 512)).astype(dtype)
            kv = mx.random.normal((1, 8193, 512)).astype(dtype)
            for lengths in (
                [0, 2049, 4096],
                [4095, 4096, 4097],
                [4097, 4098, 4099],
                [8191, 8192, 8193],
            ):
                width = 1 << (max(lengths) - 1).bit_length()
                indices = mx.broadcast_to(
                    mx.arange(width, dtype=mx.int32), (1, 3, width)
                )
                expected = []
                for row, length in enumerate(lengths):
                    indices[:, row, length:] = -9999
                    expected.append(
                        mx.fast.scaled_dot_product_attention(
                            q[:, :, row : row + 1],
                            kv[:, None, :length],
                            kv[:, None, :length],
                            scale=128**-0.5,
                        )
                        if length
                        else mx.zeros_like(q[:, :, row : row + 1])
                    )
                actual = mlx_ragged_latent_attention(
                    q,
                    kv[:, :4096],
                    indices,
                    mx.array(lengths, mx.uint32),
                    tail=kv[:, 4096:],
                    scale=128**-0.5,
                )
                self.assertIsNotNone(actual)
                self.assertTrue(
                    mx.array_equal(actual, mx.concatenate(expected, 2)).item(),
                    (dtype, lengths),
                )
                implicit = mlx_ragged_latent_attention(
                    q,
                    kv[:, :4096],
                    None,
                    mx.array(lengths, mx.uint32),
                    tail=kv[:, 4096:],
                    scale=128**-0.5,
                    implicit_width=width,
                )
                self.assertTrue(mx.array_equal(implicit, actual).item())

    def test_compiled_runtime_lengths_and_split_preserve_reductions(self):
        """One graph accepts growing visibility and a moved sealed boundary."""
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention

        mx.random.seed(819)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            q = mx.random.normal((1, 64, 3, 512)).astype(dtype)
            kv = mx.random.normal((1, 513, 512)).astype(dtype)
            prefix = mx.zeros_like(kv)
            tail = mx.zeros_like(kv)
            traced = []

            def forward(query, left, right, table, lengths, split):
                traced.append(True)
                return mlx_ragged_latent_attention(
                    query,
                    left,
                    table,
                    lengths,
                    scale=128**-0.5,
                    tail=right,
                    split=split,
                )

            graph = mx.compile(forward)
            for split in (0, 1, 255, 256, 511, 513):
                prefix[:, :split] = kv[:, :split]
                tail[:, : 513 - split] = kv[:, split:]
                for length in (1, 17, 255, 256, 257, 511, 513):
                    lengths = [0, min(17, length), length]
                    table = mx.broadcast_to(mx.arange(513), (1, 3, 513))
                    for row, visible in enumerate(lengths):
                        table[:, row, visible:] = -9999
                    expected = mlx_ragged_latent_attention(
                        q, kv, table, lengths, scale=128**-0.5
                    )
                    actual = graph(
                        q,
                        prefix,
                        tail,
                        table,
                        mx.array(lengths, mx.uint32),
                        mx.array(split, mx.uint32),
                    )
                    self.assertTrue(mx.array_equal(actual, expected).item())
            self.assertEqual(len(traced), 1)

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
