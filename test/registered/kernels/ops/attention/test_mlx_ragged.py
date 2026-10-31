# SPDX-License-Identifier: Apache-2.0
"""Indexed MLA preserves legal KV sets and singleton dtype/reduction boundaries."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxRagged(CustomTestCase):
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
