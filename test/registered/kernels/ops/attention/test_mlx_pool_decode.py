"""Numerical parity of token-indexed Metal attention against unpadded SDPA."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=8, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
if _HAS_MLX:
    import mlx.core as mx
    import numpy as np

    from sglang.kernels.ops.attention.mlx_pool_decode import mlx_pool_decode_attention


@unittest.skipUnless(_HAS_MLX, "requires MLX on Apple Silicon")
class TestMlxPoolDecode(CustomTestCase):
    def test_fragmented_slots_and_partition_boundaries(self):
        """Ragged tails and empty splits must never read unused table entries."""
        mx.random.seed(17)
        rng = np.random.default_rng(17)
        lengths = [1, 127, 128, 129, 257]
        slots = np.full((len(lengths), max(lengths)), -1, dtype=np.int32)
        for i, length in enumerate(lengths):
            slots[i, :length] = rng.choice(np.arange(1, 700), length, replace=False)
        slots = mx.array(slots)
        for dtype, atol in (
            (mx.float32, 2e-5),
            (mx.float16, 3e-3),
            (mx.bfloat16, 3e-2),
        ):
            for dim, heads, kv_heads in (
                (32, 4, 4),
                (64, 8, 2),
                (128, 8, 1),
                (256, 4, 2),
            ):
                with self.subTest(dtype=dtype, dim=dim, kv_heads=kv_heads):
                    keys = mx.random.normal((700, kv_heads, dim)).astype(dtype)
                    values = mx.random.normal(keys.shape).astype(dtype)
                    queries = mx.random.normal((len(lengths), heads, 1, dim)).astype(
                        dtype
                    )
                    # Slot 0 is padding, never a real token.
                    keys[0] = float("nan")
                    values[0] = float("nan")
                    result = mlx_pool_decode_attention(
                        queries,
                        keys,
                        values,
                        slots,
                        mx.array(lengths, mx.int32),
                        scale=dim**-0.5,
                    )
                    reference = mx.concatenate(
                        [
                            mx.fast.scaled_dot_product_attention(
                                queries[i : i + 1],
                                keys[slots[i, :length]].transpose(1, 0, 2)[None],
                                values[slots[i, :length]].transpose(1, 0, 2)[None],
                                scale=dim**-0.5,
                            )
                            for i, length in enumerate(lengths)
                        ]
                    )
                    self.assertEqual(result.dtype, dtype)
                    self.assertTrue(
                        mx.allclose(result, reference, atol=atol, rtol=atol).item()
                    )

    def test_lazy_scatter_and_large_logits(self):
        """Attention must depend on the newest lazy pool write and stable softmax."""
        dim = 64
        queries = mx.full((1, 4, 1, dim), 100.0)
        keys = mx.zeros((400, 1, dim))
        values = mx.zeros_like(keys)
        slots = mx.arange(1, 258, dtype=mx.int32)[None]
        keys[257] = 100.0
        values[257] = 3.0
        result = mlx_pool_decode_attention(
            queries, keys, values, slots, mx.array([257], mx.int32), scale=dim**-0.5
        )
        self.assertTrue(mx.allclose(result, mx.full(result.shape, 3.0)).item())


if __name__ == "__main__":
    unittest.main()
