"""FP8 cache encoding, selected-row attention and snapshot lifetime on Metal."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=20, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxFP8Cache(CustomTestCase):
    def test_native_encoding_and_signed_extreme_scales(self):
        import mlx.core as mx
        from sglang.kernels.ops.quantization.mlx_fp8_cache import (
            mlx_fp8_cache_pack,
            mlx_fp8_cache_unpack,
        )

        for dtype in (mx.bfloat16, mx.float32):
            for rows in (1, 8, 256):
                for magnitude in (0, 1e-12, 0.1, 1e12):
                    mx.random.seed(rows)
                    x = (mx.random.normal((1, rows, 512)) * magnitude).astype(dtype)
                    w, s = mx.quantize(x.reshape(-1, 512), mode="mxfp8")
                    expected = mx.concatenate(
                        [w.view(mx.uint8).reshape(x.shape), s.reshape(1, rows, 16)], -1
                    )
                    actual = mlx_fp8_cache_pack(x)
                    self.assertTrue(mx.array_equal(actual, expected).item())
        codes = mx.concatenate(
            [
                mx.arange(127, dtype=mx.uint8),
                mx.arange(128, 255, dtype=mx.uint8),
                mx.zeros(2, mx.uint8),
            ]
        )
        for scale in (0, 1, 16, 100, 127, 200, 240, 254):
            packed = mx.concatenate([codes, mx.full((8,), scale, mx.uint8)]).reshape(
                1, 264
            )
            reference = mx.dequantize(
                codes.view(mx.uint32).reshape(1, 64),
                mx.full((1, 8), scale, mx.uint8),
                mode="mxfp8",
            ).astype(mx.bfloat16)
            self.assertTrue(
                mx.array_equal(mlx_fp8_cache_unpack(packed, 256), reference).item()
            )

    def test_ragged_selected_tail_and_runtime_lengths(self):
        import mlx.core as mx
        from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention
        from sglang.kernels.ops.quantization.mlx_fp8_cache import (
            mlx_fp8_cache_pack,
            mlx_fp8_cache_unpack,
        )

        mx.random.seed(77)
        for rows in (1, 8):
            q = mx.random.normal((1, 64, rows, 512)).astype(mx.bfloat16)
            packed = mlx_fp8_cache_pack(
                (mx.random.normal((1, 273, 512)) * 0.1).astype(mx.bfloat16)
            )
            decoded = mlx_fp8_cache_unpack(packed)
            for length in (0, 1, 17, 31, 33, 127, 257, 273):
                indices = mx.broadcast_to(
                    mx.arange(384, dtype=mx.int32)[None, None], (1, rows, 384)
                )
                # Poison padding: reading it would produce an OOB access.
                indices = mx.where(
                    mx.arange(384)[None, None] < length, indices, -999999
                ).astype(mx.int32)
                lens = [max(0, length - r) for r in range(rows)]
                actual = mlx_ragged_latent_attention(
                    q,
                    packed[:, :256],
                    indices,
                    lens,
                    scale=256**-0.5,
                    tail=packed[:, 256:],
                )
                expected = mlx_ragged_latent_attention(
                    q,
                    decoded[:, :256],
                    indices,
                    lens,
                    scale=256**-0.5,
                    tail=decoded[:, 256:],
                )
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_packed_sparse_gather_promotes_logical_bf16(self):
        import mlx.core as mx
        from sglang.kernels.ops.kvcache.mlx_latent import mlx_latent_gather_cast
        from sglang.kernels.ops.quantization.mlx_fp8_cache import (
            mlx_fp8_cache_pack,
            mlx_fp8_cache_unpack,
        )

        mx.random.seed(101)
        values = mx.random.normal((1, 2307, 512)).astype(mx.bfloat16)
        packed = mlx_fp8_cache_pack(values)
        decoded = mlx_fp8_cache_unpack(packed)
        for dtype in (mx.bfloat16, mx.float32):
            for rows, width in ((1, 17), (8, 257), (256, 2051)):
                ids = mx.random.randint(0, 2307, (1, rows, width)).astype(mx.int32)
                ids = mx.where(mx.arange(width)[None, None] % 31 == 0, -999999, ids)
                actual = mlx_latent_gather_cast(
                    packed[:, :2304], ids, dtype, tail=packed[:, 2304:]
                )
                expected = (
                    mx.take_along_axis(
                        decoded[:, None], mx.maximum(ids, 0)[..., None], axis=2
                    )
                    .reshape(rows, 1, width, 512)
                    .astype(dtype)
                )
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_rejection_fork_and_append_seal(self):
        import mlx.core as mx
        from sglang.kernels.ops.quantization.mlx_fp8_cache import (
            mlx_fp8_cache_pack,
            mlx_fp8_cache_unpack,
        )
        from sglang.srt.hardware_backend.mlx.models.glm5_next import FP8MLACache
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        mx.random.seed(91)
        values = mx.random.normal((1, 520, 512)).astype(mx.bfloat16)
        decoded = mlx_fp8_cache_unpack(mlx_fp8_cache_pack(values))
        cache = FP8MLACache(4)
        snapshots = []
        for start, stop in (
            (0, 1),
            (1, 255),
            (255, 256),
            (256, 259),
            (259, 510),
            (510, 520),
        ):
            if start:
                snapshots.append(clone_cache(cache))
            cache.append_latent(values[:, start:stop], materialize=False)
            cache.offset = stop
            self.assertTrue(mx.array_equal(cache[0], decoded[:, :stop]).item())
        for snapshot in snapshots:
            restored = clone_cache(snapshot)
            suffix = mx.full((1, 7, 512), -0.125, mx.bfloat16)
            restored.append_latent(suffix, materialize=False)
            restored.offset += 7
            self.assertTrue(mx.array_equal(restored[0][:, -7:], suffix).item())
            self.assertTrue(
                mx.array_equal(snapshot[0], decoded[:, : snapshot.offset]).item()
            )
            self.assertEqual(restored.cache[0].dtype, mx.uint8)


if __name__ == "__main__":
    unittest.main()
