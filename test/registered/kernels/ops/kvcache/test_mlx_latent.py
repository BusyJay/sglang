"""Segmented latent gather retains legal slots and the promotion boundary."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxLatent(CustomTestCase):
    def test_gather_across_append_boundary_and_negative_padding(self):
        import mlx.core as mx

        from sglang.kernels.ops.kvcache.mlx_latent import mlx_latent_gather_cast

        mx.random.seed(21)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            kv = mx.random.normal((1, 259, 512)).astype(dtype)
            for split in (0, 1, 255, 256, 259):
                for width in (3, 17, 257):
                    indices = (
                        mx.arange(3 * width).reshape(1, 3, width) * 17 % 259
                    ).astype(mx.int32)
                    indices[..., 0] = -1
                    for output_dtype in (dtype, mx.float32):
                        actual = mlx_latent_gather_cast(
                            kv[:, :split], indices, output_dtype, tail=kv[:, split:]
                        )
                        expected = (
                            mx.take_along_axis(
                                kv[:, None], mx.maximum(indices, 0)[..., None], axis=2
                            )
                            .reshape(3, 1, width, 512)
                            .astype(output_dtype)
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
