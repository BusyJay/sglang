"""Four-stream mHC fusion retains Sinkhorn orientation and branch normalization."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxHc(CustomTestCase):
    def test_mix_product_rounds_before_bias_in_cancelling_streams(self):
        """Contracting multiply+bias changed real-model routing after cancellation."""
        import mlx.core as mx
        import numpy as np

        from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm

        product = np.float32(np.float32(100.7) * np.float32(30.3))
        mix = mx.array([100.7] + [0.0] * 23, mx.float32)
        base = mx.array([-float(product)] + [0.0] * 23, mx.float32)
        x = mx.broadcast_to(
            mx.array([1, -1, 0, 0], mx.bfloat16).reshape(1, 1, 4, 1),
            (1, 1, 4, 128),
        )
        result = mlx_hc_collapse_norm(
            x,
            mix,
            base,
            mx.array([30.3, 1.0, 1.0]),
            mx.ones((128,), mx.bfloat16),
            eps=1e-6,
            norm_eps=1e-5,
            iters=20,
        )
        self.assertIsNotNone(result)
        mx.eval(result)
        # Separately rounded products make both pre weights identical.
        self.assertEqual(mx.max(mx.abs(result[0])).item(), 0.0)

    def test_collapse_norm_and_transposed_residual_match_reference(self):
        import mlx.core as mx
        import mlx.nn as nn

        from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _hyper_connection

        mx.random.seed(20261030)
        for dtype in (mx.bfloat16, mx.float16):
            for iters in (1, 20):
                with self.subTest(dtype=dtype, iters=iters):
                    x = mx.random.normal((1, 1, 4, 256)).astype(dtype)
                    fn = mx.random.normal((24, 1024)) * 0.03
                    base = mx.random.normal((24,))
                    scale = mx.array([0.5, 1.5, 0.75])
                    norm = nn.RMSNorm(256)
                    norm.set_dtype(dtype)
                    norm.weight = mx.random.uniform(
                        low=0.1, high=2, shape=(256,)
                    ).astype(dtype)
                    collapsed, post, residual = _hyper_connection(
                        x, fn, base, scale, 4, 1e-6, 1e-6, iters
                    )
                    mix = (
                        mx.fast.rms_norm(
                            x.reshape(1, 1, -1).astype(mx.float32), None, 1e-6
                        )
                        @ fn.T
                    )
                    actual = mlx_hc_collapse_norm(
                        x,
                        mix,
                        base,
                        scale,
                        norm.weight,
                        eps=1e-6,
                        norm_eps=norm.eps,
                        iters=iters,
                    )
                    self.assertIsNotNone(actual)
                    expected = (norm(collapsed), post, residual)
                    mx.eval(actual, expected)
                    self.assertTrue(mx.array_equal(actual[0], expected[0]).item())
                    for got, want in zip(actual, expected):
                        error = mx.linalg.norm(got.astype(mx.float32) - want)
                        denom = mx.maximum(
                            mx.linalg.norm(want.astype(mx.float32)), 1e-6
                        )
                        self.assertLess((error / denom).item(), 0.012)
                    self.assertIsNone(
                        mlx_hc_collapse_norm(
                            x.astype(mx.float32),
                            mix,
                            base,
                            scale,
                            norm.weight,
                            eps=1e-6,
                            norm_eps=norm.eps,
                            iters=iters,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
