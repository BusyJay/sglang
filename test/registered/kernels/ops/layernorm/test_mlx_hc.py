"""Four-stream mHC fusion retains Sinkhorn orientation and branch normalization."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxHc(CustomTestCase):
    def test_fp32_mix_fusion_preserves_singleton_reduction(self):
        """FP32 cancellation must preserve the high-K projection's routing inputs."""
        import mlx.core as mx

        from sglang.kernels.ops.layernorm.mlx_hc_mix import mlx_hc_rms_mix

        mx.random.seed(171)
        for dtype in (mx.float16, mx.bfloat16):
            for dim in (256, 768, 1024, 1280, 4096):
                x = mx.random.normal((1, 3, 4, dim)).astype(dtype)
                weight = mx.random.normal((24, 4 * dim))
                # Alternate signs around cancellation; don't contract the
                # normalization scale into the projection weights.
                weight[:, 1::2] = -weight[:, ::2]
                actual = mlx_hc_rms_mix(x, weight, eps=1e-5)
                flat = x.reshape(1, 3, -1).astype(mx.float32)
                expected = mx.concatenate(
                    [
                        mx.fast.rms_norm(flat[:, j : j + 1], None, 1e-5) @ weight.T
                        for j in range(3)
                    ],
                    axis=1,
                )
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_parallel_lines_preserve_serial_coefficients_and_rounding(self):
        """A lane transpose must preserve orientation, order and cast boundaries.

        Compare coefficients before 16-bit rounding as well as the complete
        fused outputs; rounded coefficients alone can hide routing changes.
        """
        import mlx.core as mx
        import numpy as np

        from sglang.kernels.ops.layernorm import mlx_hc as parallel
        from sglang.test.kernels import mlx_hc as serial

        kernels = [
            mx.fast.metal_kernel(
                name=f"test_hc_fp32_coefficients_{name}",
                input_names=["x", "mix", "base", "scale", "norm_weight", "params"],
                output_names=["normed", "post", "comb_out"],
                source=module._SOURCE.replace(
                    "T(2 * (v < 0 ? e : 1 - e))", "(2 * (v < 0 ? e : 1 - e))"
                )
                .replace("T(comb[i])", "comb[i]")
                .replace("T(values[c])", "values[c]"),
            )
            for name, module in (("serial", serial), ("parallel", parallel))
        ]
        mx.random.seed(20261031)
        for dtype in (mx.bfloat16, mx.float16):
            for dim, count, iters in (
                (128, 1, 1),
                (256, 3, 2),
                (4096, 1, 20),
                (4096, 2, 20),
                (4096, 17, 32),
            ):
                for cancelling in (False, True):
                    with self.subTest(
                        dtype=dtype,
                        dim=dim,
                        count=count,
                        iters=iters,
                        cancelling=cancelling,
                    ):
                        x = mx.random.normal((1, count, 4, dim)).astype(dtype)
                        mix = mx.random.normal((1, count, 24)) * 8
                        scale = mx.array([30.3, -0.75, 1.5])
                        base = mx.random.normal((24,))
                        if cancelling:
                            mix = mx.full((1, count, 24), 100.7)
                            products = np.float32(100.7) * np.repeat(
                                np.array(scale), [4, 4, 16]
                            )
                            base = mx.array(-products)
                            x = mx.broadcast_to(
                                mx.array([1, -1, 1, -1], dtype).reshape(1, 1, 4, 1),
                                (1, count, 4, dim),
                            )
                        norm = mx.random.uniform(low=0.1, high=2, shape=(dim,)).astype(
                            dtype
                        )
                        inputs = [
                            x,
                            mix,
                            base,
                            scale,
                            norm,
                            mx.array([1e-6, 1e-5], mx.float32),
                        ]
                        coefficients = [
                            kernel(
                                inputs=inputs,
                                template=[
                                    ("T", dtype),
                                    ("D", dim),
                                    ("ITERS", iters),
                                    ("THREADS", dim // 4),
                                ],
                                grid=(dim // 4, count, 1),
                                threadgroup=(dim // 4, 1, 1),
                                output_shapes=[
                                    (1, count, dim),
                                    (1, count, 4),
                                    (1, count, 4, 4),
                                ],
                                output_dtypes=[dtype, mx.float32, mx.float32],
                            )
                            for kernel in kernels
                        ]
                        outputs = [
                            module.mlx_hc_collapse_norm(
                                x,
                                mix,
                                base,
                                scale,
                                norm,
                                eps=1e-6,
                                norm_eps=1e-5,
                                iters=iters,
                            )
                            for module in (serial, parallel)
                        ]
                        mx.eval(coefficients, outputs)
                        for pair in (coefficients, outputs):
                            for got, want in zip(pair[1], pair[0]):
                                self.assertTrue(mx.all(mx.isfinite(got)).item())
                                self.assertTrue(mx.array_equal(got, want).item())

    def test_multirow_collapse_preserves_independent_sinkhorn(self):
        import mlx.core as mx

        from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm

        mx.random.seed(82)
        for count in (2, 3, 5, 9, 17):
            for dtype in (mx.float16, mx.bfloat16):
                with self.subTest(count=count, dtype=dtype):
                    x = mx.random.normal((1, count, 4, 256)).astype(dtype)
                    mix = mx.random.normal((1, count, 24))
                    base = mx.random.normal((24,))
                    scale = mx.array([0.5, 1.5, 0.75])
                    norm = mx.random.uniform(low=0.1, high=2, shape=(256,)).astype(
                        dtype
                    )

                    def run(x, mix):
                        return mlx_hc_collapse_norm(
                            x,
                            mix,
                            base,
                            scale,
                            norm,
                            eps=1e-6,
                            norm_eps=1e-5,
                            iters=20,
                        )

                    actual = run(x, mix)
                    self.assertIsNotNone(actual)
                    rows = [
                        run(x[:, j : j + 1], mix[:, j : j + 1]) for j in range(count)
                    ]
                    for i in range(3):
                        expected = mx.concatenate([r[i] for r in rows], 1)
                        self.assertTrue(mx.array_equal(actual[i], expected).item())

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
