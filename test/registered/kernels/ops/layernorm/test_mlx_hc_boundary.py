# SPDX-License-Identifier: Apache-2.0
"""Cross mHC fusion must preserve both residual rounding and next projection."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxHcBoundary(CustomTestCase):
    def test_residual_and_next_mix_preserve_independent_rows(self):
        import mlx.core as mx
        from sglang.kernels.ops.layernorm.mlx_hc_boundary import mlx_hc_residual_rms_mix
        from sglang.kernels.ops.layernorm.mlx_hc_mix import mlx_hc_rms_mix

        @mx.compile
        def residual(r, p, b):
            return r + p[..., None] * b[..., None, :]

        for dtype in (mx.float16, mx.bfloat16):
            for count, dim in (
                (1, 256),
                (3, 512),
                (1, 4096),
                (4, 4096),
                (8, 4096),
                (17, 256),
            ):
                with self.subTest(dtype=dtype, count=count, dim=dim):
                    mx.random.seed(count + dim)
                    r = mx.random.normal((1, count, 4, dim)).astype(dtype)
                    p = (2 * mx.random.normal((1, count, 4))).astype(dtype)
                    b = mx.random.normal((1, count, dim)).astype(dtype)
                    # Cancellation exposes loss of the BF16 product boundary.
                    r = mx.where(
                        mx.arange(dim)[None, None, None, :] % 7 == 0,
                        -p[..., None] * b[..., None, :],
                        r,
                    )
                    for wd in (mx.bfloat16, mx.float32):
                        w = mx.random.normal((24, 4 * dim)).astype(wd)
                        want = residual(r, p, b)
                        want_mix = mlx_hc_rms_mix(want, w, eps=1e-5)
                        got = mlx_hc_residual_rms_mix(r, p, b, w, eps=1e-5)
                        mx.eval(want, want_mix, got)
                        self.assertTrue(mx.array_equal(got[0], want).item())
                        self.assertTrue(mx.array_equal(got[1], want_mix).item())

    def test_fp32_norm_weight_retains_original_epilogue(self):
        import mlx.core as mx
        from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm
        from sglang.kernels.ops.layernorm.mlx_hc_mix import mlx_hc_rms_mix
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _hyper_connection

        for count in (1, 4, 8):
            mx.random.seed(count)
            x = mx.random.normal((1, count, 4, 512)).astype(mx.bfloat16)
            fn = mx.random.normal((24, 2048)).astype(mx.bfloat16) * 0.01
            base = mx.random.normal((24,)).astype(mx.float32)
            scale = mx.ones((3,), mx.float32)
            weight = mx.random.normal((512,)).astype(mx.float32)
            mix = mlx_hc_rms_mix(x, fn, eps=1e-5)
            got = mlx_hc_collapse_norm(
                x, mix, base, scale, weight, eps=1e-6, norm_eps=1e-5, iters=5
            )
            parts = [
                _hyper_connection(x[:, j : j + 1], fn, base, scale, 4, 1e-5, 1e-6, 5)
                for j in range(count)
            ]
            want = (
                mx.concatenate(
                    [mx.fast.rms_norm(v[0], weight, 1e-5) for v in parts], 1
                ),
                mx.concatenate([v[1] for v in parts], 1),
                mx.concatenate([v[2] for v in parts], 1),
            )
            mx.eval(got, want)
            for a, b in zip(got, want):
                self.assertTrue(mx.array_equal(a, b).item())


if __name__ == "__main__":
    unittest.main()
