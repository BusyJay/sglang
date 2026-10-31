"""MLX MoE fusion preserves corrected routing and both FFN rounding sites."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxGlmMoe(CustomTestCase):
    def test_corrected_sigmoid_routing_and_ties(self):
        import mlx.core as mx
        from mlx_lm.models.kimi_linear import _group_expert_select

        from sglang.kernels.ops.moe.mlx_glm import mlx_glm_router

        mx.random.seed(20261030)
        for tied in (False, True):
            x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
            w = mx.zeros((32, 256)) if tied else mx.random.normal((32, 256)) * 0.03
            bias = mx.zeros((32,))
            expected = _group_expert_select(
                x.astype(mx.float32) @ w.T, bias, 8, 1, 1, 2.5, True, "sigmoid"
            )
            actual = mlx_glm_router(x, w, bias, top_k=8, scaling=2.5, renormalize=True)
            self.assertIsNotNone(actual)
            mx.eval(actual, expected)
            self.assertTrue(mx.array_equal(actual[0], expected[0]).item())
            self.assertLess(mx.max(mx.abs(actual[1] - expected[1])).item(), 1e-6)

    def test_affine_down_restores_dtype_before_weighted_sum(self):
        import mlx.core as mx

        from sglang.kernels.ops.moe.mlx_glm import mlx_glm_down_combine

        mx.random.seed(20261030)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                with self.subTest(dtype=dtype, bits=bits):
                    w, s, b = mx.quantize(
                        (mx.random.normal((16, 256, 512)) * 0.03).astype(dtype),
                        bits=bits,
                        group_size=64,
                    )
                    act = mx.random.normal((1, 1, 8, 512)).astype(dtype)
                    indices = mx.array([[[7, 3, 14, 0, 9, 1, 8, 2]]], mx.uint32)
                    weights = mx.softmax(mx.random.normal((1, 1, 8)), axis=-1)
                    shared = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
                    routed = mx.gather_qmm(
                        act[..., None, :],
                        w,
                        s,
                        b,
                        rhs_indices=indices,
                        transpose=True,
                        bits=bits,
                        group_size=64,
                    ).squeeze(-2)
                    routed = routed.astype(mx.bfloat16)
                    expected = (routed * weights[..., None]).sum(-2).astype(mx.bfloat16)
                    expected = expected + shared
                    actual = mlx_glm_down_combine(
                        act, indices, weights, w, s, b, shared, bits=bits, group_size=64
                    )
                    self.assertIsNotNone(actual)
                    mx.eval(actual, expected)
                    self.assertLess(mx.max(mx.abs(actual - expected)).item(), 0.04)


if __name__ == "__main__":
    unittest.main()
