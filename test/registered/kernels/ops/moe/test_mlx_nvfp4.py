# SPDX-License-Identifier: Apache-2.0
"""Serialized NVFP4 global calibration, routed selection and dense bias."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxNVFP4(CustomTestCase):
    def test_global_scale_and_selected_expert(self):
        import mlx.core as mx

        from sglang.kernels.ops.moe.mlx_nvfp4 import mlx_nvfp4_routed
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import (
            NVFP4Linear,
            NVFP4SwitchLinear,
        )

        # E2M1 2 = 1, E4M3 56 = 1: the literal global is a multiplier.
        w = mx.full((2, 128, 128), 0x22, mx.uint8)
        sc = mx.full((2, 128, 16), 56, mx.uint8)
        gl = mx.array([0.25, 2.0], mx.float32)
        indices = mx.array([[[0, 1], [1, 0]]], mx.uint32)
        x = mx.ones((1, 2, 256), mx.bfloat16)
        p = NVFP4SwitchLinear(w, sc, gl)
        out = p(x[..., None, None, :], indices)
        want = mx.broadcast_to(
            mx.array([[[64, 512], [512, 64]]], mx.bfloat16)[..., None, None], out.shape
        )
        self.assertTrue(mx.array_equal(out, want).item())
        diagnostic = mlx_nvfp4_routed(x, w, sc, gl, indices)
        self.assertTrue(mx.array_equal(diagnostic, want).item())
        dense = NVFP4Linear(w[0], sc[0], gl[0], bias=mx.ones((128,), mx.bfloat16))
        self.assertTrue(mx.all(dense(x) == 65).item())

    def test_dense_fp8_preserves_bias(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

        p = FP8Linear(
            mx.full((128, 256), 56, mx.uint8),
            mx.full((128, 8), 127, mx.uint8),
            block_size=32,
            activation_mode="a16",
            bias=mx.arange(128, dtype=mx.bfloat16),
        )
        out = p(mx.ones((1, 5, 256), mx.bfloat16))
        want = mx.broadcast_to(256 + mx.arange(128, dtype=mx.bfloat16), out.shape)
        self.assertTrue(mx.array_equal(out, want).item())


if __name__ == "__main__":
    unittest.main()
