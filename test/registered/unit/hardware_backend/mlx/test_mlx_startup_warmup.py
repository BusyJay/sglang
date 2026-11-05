# SPDX-License-Identifier: Apache-2.0
"""Startup warmup must materialize lazy GPU work before returning readiness."""

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxStartupWarmup(CustomTestCase):
    def test_materializes_every_dtype_and_small_batch_variant(self):
        """A lazy failed pipeline cannot escape startup into the first request.

        Metal's compile error is observed only when its output is evaluated.
        Changing warmup to graph construction alone, or omitting BF16/pair
        variants, would allow this real device failure to escape readiness.
        """
        import mlx.core as mx
        from mlx import nn
        from sglang.srt.hardware_backend.mlx.warmup import warmup_mlx_switch_glu

        for dtype, count in (
            (mx.float32, 1),
            (mx.float32, 2),
            (mx.bfloat16, 1),
            (mx.bfloat16, 2),
        ):
            with self.subTest(dtype=dtype, count=count):
                bad = mx.fast.metal_kernel(
                    name="sglang_warmup_failure_probe",
                    input_names=["x"],
                    output_names=["out"],
                    source="sglang_deliberate_invalid_startup_shader;",
                )

                class StartupBlock(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.gate_proj = SimpleNamespace(
                            weight=mx.zeros((1, 8, 256), mx.uint8)
                        )

                    def __call__(self, x, indices, dtype=dtype, count=count, bad=bad):
                        if x.dtype == dtype and x.shape[1] == count:
                            return bad(
                                inputs=[x],
                                grid=(1, 1, 1),
                                threadgroup=(1, 1, 1),
                                output_shapes=[(1,)],
                                output_dtypes=[mx.float32],
                            )[0]
                        return x

                block = StartupBlock()
                weight = block.gate_proj.weight
                with self.assertRaisesRegex(
                    Exception, "sglang_deliberate_invalid_startup_shader"
                ):
                    warmup_mlx_switch_glu(block, picks=8)
                self.assertIs(block.gate_proj.weight, weight)


if __name__ == "__main__":
    unittest.main()
