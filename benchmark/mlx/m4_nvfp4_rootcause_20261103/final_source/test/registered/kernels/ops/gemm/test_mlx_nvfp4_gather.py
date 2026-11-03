# SPDX-License-Identifier: Apache-2.0
"""NVFP4 expert sharing must preserve native rounding and pick ordering."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxNvfp4Gather(CustomTestCase):
    def test_shared_experts_native_rounding_and_per_pick_inputs(self):
        """Contracting scale multiplication changes multi-block native results.

        Covers odd/even repeated picks, within-token duplicates, stale worklist
        reuse and global factors, including noncontiguous token activations.
        """
        import mlx.core as mx
        import numpy as np

        from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
        from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist

        rng = np.random.default_rng(9011)
        for count, k in ((1, 512), (4, 1024), (8, 4096)):
            e, n, picks = 5, 24, 3
            weight = mx.array(rng.integers(0, 256, (e, n, k // 2), dtype=np.uint8))
            # Include subnormal scale codes and the largest finite E4M3 value.
            scales = mx.array(rng.choice(np.array([0, 1, 7, 8, 40, 56, 126], np.uint8),
                                         size=(e, n, k // 16)))
            global_scale = mx.array([0.0, 1.0, 2.688, 3.731, 0.331], mx.float32)
            for shift in (0, 1):
                ids = mx.array([[(i + shift) % e, 1, 1] for i in range(count)], mx.uint32)
                work = mlx_expert_pair_worklist(ids)
                partners, sizes = [np.array(v) for v in work]
                flat = np.array(ids).ravel()
                covered = []
                for pick, size in enumerate(sizes):
                    if not size:
                        continue
                    covered.append(pick)
                    if size == 2:
                        covered.append(int(partners[pick]))
                        self.assertEqual(flat[pick], flat[partners[pick]])
                self.assertEqual(sorted(covered), list(range(ids.size)))
                for per_pick in (False, True):
                    shape = (count, picks if per_pick else 1, 1, k * 2)
                    original = mx.array(rng.normal(size=shape).astype(np.float32))
                    for dtype in (mx.float32, mx.bfloat16):
                        x = original[..., ::2].astype(dtype)
                        expected = mx.gather_qmm(
                            x, weight.view(mx.uint32), scales, rhs_indices=ids,
                            mode="nvfp4", group_size=16, bits=4, global_scale=global_scale,
                        )
                        for explicit in (None, work):
                            actual = mlx_nvfp4_gather_rows(
                                x, weight, scales, global_scale, ids, worklist=explicit,
                            )
                            mx.eval(actual, expected)
                            self.assertTrue(mx.array_equal(actual, expected).item(),
                                            (count, k, shift, per_pick, dtype, explicit is None))

    def test_fp4_codes_and_finite_e4m3_scale_values(self):
        """Sharing cannot alter nibble sign, subnormal decode or global scaling."""
        import mlx.core as mx
        import numpy as np

        from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows

        codes = np.arange(16, dtype=np.uint8)
        fp4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
        fp4 = np.concatenate((fp4, -fp4))
        weight = np.zeros((1, 16, 256), np.uint8)
        weight[0, :, 0] = codes
        for scale_code, scale in ((1, 2**-9), (8, 2**-6), (56, 1), (126, 448)):
            scales = mx.full((1, 16, 32), scale_code, mx.uint8)
            x = np.zeros((3, 1, 1, 512), np.float32)
            x[:, 0, 0, 0] = [1, 2, -4]
            y = mlx_nvfp4_gather_rows(mx.array(x), mx.array(weight), scales,
                                     mx.array([2688.0]), mx.zeros((3, 1), mx.uint32))
            expected = x[:, 0, 0, 0, None] * fp4[None] * scale
            self.assertTrue(np.array_equal(np.array(y).reshape(3, 16), expected))


if __name__ == "__main__":
    unittest.main()
