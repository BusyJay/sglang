"""Fused KDA must preserve both convolution history and FP32 recurrent state."""

import copy
import importlib.util
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxKda(CustomTestCase):
    def test_prefill_then_fused_decode_keeps_cache_layout_and_offsets(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            KDA,
            ModelArgs,
            NativeCache,
            mlx_kda_decode,
        )

        args = ModelArgs(
            model_type="glm5_next",
            text_config=dict(
                hidden_size=256,
                qk_rope_head_dim=0,
                rms_norm_eps=1e-6,
                linear_attn_config=dict(
                    num_heads=2,
                    head_dim=128,
                    short_conv_kernel_size=4,
                    gate_lower_bound=-5.0,
                ),
            ),
        )
        for dtype in (mx.bfloat16, mx.float16):
            mx.random.seed(20261030)
            reference = KDA(args)
            reference.set_dtype(dtype)
            reference.A_log = reference.A_log.astype(mx.float32)
            reference.dt_bias = reference.dt_bias.astype(mx.float32)
            fused = copy.deepcopy(reference)
            fused.fuse_decode = True
            reference.prepare_for_inference()
            fused.prepare_for_inference()
            caches = [NativeCache(4), NativeCache(4)]

            def checked_kernel(*a, **kw):
                result = mlx_kda_decode(*a, **kw)
                self.assertIsNotNone(result)
                return result

            for length in (1, 12, 1, 4, 8, 1):
                with self.subTest(dtype=dtype, length=length):
                    x = mx.random.normal((1, length, 256)).astype(dtype)
                    expected = reference(x, caches[0])
                    with patch(
                        "sglang.srt.hardware_backend.mlx.models.glm5_next."
                        "mlx_kda_decode",
                        side_effect=checked_kernel,
                    ) as call:
                        actual = fused(x, caches[1])
                        self.assertEqual(call.call_count, int(length <= 8))
                    mx.eval(expected, actual, [c.state for c in caches])
                    error = mx.linalg.norm(actual.astype(mx.float32) - expected)
                    denom = mx.maximum(
                        mx.linalg.norm(expected.astype(mx.float32)), 1e-6
                    )
                    self.assertLess((error / denom).item(), 0.025)
                    self.assertEqual(caches[0].offset, caches[1].offset)
                    for i in range(3):
                        self.assertTrue(
                            mx.array_equal(caches[0][i], caches[1][i]).item()
                        )
                    self.assertTrue(mx.all(mx.isfinite(caches[1][3])).item())
                    self.assertEqual(caches[1][3].dtype, mx.float32)
                    state_error = mx.linalg.norm(caches[0][3] - caches[1][3])
                    state_norm = mx.maximum(mx.linalg.norm(caches[0][3]), 1e-6)
                    self.assertLess((state_error / state_norm).item(), 0.025)

    def test_verify_snapshots_match_serial_and_resume_every_prefix(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            KDA,
            ModelArgs,
            NativeCache,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        args = ModelArgs(
            model_type="glm5_next",
            text_config=dict(
                hidden_size=256,
                qk_rope_head_dim=0,
                rms_norm_eps=1e-6,
                linear_attn_config=dict(
                    num_heads=2,
                    head_dim=128,
                    short_conv_kernel_size=4,
                    gate_lower_bound=-5.0,
                ),
            ),
        )
        for dtype in (mx.bfloat16, mx.float16):
            mx.random.seed(19)
            attn = KDA(args)
            attn.set_dtype(dtype)
            attn.A_log = attn.A_log.astype(mx.float32)
            attn.dt_bias = attn.dt_bias.astype(mx.float32)
            attn.fuse_decode = True
            attn.prepare_for_inference()
            base = NativeCache(4)
            mx.eval(
                attn(mx.random.normal((1, 12, 256)).astype(dtype), base), base.cache
            )
            for count in (2, 3, 5, 9, 17):
                with self.subTest(dtype=dtype, count=count):
                    x = mx.random.normal((1, count, 256)).astype(dtype)
                    serial, shared = clone_cache(base), clone_cache(base)
                    rows, states = [], []
                    for j in range(count):
                        rows.append(attn(x[:, j : j + 1], serial))
                        states.append(clone_cache(serial))
                    actual, snapshots = attn.verify_rows(x, shared)
                    mx.eval(actual, rows, [c.cache for c in snapshots])
                    self.assertTrue(
                        mx.array_equal(actual, mx.concatenate(rows, 1)).item()
                    )
                    for got, want in zip(snapshots, states):
                        self.assertEqual(got.offset, want.offset)
                        for a, b in zip(got.cache, want.cache):
                            self.assertTrue(mx.array_equal(a, b).item())
                        continuation = mx.random.normal((1, 1, 256)).astype(dtype)
                        self.assertTrue(
                            mx.array_equal(
                                attn(continuation, clone_cache(got)),
                                attn(continuation, clone_cache(want)),
                            ).item()
                        )


if __name__ == "__main__":
    unittest.main()
