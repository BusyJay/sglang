# SPDX-License-Identifier: Apache-2.0
"""Complementary native FC1 tasks preserve full-MoE math across graph replays."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "B12X requires SM12x",
)
class TestNativeFc1Overlap(CustomTestCase):
    def test_complementary_tasks_preserve_fp32_sum_and_refresh_graph_inputs(self):
        torch.manual_seed(73)
        device, e, h, i, topk = torch.device("cuda"), 16, 2560, 640, 10
        tensors = dict(
            w13_weight=torch.randint(
                0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device=device
            ),
            w2_weight=torch.randint(
                0, 256, (e, h, i // 2), dtype=torch.uint8, device=device
            ),
            w13_blockscale_swizzled=torch.full(
                (e, 1280, 160), 0.5, dtype=torch.float8_e4m3fn, device=device
            ),
            w2_blockscale_swizzled=torch.full(
                (e, 2560, 40), 0.5, dtype=torch.float8_e4m3fn, device=device
            ),
            g1_alphas=torch.linspace(0.01, 0.02, e, device=device),
            g2_alphas=torch.linspace(0.02, 0.01, e, device=device),
            w13_input_scale_quant=torch.tensor(0.7, device=device),
            w2_input_scale_quant=torch.tensor(1.1, device=device),
        )
        for resident in (False, True):
            specs = {}
            adapter = CachedB12xW4A16(
                device,
                h,
                i,
                e,
                enable_fp32=True,
                enable_fc1_overlap=True,
                scale_encoding=specs,
            )
            plan = torch.zeros(4 + 2 * e, dtype=torch.int32, device=device)
            resident_tables = (
                plan,
                tensors["g1_alphas"].clone(),
                tensors["g2_alphas"].clone(),
            )
            options = {"resident_alphas": resident_tables} if resident else {}
            for tokens in (1, 2, 3):
                x = torch.empty(tokens, h, device=device, dtype=torch.bfloat16)
                ids = torch.arange(topk, device=device, dtype=torch.int32).repeat(
                    tokens, 1
                )
                weights = torch.empty(tokens, topk, device=device)
                # Old plans must not scatter into a newly bound cache at warmup.
                plan[1] = 1
                plan[4:6] = torch.tensor([8, 2], device=device)
                old_alpha = tensors["g1_alphas"].clone()
                # New bindings must capture without a device-wide host wait.
                with patch.object(
                    torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("FC1 binding synchronized the device"),
                ):
                    stage = adapter.fc1_stage(x, ids, weights, tensors, **options)
                torch.testing.assert_close(
                    tensors["g1_alphas"], old_alpha, rtol=0, atol=0
                )
                plan.zero_()
                for misses in (0, 1, 3, 10):
                    for cpu in (False, True):
                        with self.subTest(
                            resident=resident, tokens=tokens, misses=misses, cpu=cpu
                        ):
                            x.copy_(torch.randn_like(x) * 0.1)
                            weights.copy_(torch.rand_like(weights))
                            weights.div_(weights.sum(-1, keepdim=True))
                            lookup = torch.arange(e, device=device, dtype=torch.int32)
                            if misses:
                                lookup[topk - misses : topk] = -1
                            remapped = ids.clone()
                            if cpu:
                                # A CPU route retains a safe, initialized GPU slot
                                # with zero weight, as required by the native ABI.
                                lookup[topk - 1] = -1
                                remapped[:, -1] = 0
                                weights[:, -1] = 0
                            cache_partition_routes(
                                ids, lookup, e, out=(stage.early, stage.arena)
                            )
                            stage.remapped.copy_(remapped)
                            expected = adapter.run(
                                x,
                                remapped,
                                weights,
                                tensors,
                                output_dtype=torch.float32,
                            ).clone()
                            stage.begin(x)
                            actual = stage.finish()
                            self.assertTrue(torch.isfinite(actual).all().item())
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            # Stage identity is stable while activation/route data
                            # changes; graph replay must never retain old values.
                            self.assertIs(
                                stage,
                                adapter.fc1_stage(x, ids, weights, tensors, **options),
                            )


if __name__ == "__main__":
    unittest.main()
