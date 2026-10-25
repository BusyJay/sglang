# SPDX-License-Identifier: Apache-2.0
"""Routed alpha refresh must never read or overwrite unrelated global slots."""

import unittest

import torch

from sglang.kernels.ops.moe.qwen38_route_alphas import refresh_route_alphas
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestRouteAlphas(CustomTestCase):
    def test_masked_duplicates_and_global_slot_stride(self):
        for dtype in (torch.int32, torch.int64):
            for capacity in (1025, 7006):
                for tokens in (1, 3):
                    with self.subTest(dtype=dtype, capacity=capacity, tokens=tokens):
                        row = [-1, 0, 600, capacity - 1, 600, -1, 1, 1, capacity - 1, 0]
                        ids = torch.tensor([row] * tokens, device="cuda", dtype=dtype)
                        alpha1 = torch.full((capacity,), float("nan"), device="cuda")
                        alpha2 = alpha1.clone()
                        slots = ids.flatten().clamp_min(0).long().unique()
                        alpha1[slots] = (
                            torch.arange(
                                slots.numel(), device="cuda", dtype=torch.float32
                            )
                            * 0.13
                        )
                        alpha2[slots] = alpha1[slots] * 0.43
                        scale1 = torch.tensor(0.13, device="cuda")
                        scale2 = torch.tensor([1.7], device="cuda")
                        # A 16-byte-aligned second row with non-aligned capacity.
                        output = torch.full(
                            (2, (capacity + 3) // 4 * 4), 123.0, device="cuda"
                        )
                        first, second = output[:, :capacity].unbind()
                        expected = output.clone()
                        expected[0, slots] = alpha1[slots] * scale1
                        expected[1, slots] = alpha2[slots] * scale2
                        refresh_route_alphas(
                            ids, alpha1, alpha2, scale1, scale2, first, second
                        )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        # New fills may update scales between calls; no stale snapshot.
                        alpha1[slots] *= 2
                        refresh_route_alphas(
                            ids, alpha1, alpha2, scale1, scale2, first, second
                        )
                        expected[0, slots] *= 2
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_fc1_complement_preserves_cpu_placeholders_and_unrouted_alphas(self):
        from sglang.kernels.ops.moe.qwen38_route_alphas import prepare_fc1_routes

        early = torch.tensor([[2, -1, 2, -1]], dtype=torch.int32, device="cuda")
        remapped = torch.tensor([[2, 0, 2, 3]], dtype=torch.int32, device="cuda")
        late = torch.empty_like(early)
        alpha = torch.full((8,), float("nan"), device="cuda")
        alpha[2] = 3
        outputs = torch.full((2, 8), 17.0, device="cuda")
        scale = torch.tensor(0.5, device="cuda")
        prepare_fc1_routes(
            early, remapped, late, alpha, alpha, scale, scale, *outputs.unbind()
        )
        self.assertEqual(late.tolist(), [[-1, 0, -1, 3]])
        expected = torch.full_like(outputs, 17.0)
        expected[:, 2] = 1.5
        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)

    def test_resident_scatter_and_all_route_alphas_match_separate_operations(self):
        from sglang.kernels.ops.moe.qwen38_route_alphas import prepare_fc1_routes

        early = torch.tensor([[2, -1, 2, -1]], dtype=torch.int32, device="cuda")
        remapped = torch.tensor([[2, 0, 2, 3]], dtype=torch.int32, device="cuda")
        plan = torch.zeros(16, dtype=torch.int32, device="cuda")
        plan[:8] = torch.tensor([0, 2, 3, 0, 5, 0, 7, 3], device="cuda")
        table = torch.arange(8, dtype=torch.float32, device="cuda") + 1
        alpha = torch.full((8,), float("nan"), device="cuda")
        alpha[2] = 3
        outputs = torch.full((2, 8), 17.0, device="cuda")
        scale = torch.tensor(0.5, device="cuda")
        late = torch.empty_like(early)
        prepare_fc1_routes(
            early,
            remapped,
            late,
            alpha,
            alpha,
            scale,
            scale,
            *outputs.unbind(),
            resident_alphas=(plan, table, table),
        )
        self.assertEqual(late.tolist(), [[-1, 0, -1, 3]])
        expected = torch.full_like(outputs, 17.0)
        expected[:, [0, 2, 3]] = torch.tensor([3.0, 1.5, 4.0], device="cuda")
        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
        self.assertEqual(alpha[[0, 2, 3]].tolist(), [6.0, 3.0, 8.0])
        self.assertTrue(torch.isnan(alpha[[1, 4, 5, 6, 7]]).all().item())

    def test_invalid_route_geometry_is_rejected(self):
        alpha = torch.ones(8, device="cuda")
        scale = torch.ones((), device="cuda")
        ids = torch.zeros((4, 10), dtype=torch.int32, device="cuda")
        with self.assertRaises(Exception):
            refresh_route_alphas(
                ids, alpha, alpha, scale, scale, alpha.clone(), alpha.clone()
            )


if __name__ == "__main__":
    unittest.main()
