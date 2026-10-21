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
