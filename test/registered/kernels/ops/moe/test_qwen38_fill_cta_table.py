"""Candidate CTA buckets preserve all bytes and untouched slots for 0..30 misses."""

import unittest

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFillCtaTable(CustomTestCase):
    def test_all_miss_counts_and_vector_tail(self):
        module = _module()
        # Both vectorized and byte fallback paths; distinct tensor/expert bytes.
        for size in (4096, 4103):
            source = (
                torch.arange(40 * size, device="cuda").to(torch.uint8).view(40, size)
            )
            source[:, 0] = torch.arange(40, device="cuda", dtype=torch.uint8)
            baseline = torch.empty(32, size, device="cuda", dtype=torch.uint8)
            candidate = torch.empty_like(baseline)
            failed = torch.zeros(1, device="cuda", dtype=torch.int32)
            for misses in range(31):
                with self.subTest(size=size, misses=misses):
                    plan = torch.zeros(68, dtype=torch.int32, device="cuda")
                    slots = torch.arange(misses - 1, -1, -1, device="cuda")
                    experts = torch.arange(5, 5 + misses, device="cuda")
                    plan[1] = misses
                    if misses:
                        plan[4 : 4 + 2 * misses] = (
                            torch.stack((experts, slots), 1).flatten().int()
                        )
                    for target, policy in ((baseline, 0), (candidate, -1)):
                        target.fill_(255)
                        layout = torch.tensor(
                            [[target.data_ptr(), size, source.data_ptr()]],
                            device="cuda",
                            dtype=torch.int64,
                        )
                        module.mailbox_fill_grid(
                            plan, layout, 30, failed.data_ptr(), policy
                        )
                    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
                    if misses:
                        torch.testing.assert_close(
                            candidate[slots], source[experts], rtol=0, atol=0
                        )
                    self.assertTrue(torch.all(candidate[misses:] == 255).item())


if __name__ == "__main__":
    unittest.main()
