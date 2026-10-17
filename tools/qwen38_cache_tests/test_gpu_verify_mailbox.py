"""Three-token verify plans must match generic routing through eviction/hits."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class VerifyPlannerTests(unittest.TestCase):
    def test_thirty_routes_match_reference_and_bound_samples(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module, cache_plan, cache_route

        def state():
            return [
                torch.zeros(65, dtype=torch.int32, device="cuda"),
                torch.full((64,), -1, dtype=torch.int32, device="cuda"),
                torch.full((32,), -1, dtype=torch.int32, device="cuda"),
                torch.zeros(32, dtype=torch.float32, device="cuda"),
                torch.zeros(32, dtype=torch.int64, device="cuda"),
                torch.zeros(32, dtype=torch.int64, device="cuda"),
                torch.zeros(1, dtype=torch.int64, device="cuda"),
                torch.zeros(4 + 4 * 32, dtype=torch.int32, device="cuda"),
            ]

        for policy in (0, 1):
            actual, expected = state(), state()
            stats = torch.zeros(4, dtype=torch.int64, device="cuda")
            for step in range(16):
                # Alternate disjoint sets, repeated sets and duplicate/padded routes.
                ids = ((torch.arange(30, device="cuda") + (step // 2) * 19) % 64).int().view(3, 10)
                if step % 4 == 3:
                    ids[1].copy_(ids[0])
                    ids[2, -1] = -1
                output, reference = torch.empty_like(ids), torch.empty_like(ids)
                sample = torch.full((35,), -777, dtype=torch.int32, device="cuda")
                _module().decode_plan_parallel(
                    ids.view(1, -1), output.view(1, -1), *actual,
                    0.95, policy, 0.7, stats.data_ptr(), sample.data_ptr() + 4, True,
                )
                cache_route(ids, expected[1], reference, expected[0])
                cache_plan(*expected, policy=policy)
                cache_route(ids, expected[1], reference, expected[0])
                self.assertTrue(torch.equal(output, reference))
                for a, b in zip(actual[:-1], expected[:-1]):
                    self.assertTrue(torch.equal(a, b), (policy, step))
                a, b = actual[-1].cpu(), expected[-1].cpu()
                self.assertTrue(torch.equal(a[:4], b[:4]))
                misses, active = int(a[1]), int(a[2])
                self.assertTrue(torch.equal(a[4:4+2*misses], b[4:4+2*misses]))
                self.assertTrue(torch.equal(a[68:68+active], b[68:68+active]))
                self.assertTrue(torch.equal(a[100:100+active], b[100:100+active]))
                sample = sample.cpu()
                self.assertEqual((int(sample[0]), int(sample[-1])), (-777, -777))
                self.assertEqual(int(sample[1]), active)
                self.assertEqual(sample[2:2+active].tolist(), sorted(set(ids[ids >= 0].tolist())))
