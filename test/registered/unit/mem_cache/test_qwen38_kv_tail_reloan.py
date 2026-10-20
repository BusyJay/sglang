# SPDX-License-Identifier: Apache-2.0
"""KV tail re-loan after a long->short switch.

A long request leaves cached prefixes physically spread across the pool. The
shrink that hands the KV tail back to the expert pool unmaps that tail, so it
needs every page above the next lower stage free; cold cached prefixes can pin
it indefinitely even though the working set dropped two stages down. These
cases use the real paged allocator at the production stage geometry.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.qwen38_kv_stages import OneWayKVStages, StagedKVLoan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, stage="base-a", runner_config="cpu")


class TestTailReloan(CustomTestCase):
    PAGE = 64
    CAPS = [80000, 160000, 220032]
    BACKED = 220032 // 64  # pages backed at the top stage
    USED_PAGES = 98_048 // 64  # prod-like 0.44 usage, mostly cached prefixes

    def _make_loan(self, in_use, hook=None, tail_evict_tokens=None):
        """Real allocator parked at the top stage while holding `in_use` pages."""
        import torch

        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        layout = []
        runtime = SimpleNamespace(
            kv_cache_stages=self.CAPS,
            expert_loan_budgets=[7045, 6681, 6426],
            pending_kv_stage=None,
            staged_trim_enqueued=False,
            kv_reborrow_probe=False,
            kv_tail_evict_tokens=tail_evict_tokens,
            staged_pool=SimpleNamespace(
                release_stage=lambda i: None, grow_stage=lambda i: None
            ),
            grow_staged_layout=lambda target: layout.append(target),
        )
        pool = SimpleNamespace(
            _post_capture_owner=SimpleNamespace(
                finalize=lambda n: None, shrink_capacity=lambda n: None
            )
        )
        allocator = PagedTokenToKVPoolAllocator(
            220032, self.PAGE, torch.float32, "cpu", None, False
        )
        allocator.staged_loan = StagedKVLoan(allocator, pool, runtime)
        allocator.backed_num_pages = self.BACKED
        allocator.free_pages = torch.tensor(
            [p for p in range(1, self.BACKED + 1) if p not in in_use],
            dtype=torch.int64,
        )
        loan = allocator.staged_loan
        # Park where a long request would have left it.
        loan.policy.index = 2
        loan.reserve = 8192  # high-water mark from the first 8192-token prefill
        loan.evict_hook = hook
        self._layout = layout
        return loan

    def _short_traffic_probe(self, loan):
        """One decode batch after the long request ended, then its result."""
        loan.before_forward(3)
        self.assertTrue(loan.runtime.kv_reborrow_probe)
        loan.free_mask.zero_()
        pages = loan.allocator.get_all_free_pages()
        if pages.numel():
            loan.free_mask[pages] = True
        loan.free_snapshot.copy_(loan.free_mask)
        loan.probe = SimpleNamespace(query=lambda: True)
        loan._reborrow()

    def _scattered(self, used_pages, seed=11):
        import random

        random.seed(seed)
        return set(random.sample(range(1, self.BACKED + 1), used_pages))

    def test_cold_prefix_pin_without_hook_keeps_the_previous_no_op(self):
        loan = self._make_loan(self._scattered(self.USED_PAGES))
        self._short_traffic_probe(loan)
        self.assertEqual(loan.policy.index, 2)
        self.assertEqual(self._layout, [])
        self.assertEqual(loan.tail_evict_tokens, 0)

    def test_cold_prefix_pin_is_evicted_so_the_tail_returns_to_experts(self):
        import torch

        in_use = self._scattered(self.USED_PAGES)
        calls = []

        def hook(tokens):
            calls.append(tokens)
            # The coldest cached prefixes are the topmost pinned pages.
            freed = sorted((page for page in in_use), reverse=True)[
                : tokens // self.PAGE
            ]
            in_use.difference_update(freed)
            loan.allocator.free_pages = torch.tensor(
                sorted(loan.allocator.free_pages.tolist() + freed), dtype=torch.int64
            )

        loan = self._make_loan(in_use, hook=hook)
        self._short_traffic_probe(loan)
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0], loan.tail_evict_limit_tokens)
        self.assertEqual(loan.policy.index, 1)
        self.assertEqual(self._layout, [1])
        self.assertEqual(loan.tail_evict_attempts, 0)

    def test_clear_tail_reloans_without_touching_the_prefix_cache(self):
        # Working set clustered below the margin: the re-loan needs no eviction,
        # so prefix cache is never dropped for a shrink that is already possible.
        in_use = set(range(1, (self.CAPS[1] - 2 * 8192) // self.PAGE))
        self.assertTrue(in_use)
        calls = []
        loan = self._make_loan(in_use, hook=lambda tokens: calls.append(tokens))
        self._short_traffic_probe(loan)
        self.assertEqual(calls, [])
        self.assertEqual(loan.policy.index, 1)
        self.assertEqual(self._layout, [1])

    def test_eviction_budget_stops_retrying_when_the_tail_stays_pinned(self):
        # Locked or in-flight pages above the boundary cannot be evicted.
        calls = []
        loan = self._make_loan(
            self._scattered(self.USED_PAGES), hook=lambda tokens: calls.append(tokens)
        )
        for _ in range(loan.tail_evict_attempts_allowed + 2):
            self._short_traffic_probe(loan)
        self.assertEqual(len(calls), loan.tail_evict_attempts_allowed)
        self.assertEqual(loan.policy.index, 2)

    def test_zero_budget_disables_tail_eviction(self):
        calls = []
        loan = self._make_loan(
            self._scattered(self.USED_PAGES),
            hook=lambda tokens: calls.append(tokens),
            tail_evict_tokens=0,
        )
        self._short_traffic_probe(loan)
        self.assertEqual(calls, [])
        self.assertEqual(loan.policy.index, 2)

    def test_loan_without_stages_keeps_one_way_policy(self):
        # A plain shrink decision still never returns the loan on its own.
        returns = []
        policy = OneWayKVStages(self.CAPS, returns.append)
        self.assertEqual(policy.prepare(0, 1), self.CAPS[0])
        self.assertEqual(returns, [])


if __name__ == "__main__":
    unittest.main()
