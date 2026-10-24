"""Physical-page loan accounting must preserve runtime headroom after rebasing."""

import unittest

from sglang.srt.layers.moe.qwen38_auto_capacity import (
    auto_kv_stages,
    forecast_loan_budgets,
    plan_auto_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestAutoCapacity(CustomTestCase):
    def test_initial_backing_not_base_delta_binds_capacity(self):
        # The two differently sized row tensors cross physical pages at
        # different slot counts. Brute force is independent of binary search.
        def physical(n):
            return ((n * 3 + 15) // 16 + (n * 5 + 15) // 16) * 16 + 7

        loans = [47, 19, 0]
        current = forecast_loan_budgets(4, 40, loans, physical)[0]
        for free in range(5, 160):
            base, budgets = plan_auto_capacity(4, 40, current, loans, physical, free, 5)

            def stage_for(n, loan):
                return max(i for i in range(n, 41) if physical(i) - physical(n) <= loan)

            valid = [
                n
                for n in range(4, 41)
                if physical(stage_for(n, loans[0])) - physical(current) <= free - 5
            ]
            self.assertEqual(base, max(valid))
            self.assertEqual(budgets, [stage_for(base, loan) for loan in loans])
            self.assertLessEqual(physical(budgets[0]) - physical(current), free - 5)

    def test_insufficient_minimum_headroom_fails_before_growth(self):
        with self.assertRaisesRegex(ValueError, "Reduce KV context"):
            plan_auto_capacity(4, 40, 8, [32, 16, 0], lambda n: n * 8, 63, 64)

    def test_context_derived_stages_keep_full_prefill_reserve(self):
        self.assertEqual(auto_kv_stages(220000, 8192), [80000, 160000, 220032])
        for context in (32768, 64000, 128000, 170000, 220000):
            stages = auto_kv_stages(context, 8192)
            self.assertTrue(2 * 8192 < stages[0] < stages[1] < stages[2])
            self.assertTrue(all(n % 64 == 0 for n in stages))
            self.assertGreaterEqual(stages[-1], context)
            self.assertLess(stages[-1] - context, 64)


if __name__ == "__main__":
    unittest.main()
