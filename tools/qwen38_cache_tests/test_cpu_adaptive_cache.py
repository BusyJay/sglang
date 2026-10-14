import unittest
import random

from ._load_source import load_source

load_source(
    "sglang.srt.layers.moe.qwen38_expert_cache",
    "python/sglang/srt/layers/moe/qwen38_expert_cache.py",
)
adaptive = load_source(
    "qwen38_cache_adaptive_test",
    "python/sglang/srt/layers/moe/qwen38_cache_adaptive.py",
)


class TestAdaptiveCache(unittest.TestCase):
    def test_relocation_preserves_paths_and_cycles(self):
        values = list("abcdefghi")
        original = list(values)
        scratch = [None]
        moves = {0: 2, 2: 4, 4: 0, 1: 6, 6: 7, 3: 3}

        def move(src, dst):
            values[dst] = values[src]

        adaptive.relocate_slots(
            moves,
            move,
            lambda src: scratch.__setitem__(0, values[src]),
            lambda dst: values.__setitem__(dst, scratch[0]),
        )
        for src, dst in moves.items():
            self.assertEqual(values[dst], original[src])

    def test_relocation_random_injective_maps(self):
        rng = random.Random(17)
        for _ in range(100):
            size = 12
            count = rng.randrange(size + 1)
            sources = rng.sample(range(size), count)
            targets = rng.sample(range(size), count)
            moves = dict(zip(sources, targets))
            values = list(range(size))
            scratch = [None]
            adaptive.relocate_slots(
                moves,
                lambda src, dst: values.__setitem__(dst, values[src]),
                lambda src: scratch.__setitem__(0, values[src]),
                lambda dst: values.__setitem__(dst, scratch[0]),
            )
            for src, dst in moves.items():
                self.assertEqual(values[dst], src)

    def test_fixed_budget_and_hotspot_migration(self):
        controller = adaptive.AdaptiveLayerAllocator(
            [2, 2],
            8,
            candidates=[1, 2, 3],
            window=16,
            min_gain=0,
            cooldown_windows=0,
        )
        proposal = None
        for step in range(16):
            controller.observe(0, [step % 3])
            proposal = controller.observe(1, [7]) or proposal
        self.assertEqual(proposal, [3, 1])
        self.assertEqual(sum(proposal), 4)
        controller.applied(proposal)

        proposal = None
        for step in range(16):
            controller.observe(0, [0])
            proposal = controller.observe(1, [5 + step % 3]) or proposal
        self.assertEqual(proposal, [1, 3])

    def test_insufficient_gain_does_not_repartition(self):
        controller = adaptive.AdaptiveLayerAllocator(
            [2, 2],
            8,
            candidates=[1, 2, 3],
            window=4,
            min_gain=0,
            cooldown_windows=0,
        )
        for _ in range(4):
            controller.observe(0, [0])
            self.assertIsNone(controller.observe(1, [1]))

    def test_default_candidates_keep_cutlass_aligned(self):
        controller = adaptive.AdaptiveLayerAllocator([112] * 2, 512, min_slots=10)
        self.assertEqual(controller.candidates, (64, 80, 96, 112, 128))

    def test_dynamic_programming_handles_nonconcave_curves(self):
        self.assertEqual(
            adaptive.allocate_slots([{1: 10, 2: 9, 3: 1}, {1: 8, 2: 1, 3: 0}], 4),
            ([3, 1], 9),
        )

    def test_residual_slots_are_all_assigned(self):
        layout, cost = adaptive.allocate_slots([{8: 0}] * 5, 42, max_capacity=16)
        self.assertEqual(layout, [9, 9, 8, 8, 8])
        self.assertEqual(cost, 0)

    def test_equal_cost_prefers_more_measured_slots(self):
        layout, _ = adaptive.allocate_slots([{1: 0, 2: 0}] * 2, 4)
        self.assertEqual(layout, [2, 2])

    def test_residual_slots_respect_expert_limit(self):
        layout, cost = adaptive.allocate_slots(
            [{2: 0}, {1: 0}],
            4,
            max_capacity=2,
        )
        self.assertEqual(layout, [2, 2])
        with self.assertRaises(ValueError):
            adaptive.allocate_slots([{2: 0}] * 2, 5, max_capacity=2)

    def test_applied_residual_capacities_can_be_observed_again(self):
        controller = adaptive.AdaptiveLayerAllocator(
            [8] * 5,
            16,
            candidates=[8],
            budget=42,
            window=2,
        )
        controller.applied([9, 9, 8, 8, 8])
        for _ in range(2):
            for layer in range(5):
                controller.observe(layer, [layer])
        self.assertEqual(sum(controller.last_evaluation["proposed_slots"]), 42)
        self.assertIn(9, controller.candidates)


if __name__ == "__main__":
    unittest.main()
