"""Select from valid cache ownership and actual transfers, not cache miss ratio alone."""

import unittest

from sglang.srt.layers.moe.qwen38_cache_overlap_policy import select_prefill_cache_path
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestOverlapPolicy(CustomTestCase):
    @staticmethod
    def select(misses, tokens, arena=(), *, w4a4=True, geometry=(2560, 640, 10)):
        counts = [tokens] * 10 + [0]
        resident = {expert: expert for expert in range(10 - misses)}
        return select_prefill_cache_path(
            counts,
            resident,
            None if arena is None else set(arena),
            hidden=geometry[0],
            width=geometry[1],
            top_k=geometry[2],
            w4a4=w4a4,
        )

    def test_ready_arena_must_not_split_even_with_many_cache_misses(self):
        for misses in range(11):
            self.assertEqual(self.select(misses, 2048, range(10)), "arena")
            self.assertEqual(self.select(misses, 2048, None), "arena")

    def test_arena_resident_misses_cannot_be_counted_as_transfer_opportunity(self):
        self.assertEqual(self.select(6, 2048), "split")
        # All six cache misses are already in arena: nothing to overlap.
        self.assertEqual(self.select(6, 2048, range(4, 10)), "arena")

    def test_tiny_and_large_batches_have_different_amortization(self):
        self.assertEqual(self.select(0, 3), "cache")
        for misses in range(1, 11):
            self.assertEqual(self.select(misses, 3), "arena")
        for misses in (1, 2):
            self.assertEqual(self.select(misses, 2048), "arena")
        for misses in range(3, 10):
            self.assertEqual(self.select(misses, 2048), "split")
        self.assertEqual(self.select(10, 2048), "arena")

    def test_uncalibrated_backend_or_geometry_keeps_single_dispatch(self):
        self.assertEqual(self.select(5, 2048, w4a4=False), "arena")
        self.assertEqual(self.select(5, 2048, geometry=(1024, 256, 10)), "arena")
        # All-hit cache still needs no split or early projection on any backend.
        self.assertEqual(self.select(0, 2048, w4a4=False), "cache")


if __name__ == "__main__":
    unittest.main()
