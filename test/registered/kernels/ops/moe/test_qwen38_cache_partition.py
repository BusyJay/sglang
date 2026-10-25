"""Borrowed tails and decode plan mutations must not become early cache reads."""

import unittest

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import _module, cache_partition_routes
from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCachePartition(CustomTestCase):
    def test_partition_excludes_borrowed_tail_and_preserves_padding(self):
        ids = torch.tensor([[4, 1, 3, -1, 8, -2]], dtype=torch.int32, device="cuda")
        lookup = torch.tensor(
            [-1, 3, -1, 2, 1, -1, -1, -1], dtype=torch.int32, device="cuda"
        )
        for limit in (0, 2, 4):
            cache, arena = cache_partition_routes(ids, lookup, limit)
            hits = {4: 1} if limit == 2 else {4: 1, 1: 3, 3: 2} if limit == 4 else {}
            self.assertEqual(
                cache.tolist(), [[hits.get(i, -1) for i in ids[0].tolist()]]
            )
            self.assertEqual(
                arena.tolist(),
                [[i if 0 <= i < 8 and i not in hits else -1 for i in ids[0].tolist()]],
            )

    def test_global_slots_are_independent_of_expert_id_range(self):
        # Prefill of a global cache used to fail when slot_limit exceeded the
        # layer's expert count, before the first model request could complete.
        ids = torch.tensor([[0, 1, 2, 3, -1, 4]], dtype=torch.int32, device="cuda")
        lookup = torch.tensor([1, 600, 1024, 2048], dtype=torch.int32, device="cuda")
        for limit, expected in (
            (601, [[1, 600, -1, -1, -1, -1]]),
            (2049, [[1, 600, 1024, 2048, -1, -1]]),
        ):
            with self.subTest(limit=limit):
                cache, arena = cache_partition_routes(ids, lookup, limit)
                self.assertEqual(cache.tolist(), expected)
                expected_arena = torch.where(
                    (ids >= 0) & (ids < 4) & (cache < 0), ids, -1
                )
                torch.testing.assert_close(arena, expected_arena)

    def test_warp_route_union_preserves_serial_policy_state(self):
        """Sorted deduplication must preserve eviction ties and LFU ages, even
        when lane 31 is unique, routes repeat, or every route is masked.
        Compare against separate histogram/serial-policy/remap kernels.
        """
        module = _module()
        fields = (
            "counts",
            "lookup",
            "keys",
            "frequency",
            "ages",
            "generation",
            "clock",
            "plan",
        )
        rows = (
            # Interleave resident and missing IDs in the sorted union.
            list(range(32)),
            list(range(511, 479, -1)),
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27] * 3,
            [-1] * 32,
            [3, -1, 3, 400, 0, 400, 511, 6, -1, 42] * 3,
            [511],
        )
        for slots in (32, 33, 129, 147):
            for policy in (0, 1):
                for parallel_miss in (0, 1, 2):
                    with self.subTest(
                        slots=slots, policy=policy, parallel_miss=parallel_miss
                    ):
                        states = []
                        for _ in range(2):
                            buffers = AdaptiveMetadataBuffers(
                                1, 512, slots, torch.device("cuda")
                            )
                            state = buffers.reset(0, slots)
                            buffers.restore(
                                0,
                                state,
                                [
                                    # High age bits must survive warp shuffles;
                                    # tied keys still prefer the smaller slot.
                                    (
                                        3 * e,
                                        float(e % 5),
                                        (1 << 40) + (e % 3) * (1 << 32) + e // 3,
                                    )
                                    for e in range(slots // 2)
                                ],
                            )
                            states.append(state)
                        actual, expected = states
                        for row in rows:
                            ids = torch.tensor([row], dtype=torch.int32, device="cuda")
                            output, reference = (
                                torch.empty_like(ids),
                                torch.empty_like(ids),
                            )
                            for state in states:
                                state["plan"].zero_()
                            module.decode_plan_parallel(
                                ids,
                                output,
                                *(actual[k] for k in fields),
                                0.95,
                                policy,
                                0.7,
                                0,
                                0,
                                parallel_miss,
                            )
                            module.route(
                                ids, expected["lookup"], reference, expected["counts"]
                            )
                            module.plan(
                                *(expected[k] for k in fields), 0.95, policy, 0.7
                            )
                            module.route(
                                ids, expected["lookup"], reference, expected["counts"]
                            )
                            torch.testing.assert_close(
                                output, reference, rtol=0, atol=0
                            )
                            for field in fields:
                                torch.testing.assert_close(
                                    actual[field],
                                    expected[field],
                                    rtol=0,
                                    atol=0,
                                    msg=lambda message, field=field: (
                                        f"{field}: {message}"
                                    ),
                                )

    def test_decode_snapshots_hits_before_installing_misses(self):
        for native in (False, True):
            planner = (
                _module().decode_plan_parallel_fc1
                if native
                else _module().decode_plan_parallel_early
            )
            for policy in (0, 1):
                for row in ([0, 1], [2, 3], [0, 4], [0, 0, 4], [-1, -1]):
                    with self.subTest(policy=policy, row=row):
                        buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
                        state = buffers.reset(0, 4)
                        buffers.restore(0, state, [(0, 2.0, 1), (1, 1.0, 2)])
                        ids = torch.tensor([row], dtype=torch.int32, device="cuda")
                        output, early, late = (torch.empty_like(ids) for _ in range(3))
                        planner(
                            ids,
                            output,
                            *(
                                state[k]
                                for k in (
                                    "counts",
                                    "lookup",
                                    "keys",
                                    "frequency",
                                    "ages",
                                    "generation",
                                    "clock",
                                    "plan",
                                )
                            ),
                            0.95,
                            policy,
                            0.7,
                            0,
                            0,
                            2,
                            early,
                            late,
                        )
                        mixed = native or (
                            any(i in (0, 1) for i in row) and any(i >= 2 for i in row)
                        )
                        expected = [[i if mixed and i in (0, 1) else -1 for i in row]]
                        self.assertEqual(early.tolist(), expected)
                        late_ids = late.tolist()[0]
                        remapped = output.tolist()[0]
                        for i, expert in enumerate(row):
                            self.assertEqual(
                                late_ids[i],
                                -1 if mixed and expert in (0, 1) else remapped[i],
                            )
                        # Every valid route contributes once across the two branches.
                        actual = (early >= 0).int() + (late >= 0).int()
                        torch.testing.assert_close(actual, (ids >= 0).int())


if __name__ == "__main__":
    unittest.main()
