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

    def test_decode_snapshots_hits_before_installing_misses(self):
        for policy in (0, 1):
            for row in ([0, 1], [2, 3], [0, 4], [0, 0, 4], [-1, -1]):
                with self.subTest(policy=policy, row=row):
                    buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
                    state = buffers.reset(0, 4)
                    buffers.restore(0, state, [(0, 2.0, 1), (1, 1.0, 2)])
                    ids = torch.tensor([row], dtype=torch.int32, device="cuda")
                    output, early, late = (torch.empty_like(ids) for _ in range(3))
                    _module().decode_plan_parallel_early(
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
                    mixed = any(i in (0, 1) for i in row) and any(i >= 2 for i in row)
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
