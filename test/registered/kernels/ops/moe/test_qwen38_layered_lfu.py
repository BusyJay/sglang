# SPDX-License-Identifier: Apache-2.0
"""CPU misses accumulate heat and can migrate back into their layer's GPU cache."""

import mmap
import unittest

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestLayeredLfuAdmission(CustomTestCase):
    def setUp(self):
        self.module = _module()
        self.buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
        self.state = self.buffers.reset(0, 4)
        self.heat = torch.zeros(8, dtype=torch.float32, device="cuda")
        self.mapping = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
        self.host = torch.frombuffer(self.mapping, dtype=torch.uint8)
        self.address = self.module.mailbox_register(self.host)
        self.sequence = 0

    def tearDown(self):
        torch.cuda.synchronize()
        self.module.mailbox_unregister(self.host)
        del self.host
        self.mapping.close()

    def restore(self, records):
        self.buffers.restore(0, self.state, records)

    def run_plan(self, routes, limit, decay=0.95):
        ids = torch.tensor(routes, dtype=torch.int32, device="cuda")
        tokens, topk = ids.shape
        weights = torch.ones_like(ids, dtype=torch.float32)
        remapped = torch.empty_like(ids)
        cpu_ids = torch.empty_like(ids)
        gpu_weights = torch.empty_like(weights)
        stats = torch.zeros(5, dtype=torch.int64, device="cuda")
        desc = torch.empty(2, dtype=torch.int32, device="cuda")
        self.sequence += 1
        self.module.decode_plan_layered_lfu(
            ids.view(1, -1),
            remapped.view(1, -1),
            *(
                self.state[k]
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
            decay,
            0,
            0.7,
            0,
            0,
            2,
            torch.ones((tokens, 32), dtype=torch.bfloat16, device="cuda"),
            weights,
            torch.empty((0, 47), dtype=torch.int32, device="cuda"),
            gpu_weights,
            cpu_ids,
            desc,
            stats,
            self.address,
            0,
            self.sequence,
            -1,
            1,
            0,
            self.heat,
            limit,
        )
        cpu = set(cpu_ids[cpu_ids >= 0].tolist())
        plan = self.state["plan"].tolist()
        fills = set(plan[4 : 4 + 2 * plan[1] : 2])
        self.assertFalse(cpu & fills)
        self.assertEqual(stats[0].item(), len(cpu) + len(fills))
        self.assertEqual(desc[0].item(), len(cpu))
        self.assertTrue(torch.all(remapped >= 0).item())
        torch.testing.assert_close(
            gpu_weights, (cpu_ids < 0).float() * (ids >= 0).float()
        )
        return cpu, fills

    def test_admission_metadata_spans_warps_and_partial_capacity(self):
        """Resident heat and empty slots must include the entire expert domain."""
        self.buffers = AdaptiveMetadataBuffers(1, 512, 129, torch.device("cuda"))
        for free_slot in (False, True):
            with self.subTest(free_slot=free_slot):
                self.state = self.buffers.reset(0, 129)
                self.heat = torch.zeros(512, dtype=torch.float32, device="cuda")
                records = [(e, 10, e + 1) for e in range(128)]
                if not free_slot:
                    # The eligible minimum lies outside the first 128 experts.
                    records.append((511, 0.5, 129))
                self.restore(records)
                self.heat[401] = 2
                cpu, fills = self.run_plan([[0, 400, 401]], 1)
                self.assertEqual(cpu, {400} if free_slot else set())
                self.assertEqual(fills, {401} if free_slot else {400, 401})
                torch.testing.assert_close(
                    self.heat[[0, 127, 400, 401]],
                    torch.tensor([10.5, 9.5, 1, 2.9], device="cuda"),
                )

    def test_count_gate_is_all_or_nothing_and_hits_are_protected(self):
        for limit, routes, expected in (
            (1, [[0, 4]], {4}),
            (1, [[0, 4, 5]], set()),
            (2, [[0, 4, 5]], {4, 5}),
            (0, [[0, 4]], set()),
        ):
            with self.subTest(limit=limit, routes=routes):
                self.state = self.buffers.reset(0, 4)
                self.heat.zero_()
                self.restore([(0, 0.1, 1), (1, 10, 2), (2, 10, 3), (3, 10, 4)])
                cpu, _ = self.run_plan(routes, limit)
                self.assertEqual(cpu, expected)
                self.assertEqual(self.state["lookup"][0].item(), 0)

    def test_cpu_heat_promotes_back_to_gpu_and_preserves_frequency(self):
        self.restore([(0, 100, 1), (1, 3, 2), (2, 4, 3), (3, 5, 4)])
        for expected in ({4}, {4}, set(), set()):
            cpu, fills = self.run_plan([[0, 4]], 1)
            self.assertEqual(cpu, expected)
            if cpu:
                self.assertEqual(self.state["lookup"][4].item(), -1)
            else:
                slot = self.state["lookup"][4].item()
                self.assertGreaterEqual(slot, 0)
                self.assertAlmostEqual(
                    self.state["frequency"][slot].item(), self.heat[4].item()
                )
        self.assertAlmostEqual(
            self.heat[4].item(), 1 + 0.95 + 0.95**2 + 0.95**3, places=5
        )
        self.assertEqual(fills, set())

    def test_partial_free_capacity_ties_and_b12x_all_cpu_guard(self):
        self.restore([(0, 10, 1), (1, 10, 2), (2, 10, 3)])
        self.heat[5] = 2
        self.assertEqual(self.run_plan([[0, 4, 5]], 1), ({4}, {5}))
        # Equal-heat misses reserve the larger expert ID for the free slot,
        # independently of their order in the router output.
        self.state = self.buffers.reset(0, 4)
        self.heat.zero_()
        self.restore([(0, 10, 1), (1, 10, 2), (2, 10, 3)])
        self.assertEqual(self.run_plan([[0, 5, 4]], 1), ({4}, {5}))
        self.state = self.buffers.reset(0, 4)
        self.heat.zero_()
        self.restore([(0, 2, 1), (1, 2, 2), (2, 2, 3), (3, 2, 4)])
        self.heat[4] = 1
        self.assertEqual(self.run_plan([[0, 4]], 1, decay=1), (set(), {4}))
        self.state = self.buffers.reset(0, 4)
        self.heat.zero_()
        self.restore([(0, 10, 1), (1, 10, 2), (2, 10, 3), (3, 10, 4)])
        self.assertEqual(self.run_plan([[4]], 1), (set(), {4}))

    def test_free_slots_admit_misses_and_duplicate_routes_count_once(self):
        self.restore([(0, 10, 1)])
        self.assertEqual(self.run_plan([[0, 4, 5]], 2), (set(), {4, 5}))
        self.state = self.buffers.reset(0, 4)
        self.heat.zero_()
        self.restore([(0, 10, 1), (1, 10, 2), (2, 10, 3), (3, 10, 4)])
        self.assertEqual(self.run_plan([[0, 4, 4], [0, 4, -1]], 1), ({4}, set()))
        self.assertEqual(self.heat[4].item(), 1)
        # Repacking slots does not reset a nonresident CPU expert's heat.
        self.restore([(0, 10, 1), (1, 10, 2), (2, 10, 3), (3, 10, 4)])
        self.run_plan([[0, 4]], 1)
        self.assertAlmostEqual(self.heat[4].item(), 1.95, places=5)
        self.state = self.buffers.reset(0, 2)
        self.restore([(0, 10, 1), (1, 10, 2)])
        self.assertEqual(self.run_plan([[0, 4]], 1), ({4}, set()))
        self.assertAlmostEqual(self.heat[4].item(), 2.8525, places=5)
        self.restore([(0, 10, 1), (1, 1, 2)])
        self.assertEqual(self.run_plan([[0, 4]], 1), (set(), {4}))
        slot = self.state["lookup"][4].item()
        self.assertAlmostEqual(
            self.state["frequency"][slot].item(), self.heat[4].item()
        )


if __name__ == "__main__":
    unittest.main()
