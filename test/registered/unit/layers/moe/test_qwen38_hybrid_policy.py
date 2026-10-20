# SPDX-License-Identifier: Apache-2.0
"""CPU-only global cache policy invariants."""

import unittest

from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu, choose_all_gpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestHybridPolicy(CustomTestCase):
    def test_eviction_precedes_invisible_fill(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(0, {0: 0})
        p.frequency[0][0] = 0.8
        epoch, snapshot = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.plan(0, epoch, [1])
        p.release(0, epoch)
        evicted, fills = p.plan(0, epoch, [1])
        self.assertEqual(evicted, [(0, 0)])
        self.assertEqual(fills, [(1, 0)])
        self.assertEqual(snapshot, {0: 0})
        self.assertEqual(p.layers[0].mapping, {})
        with self.assertRaises(RuntimeError):
            p.begin(0)
        p.commit(0, epoch, fills)
        self.assertEqual(p.begin(0)[1], {1: 0})

    def test_locked_layer_cannot_be_stolen(self):
        p = GlobalLfu(2, 8, [1, 1], admission=0)
        p.seed(1, {2: 1})
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [0, 1, 3])
        p.commit(0, epoch, fills)
        self.assertEqual(len(fills), 1)
        self.assertEqual(p.layers[1].mapping, {2: 1})
        self.assertEqual(p.owner[1], 1)

    def test_stale_epoch_and_noop_clean(self):
        p = GlobalLfu(1, 4, [0])
        epoch, _ = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.release(0, epoch - 1)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [])
        with self.assertRaises(RuntimeError):
            p.commit(0, epoch - 1, fills)
        p.commit(0, epoch, fills)
        self.assertEqual(p.begin(0)[0], epoch + 1)

    def test_cpu_misses_drive_lfu_and_admission(self):
        p = GlobalLfu(1, 4, [1])
        for iteration in range(2):
            epoch, _ = p.begin(0)
            p.release(0, epoch)
            _, fills = p.plan(0, epoch, [2, 2, 2])
            self.assertEqual(len(fills), iteration)
            p.commit(0, epoch, fills)
        self.assertAlmostEqual(p.frequency[0][2], 1.95)
        self.assertIn(2, p.layers[0].mapping)

    def test_only_released_empty_ownership_can_move(self):
        p = GlobalLfu(2, 8, [2, 0], admission=0)
        epoch, _ = p.begin(1)
        self.assertIsNone(p.immediate(1, epoch, {3}))
        p.release(1, epoch)
        _, fills = p.plan(1, epoch, [3])
        self.assertEqual(fills, [])
        p.commit(1, epoch, fills)
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        _, fills = p.plan(0, epoch, [])
        p.commit(0, epoch, fills)
        self.assertEqual(p.owner, [None, None])
        epoch, _ = p.begin(1)
        p.release(1, epoch)
        _, fills = p.plan(1, epoch, [3])
        p.commit(1, epoch, fills)
        self.assertEqual(p.owner[0], 1)

    def test_immediate_protects_resident_and_uses_own_slots(self):
        p = GlobalLfu(2, 8, [2, 2])
        p.seed(0, {0: 0, 1: 1})
        p.seed(1, {6: 2, 7: 3})
        epoch, _ = p.begin(0)
        self.assertIsNone(p.immediate(0, epoch, {0, 2, 3}))
        fills = p.immediate(0, epoch, {0, 2})
        self.assertEqual(fills, [(2, 1)])
        self.assertEqual(p.layers[0].mapping, {0: 0})
        self.assertEqual(p.layers[1].mapping, {6: 2, 7: 3})

    def test_zero_slot_reclaims_completed_lowest_value(self):
        p = GlobalLfu(3, 8, [2, 1, 0])
        p.seed(0, {0: 0, 1: 1})
        p.seed(1, {2: 2})
        p.frequency[0][0], p.frequency[0][1] = 8, 3
        p.frequency[1][2] = 1
        for layer in (0, 1):
            epoch, _ = p.begin(layer)
            p.release(layer, epoch)
            p.commit(layer, epoch, [])
        epoch, _ = p.begin(2)
        evicted = p.reclaim_zero_slot(2, epoch, {4, 5})
        self.assertEqual(evicted, [(1, 2, 2), (0, 1, 1)])
        self.assertEqual(p.layers[0].mapping, {0: 0})
        self.assertEqual(p.immediate(2, epoch, {4, 5}), [(4, 1), (5, 2)])

    def test_zero_slot_never_reclaims_active_or_never_released(self):
        p = GlobalLfu(3, 8, [1, 1, 0])
        p.seed(0, {0: 0})
        p.seed(1, {1: 1})
        p.begin(0)
        epoch, _ = p.begin(2)
        with self.assertRaises(RuntimeError):
            p.reclaim_zero_slot(2, epoch, {2})
        self.assertEqual(p.owner, [0, 1])
        self.assertEqual(p.layers[0].mapping, {0: 0})
        self.assertEqual(p.layers[1].mapping, {1: 1})

    def test_cold_eviction_releases_global_slot(self):
        p = GlobalLfu(1, 4, [1])
        p.seed(0, {0: 0})
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        evicted, fills = p.plan(0, epoch, [2])
        self.assertEqual(evicted, [(0, 0)])
        self.assertEqual(fills, [])
        self.assertEqual(p.owner, [None])
        p.commit(0, epoch, fills)

    def test_crossover_capacity_gate(self):
        self.assertFalse(choose_all_gpu(3, 3, 4, 3))
        self.assertFalse(choose_all_gpu(2, 4, 4, 3))
        self.assertTrue(choose_all_gpu(3, 4, 4, 3))

    def test_commit_rejects_wrong_reservation(self):
        p = GlobalLfu(2, 4, [1, 1])
        epoch, _ = p.begin(0)
        p.release(0, epoch)
        with self.assertRaises(RuntimeError):
            p.commit(0, epoch, [(1, 1)])

    def test_resize_selects_global_heat_across_layer_boundaries(self):
        p = GlobalLfu(3, 8, [2, 2, 2])
        for layer in range(3):
            p.seed(layer, {0: 0, 1: 1})
            p.frequency[layer][:2] = [layer * 10 + 1, layer * 10 + 2]
        plan = p.plan_resize(3)
        self.assertEqual(set(plan.keys), {(1, 1), (2, 0), (2, 1)})
        self.assertEqual(len(plan.moves), 3)
        # No policy change before copies finish.
        self.assertEqual(len(p.keys), 6)
        p.commit_resize(plan)
        self.assertEqual(p.layers[0].mapping, {})
        self.assertEqual(len(p.layers[2].mapping), 2)
        self.assertEqual(len(p.keys), 3)

    def test_growth_preserves_payload_addresses_and_creates_unowned_slots(self):
        p = GlobalLfu(2, 8, [1, 1])
        p.seed(0, {2: 0})
        p.seed(1, {3: 1})
        before = [dict(state.mapping) for state in p.layers]
        plan = p.plan_resize(7)
        self.assertEqual(plan.moves, ())
        self.assertEqual(plan.evicted, ())
        p.commit_resize(plan)
        self.assertEqual([state.mapping for state in p.layers], before)
        self.assertEqual(p.owner[2:], [None] * 5)

    def test_resize_preserves_surviving_prefix_without_moves(self):
        p = GlobalLfu(2, 8, [2, 2])
        p.seed(0, {0: 0, 1: 1})
        p.seed(1, {0: 2, 1: 3})
        p.frequency[0][:2] = [9, 8]
        p.frequency[1][:2] = [7, 6]
        plan = p.plan_resize(2)
        self.assertEqual(plan.moves, ())
        p.commit_resize(plan)
        self.assertEqual(p.layers[0].mapping, {0: 0, 1: 1})
        self.assertEqual(p.layers[1].mapping, {})

    def test_resize_rejects_computing_managing_and_stale_plan(self):
        p = GlobalLfu(1, 4, [2])
        epoch, _ = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.plan_resize(1)
        p.release(0, epoch)
        with self.assertRaises(RuntimeError):
            p.plan_resize(1)
        p.commit(0, epoch, [])
        plan = p.plan_resize(1)
        p.commit_resize(p.plan_resize(3))
        with self.assertRaises(RuntimeError):
            p.commit_resize(plan)

    def test_resize_moves_keep_all_weight_bytes_through_stage_cycles(self):
        p = GlobalLfu(4, 8, [4] * 4)
        for layer in range(4):
            p.seed(layer, {e: e for e in range(4)})
            p.frequency[layer][:4] = [((layer + 1) * 13 + e * 7) % 19 for e in range(4)]
        payload = [bytes([key[0], key[1]]) * 32 for key in p.keys]
        expected = {key: value for key, value in zip(p.keys, payload)}
        for capacity in (11, 16, 6, 14, 4):
            old = list(payload)
            plan = p.plan_resize(capacity)
            payload += [None] * max(0, capacity - len(payload))
            for source, destination in plan.moves:
                self.assertGreaterEqual(source, capacity)
                self.assertLess(destination, capacity)
                payload[destination] = payload[source]
            for slot, key in enumerate(plan.keys):
                if key is not None:
                    self.assertEqual(payload[slot], expected[key])
            p.commit_resize(plan)
            payload = payload[:capacity]
            for state in p.layers:
                for expert, slot in state.mapping.items():
                    self.assertLess(slot, capacity)
            self.assertEqual(
                len(old) < capacity, not plan.evicted and capacity > len(old)
            )

    def test_b12x_zero_slot_is_not_admitted_reclaimed_or_resize_destination(self):
        p = GlobalLfu(2, 8, [2, 2], admission=0)
        p.reserved.add(0)
        p.owner[0] = -1
        p.seed(1, {1: 2, 2: 3})
        p.frequency[1][1:3] = [8, 9]
        plan = p.plan_resize(2)
        self.assertIsNone(plan.keys[0])
        self.assertEqual(plan.keys[1], (1, 2))
        self.assertEqual(plan.moves, ((3, 1),))
        p.commit_resize(plan)
        self.assertEqual(p.owner[0], -1)
        epoch, _ = p.begin(0)
        with self.assertRaises(RuntimeError):
            p.reclaim_zero_slot(0, epoch, {3, 4})
        self.assertEqual(p.owner[0], -1)

    def test_deterministic_pressure_preserves_ownership(self):
        p = GlobalLfu(4, 16, [4] * 4, admission=0)
        for step in range(120):
            layer = step % 4
            epoch, _ = p.begin(layer)
            before = [dict(s.mapping) for s in p.layers]
            p.release(layer, epoch)
            _, fills = p.plan(layer, epoch, [(step * 7) % 16, (step * 11 + 3) % 16])
            p.commit(layer, epoch, fills)
            slots = []
            for idx, state in enumerate(p.layers):
                if idx != layer:
                    self.assertEqual(state.mapping, before[idx])
                for expert, slot in state.mapping.items():
                    self.assertEqual(p.owner[slot], idx)
                    self.assertEqual(p.keys[slot], (idx, expert))
                    slots.append(slot)
            self.assertEqual(len(slots), len(set(slots)))


if __name__ == "__main__":
    unittest.main()
