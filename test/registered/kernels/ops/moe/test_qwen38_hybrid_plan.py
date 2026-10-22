# SPDX-License-Identifier: Apache-2.0
"""Compare device admission and CPU subsets to the independent host policy."""

import mmap
import random
import unittest

import torch
from sglang.kernels.ops.moe.qwen38_hybrid_plan import (
    COST_COLUMNS,
    hybrid_merge,
    hybrid_plan,
    hybrid_plan_indexed,
    hybrid_plan_stage_indexed,
    hybrid_result,
    hybrid_stage,
    hybrid_wait,
)
from sglang.srt.layers.moe.qwen38_heap_index import build_heap_index
from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def materialize_gpu(policy, layer, requested, counts):
    """Reference forced placement after one LFU observation and admission."""
    fills, evictions = [], 0
    for expert in sorted(
        requested - policy.layers[layer].mapping.keys(),
        key=lambda e: (counts[e], e),
    ):
        free = [
            s
            for s, key in enumerate(policy.keys)
            if key is None and s not in policy.reserved
        ]
        if free:
            slot = min(free)
        else:
            victims = [
                (policy.frequency[l][e], l, e, s)
                for l, state in enumerate(policy.layers)
                for e, s in state.mapping.items()
                if l != layer or e not in requested
            ]
            if not victims:
                raise ValueError("insufficient GPU pool capacity")
            _, old_layer, old, slot = min(victims)
            del policy.layers[old_layer].mapping[old]
            evictions += 1
        policy.set_owner(slot, layer)
        policy.keys[slot] = (layer, expert)
        policy.layers[layer].mapping[expert] = slot
        fills.append((expert, slot))
    return fills, evictions


class TestHybridPlan(CustomTestCase):
    def test_prefill_histogram_preserves_ids_and_counts_invalid_routes(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_route_counts

        ids = torch.randint(-1, 512, (8192, 10), dtype=torch.int32, device="cuda")
        ids[0, 0], ids[1, 1] = -2, 512
        original = ids.clone()
        counts = torch.empty(513, dtype=torch.int32, device="cuda")
        cache_route_counts(ids, counts)
        host = original.cpu().flatten()
        expected = torch.zeros(513, dtype=torch.int32)
        expected[:512] = torch.bincount(
            host[(host >= 0) & (host < 512)].long(), minlength=512
        ).to(torch.int32)
        expected[-1] = ((host < -1) | (host >= 512)).sum()
        torch.testing.assert_close(counts.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ids, original, rtol=0, atol=0)
        cache_route_counts(ids[:0], counts)
        torch.testing.assert_close(
            counts.cpu(), torch.zeros_like(expected), rtol=0, atol=0
        )

    def test_plan_stage_publishes_only_selected_cpu_work(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module as mailbox_module

        mapping = mmap.mmap(-1, 4096, flags=mmap.MAP_SHARED)
        host = torch.frombuffer(mapping, dtype=torch.uint8)
        head = torch.frombuffer(mapping, dtype=torch.int32, count=3)
        host_x = torch.frombuffer(mapping, dtype=torch.bfloat16, count=24, offset=12)
        host_ids = torch.frombuffer(mapping, dtype=torch.int32, count=6, offset=60)
        host_weights = torch.frombuffer(
            mapping, dtype=torch.float32, count=6, offset=84
        )
        module = mailbox_module()
        address = module.mailbox_register(host)
        try:
            x = torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(3, 8)
            ids = torch.tensor(
                [[0, 1], [2, 3], [4, -1]], device="cuda", dtype=torch.int32
            )
            weights = torch.ones((3, 2), device="cuda", dtype=torch.float32)
            cost_row = [3, 4, 1, 1, 1, 1] + [1] + [0] * 31 + [1] + [0] * 31
            costs = torch.tensor([cost_row], device="cuda", dtype=torch.float64)

            def run(with_costs, sequence):
                policy = GlobalLfu(2, 16, [4, 1])
                policy.seed(0, {e: e for e in range(4)})
                policy.seed(1, {10: 4})
                policy.frequency[1][10] = 10
                state = self.tensors(policy)
                cache_ids, cpu_ids = torch.empty_like(ids), torch.empty_like(ids)
                plan = torch.empty(
                    4 + 2 * ids.numel(), device="cuda", dtype=torch.int32
                )
                stats = torch.zeros(7, device="cuda", dtype=torch.int64)
                owners = [-1 if o is None else o for o in policy.owner]
                keys = [-1 if k is None else k[1] for k in policy.keys]
                index = build_heap_index(
                    policy.frequency, owners, keys, policy.experts, len(keys) + 4
                ).to("cuda")
                hybrid_plan_stage_indexed(
                    ids,
                    *state,
                    with_costs,
                    cache_ids,
                    cpu_ids,
                    plan,
                    stats,
                    0,
                    0.95,
                    1.9,
                    x,
                    weights,
                    address + 12,
                    address + 60,
                    address + 84,
                    address,
                    sequence,
                    index,
                )
                torch.cuda.synchronize()
                return cpu_ids

            head.zero_()
            selected = run(costs, 1)
            self.assertEqual(selected[selected >= 0].tolist(), [4])
            self.assertEqual(head.tolist(), [1, 0, 3])
            torch.testing.assert_close(host_x.reshape(3, 8), x.cpu())
            torch.testing.assert_close(host_ids.reshape(3, 2), selected.cpu())
            torch.testing.assert_close(host_weights.reshape(3, 2), weights.cpu())
            head.fill_(77)
            host_x.fill_(91)
            host_ids.fill_(92)
            host_weights.fill_(93)
            gpu_only = run(costs[:0], 2)
            self.assertTrue((gpu_only == -1).all())
            self.assertEqual(head.tolist(), [77, 77, 77])
            self.assertTrue((host_x == 91).all())
            self.assertTrue((host_ids == 92).all())
            self.assertTrue((host_weights == 93).all())
        finally:
            torch.cuda.synchronize()
            module.mailbox_unregister(host)
            del head, host_x, host_ids, host_weights, host
            mapping.close()

    def test_mapped_three_token_round_trip_and_gpu_only_zero(self):
        """Only selected routes publish a request; results round-trip."""
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module as mailbox_module

        x = torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(3, 8)
        ids = torch.tensor(
            [[2, -1], [3, 4], [-1, -1]], device="cuda", dtype=torch.int32
        )
        weights = torch.arange(6, device="cuda", dtype=torch.float32).reshape(3, 2)
        mapping = mmap.mmap(-1, 4096, flags=mmap.MAP_SHARED)
        try:
            host = torch.frombuffer(mapping, dtype=torch.uint8)
            head = torch.frombuffer(mapping, dtype=torch.int32, count=1)
            host_layer = torch.frombuffer(mapping, dtype=torch.int32, count=1, offset=4)
            host_tokens = torch.frombuffer(
                mapping, dtype=torch.int32, count=1, offset=8
            )
            host_x = torch.frombuffer(
                mapping, dtype=torch.bfloat16, count=24, offset=12
            )
            host_ids = torch.frombuffer(mapping, dtype=torch.int32, count=6, offset=60)
            host_weights = torch.frombuffer(
                mapping, dtype=torch.float32, count=6, offset=84
            )
            host_result = torch.frombuffer(
                mapping, dtype=torch.float32, count=24, offset=108
            )
            module = mailbox_module()
            address = module.mailbox_register(host)
            try:
                head.zero_()
                hybrid_stage(
                    x,
                    ids,
                    weights,
                    address + 12,
                    address + 60,
                    address + 84,
                    address,
                    7,
                    1,
                )
                torch.cuda.synchronize()
                self.assertEqual(head.item(), 1)
                self.assertEqual(host_layer.item(), 7)
                self.assertEqual(host_tokens.item(), 3)
                torch.testing.assert_close(host_x.reshape(3, 8), x.cpu())
                torch.testing.assert_close(host_ids.reshape(3, 2), ids.cpu())
                torch.testing.assert_close(host_weights.reshape(3, 2), weights.cpu())
                host_result.copy_(torch.arange(24, dtype=torch.float32) + 7)
                output = torch.empty((3, 8), device="cuda", dtype=torch.float32)
                hybrid_result(ids, output, address + 108)
                torch.testing.assert_close(output.cpu(), host_result.reshape(3, 8))
                ids.fill_(-1)
                host_x.fill_(91)
                host_ids.fill_(92)
                host_weights.fill_(93)
                hybrid_stage(
                    x,
                    ids,
                    weights,
                    address + 12,
                    address + 60,
                    address + 84,
                    address,
                    8,
                    2,
                )
                torch.cuda.synchronize()
                self.assertEqual(head.item(), 1)
                self.assertEqual(host_layer.item(), 7)
                self.assertEqual(host_tokens.item(), 3)
                self.assertTrue((host_x == 91).all())
                self.assertTrue((host_ids == 92).all())
                self.assertTrue((host_weights == 93).all())
                hybrid_result(ids, output, address + 108)
                self.assertTrue((output == 0).all())
                torch.cuda.synchronize()
            finally:
                module.mailbox_unregister(host)
            del (
                head,
                host_layer,
                host_tokens,
                host_x,
                host_ids,
                host_weights,
                host_result,
                host,
            )
        finally:
            mapping.close()

    def five_rejected_policy(self):
        p = GlobalLfu(2, 16, [5, 5])
        p.seed(1, {e: e - 5 for e in range(10, 15)})
        p.frequency[1] = [10.0] * 16
        return p

    def tensors(self, policy):
        lookup = [[-1] * policy.experts for _ in policy.layers]
        for layer, state in enumerate(policy.layers):
            for e, s in state.mapping.items():
                lookup[layer][e] = s
        return (
            torch.tensor(lookup, dtype=torch.int32, device="cuda"),
            torch.tensor(policy.frequency, dtype=torch.float64, device="cuda"),
            torch.tensor(
                [-1 if o is None else o for o in policy.owner],
                dtype=torch.int32,
                device="cuda",
            ),
            torch.tensor(
                [-1 if k is None else k[1] for k in policy.keys],
                dtype=torch.int32,
                device="cuda",
            ),
        )

    def invoke(self, ids, state, costs=None, layer=0, index=None):
        ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
        routes = [torch.empty_like(ids) for _ in range(2)]
        plans = [
            torch.empty(4 + 2 * ids.numel(), dtype=torch.int32, device="cuda")
            for _ in range(1)
        ]
        stats = torch.zeros(7, dtype=torch.int64, device="cuda")
        if costs is None:
            costs = torch.empty((0, COST_COLUMNS), dtype=torch.float64, device="cuda")
        arguments = (ids, *state, costs, *routes, *plans, stats, layer, 0.95, 1.9)
        if index is None:
            hybrid_plan(*arguments)
        else:
            hybrid_plan_indexed(*arguments, index)
        torch.cuda.synchronize()
        return routes, plans, stats

    def test_all_admitted_and_select_one_of_five_rejected(self):
        p = GlobalLfu(1, 16, [10])
        routes, plans, stats = self.invoke([list(range(10))], self.tensors(p))
        self.assertEqual(plans[0][1].item(), 10)
        self.assertTrue((routes[1] == -1).all())
        p = self.five_rejected_policy()
        # Five free slots; five hot global residents prevent LFU replacement.
        row = [1, 9, 1, 1, 5, 10] + [1] + [0] * 31 + [1] * 5 + [0] * 27
        # Exercise both shared-table capacity and the global-memory fallback.
        # Only the final row matches, so truncating a large table loses CPU work.
        for count in (1, 32, 33, 64, 65):
            with self.subTest(cost_rows=count):
                costs = torch.zeros(
                    (count, COST_COLUMNS), dtype=torch.float64, device="cuda"
                )
                costs[-1] = torch.tensor(row, dtype=torch.float64, device="cuda")
                routes, plans, stats = self.invoke(
                    [list(range(10))], self.tensors(p), costs
                )
                self.assertEqual(plans[0][1].item(), 9)
                self.assertEqual(plans[0][2].item(), 1)
                self.assertEqual(stats[3].item(), 5)
                self.assertEqual(stats[6].item(), 4)
                self.assertEqual(stats[0].item(), 4)
                self.assertEqual(routes[1][routes[1] >= 0].tolist(), [5])
                self.assertEqual(int((routes[0] >= 0).sum()), 9)
                self.assertEqual(stats[5].item(), 1)

    def test_random_windows_match_global_lfu(self):
        p = GlobalLfu(2, 16, [6, 6])
        p.seed(0, {e: e for e in range(5)})
        p.seed(1, {e: e for e in range(5, 10)})
        state = self.tensors(p)
        rng = random.Random(42)
        for step in range(80):
            layer = step % 2
            rows = [
                [rng.randrange(-1, 16) for _ in range(4)] for _ in range(1 + step % 3)
            ]
            requested = {e for row in rows for e in row if e >= 0}
            epoch, _ = p.begin(layer)
            p.release(layer, epoch)
            evicted, fills = p.plan(layer, epoch, requested, safe_layers=range(2))
            p.commit(layer, epoch, fills)
            forced, forced_evictions = materialize_gpu(
                p,
                layer,
                requested,
                {e: sum(row.count(e) for row in rows) for e in requested},
            )
            routes, plans, stats = self.invoke(rows, state, layer=layer)
            mapping = {
                e: s for e, s in enumerate(state[0][layer].cpu().tolist()) if s >= 0
            }
            self.assertEqual(mapping, p.layers[layer].mapping)
            self.assertEqual(plans[0][1].item(), len(fills) + len(forced))
            self.assertEqual(stats[0].item(), len(evicted) + forced_evictions)
            # Match host FP64 arithmetic and global ownership exactly.
            torch.testing.assert_close(
                state[1].cpu(),
                torch.tensor(p.frequency, dtype=torch.float64),
                rtol=0,
                atol=0,
            )
            expected_owners = [-1 if o is None else o for o in p.owner]
            self.assertEqual(state[2].cpu().tolist(), expected_owners)
            self.assertTrue((routes[1] == -1).all())

    def test_colder_expert_in_other_layer_is_replaced(self):
        p = GlobalLfu(2, 16, [1, 1], admission=0)
        p.seed(0, {0: 0})
        p.seed(1, {2: 1})
        p.frequency[0][0] = 10
        p.frequency[1][2] = 0.8
        state = self.tensors(p)
        ids = torch.tensor([[3]], dtype=torch.int32, device="cuda")
        routes = [torch.empty_like(ids) for _ in range(2)]
        plans = [torch.empty(6, dtype=torch.int32, device="cuda") for _ in range(1)]
        costs = torch.empty((0, COST_COLUMNS), dtype=torch.float64, device="cuda")
        stats = torch.zeros(7, dtype=torch.int64, device="cuda")
        hybrid_plan(ids, *state, costs, *routes, *plans, stats, 0, 0.95, 0)
        torch.cuda.synchronize()
        self.assertEqual(state[0][0, 0].item(), 0)
        self.assertEqual(state[0][0, 3].item(), 1)
        self.assertEqual(state[0][1, 2].item(), -1)
        self.assertEqual(state[2].tolist(), [0, 0])
        self.assertEqual(plans[0][1].item(), 1)

    def test_indexed_global_minima_follow_repeated_replacements(self):
        """Shared layer minima must follow empty/refilled layers and heat ties."""
        for layers, experts in ((2, 16), (2, 512), (48, 512), (64, 16), (65, 16)):
            with self.subTest(layers=layers, experts=experts):
                policy = GlobalLfu(layers, experts, [8] * layers)
                for layer in range(layers):
                    policy.seed(
                        layer,
                        {
                            e: 3 * layer + i
                            for i, e in enumerate((0, experts // 2, experts - 1))
                        },
                    )
                    policy.frequency[layer] = [2.0] * experts
                state = self.tensors(policy)
                reference = tuple(t.clone() for t in state)
                index = build_heap_index(
                    policy.frequency,
                    state[2].tolist(),
                    state[3].tolist(),
                    experts,
                    len(policy.keys) + 7,
                ).to("cuda")
                rng = random.Random(51)
                for step in range(80):
                    layer = step % layers
                    rows = [
                        [rng.randrange(-1, experts) for _ in range(4)]
                        for _ in range(1 + step % 3)
                    ]
                    if step % 17 == 0:
                        rows = [[-1] * 4]
                    actual = self.invoke(rows, state, layer=layer, index=index)
                    expected = self.invoke(rows, reference, layer=layer)
                    # Free-slot order may differ; resident experts, heat and
                    # eviction counts must match the independent scan policy.
                    torch.testing.assert_close(state[1], reference[1], rtol=0, atol=0)
                    self.assertEqual(
                        (state[0] >= 0).tolist(), (reference[0] >= 0).tolist()
                    )
                    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
                    self.assertEqual(actual[1][0][1].item(), expected[1][0][1].item())
                    torch.testing.assert_close(
                        actual[0][1], expected[0][1], rtol=0, atol=0
                    )

    def test_forced_global_replacement_protects_active_hits(self):
        p = GlobalLfu(2, 16, [1, 1])
        p.seed(0, {0: 0})
        p.seed(1, {2: 1})
        p.frequency[1][2] = 10
        state = self.tensors(p)
        routes, plans, stats = self.invoke([[0, 3]], state)
        self.assertEqual(state[0][0, 0].item(), 0)
        self.assertEqual(state[0][0, 3].item(), 1)
        self.assertEqual(state[0][1, 2].item(), -1)
        self.assertEqual(routes[0].tolist(), [[0, 1]])
        self.assertEqual(plans[0][4:6].tolist(), [3, 1])
        self.assertEqual(stats[3].item(), 0)
        self.assertEqual(stats[6].item(), 1)
        self.assertEqual(stats[0].item(), 1)

    def test_cost_profile_mismatch_keeps_all_gpu(self):
        p = self.five_rejected_policy()
        row = [1, 9, 1, 1, 1, 1] + [1] + [0] * 31 + [1] + [0] * 31
        costs = torch.tensor([row], dtype=torch.float64, device="cuda")
        routes, plans, _ = self.invoke([list(range(10))], self.tensors(p), costs)
        self.assertTrue((routes[1] == -1).all())
        self.assertEqual(plans[0][1].item(), 10)
        self.assertEqual(plans[0][2].item(), 0)
        self.assertEqual(int((routes[0] >= 0).sum()), 10)

    def test_gpu_only_wait_bypasses_unpublished_cpu_completion(self):
        ids = torch.full((3, 10), -1, dtype=torch.int32, device="cuda")
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        hybrid_wait(ids, flag, 1)
        torch.cuda.synchronize()

    def test_compact_prefill_reuses_noncontiguous_global_slots(self):
        from collections import Counter
        from types import SimpleNamespace

        from sglang.srt.layers.moe.qwen38_hybrid import HybridCache

        subject = HybridCache.__new__(HybridCache)
        subject.pool = {
            "weights": torch.arange(64 * 32, device="cuda", dtype=torch.int32).reshape(
                64, 32
            ),
            "scales": torch.arange(64 * 8, device="cuda", dtype=torch.float32).reshape(
                64, 8
            ),
        }
        source = {name: tensor.clone() for name, tensor in subject.pool.items()}
        arena = {
            name: torch.full_like(tensor[:8], -1) for name, tensor in source.items()
        }
        subject.runtime = SimpleNamespace(
            device=torch.device("cuda"),
            transfer=torch.cuda.Stream(),
            manifest=SimpleNamespace(num_experts=8),
            arena=arena,
            prefill_reused_experts=0,
            prefill_reuse_d2d_bytes=0,
        )
        subject.stats = Counter()
        subject.runtime.transfer.wait_stream(torch.cuda.current_stream())
        subject._copy_prefill_hits({2, 5}, {2: 4, 5: 51})
        torch.cuda.current_stream().wait_stream(subject.runtime.transfer)
        for name, tensor in arena.items():
            expected = torch.full_like(tensor, -1)
            expected[2], expected[5] = source[name][4], source[name][51]
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
            torch.testing.assert_close(subject.pool[name], source[name], rtol=0, atol=0)
        self.assertEqual(subject.runtime.prefill_reused_experts, 2)
        self.assertEqual(subject.runtime.prefill_reuse_d2d_bytes, 2 * (32 * 4 + 8 * 4))
        # Admission reads the computed arena, with destinations scattered into
        # the global cache; it must preserve all unselected cache rows.
        subject.fill_stream = torch.cuda.Stream()
        done = torch.cuda.Event()
        done.record()
        copied = subject._copy_prefill_admissions([(5, 7), (2, 59)], done)
        copied.synchronize()
        for name, tensor in subject.pool.items():
            expected = source[name].clone()
            expected[7], expected[59] = arena[name][5], arena[name][2]
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        self.assertEqual(
            subject.stats["prefill_admission_d2d_bytes"], 2 * (32 * 4 + 8 * 4)
        )

    def test_merge_reads_selected_result_and_skips_gpu_only_mmap(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module as mailbox_module

        # Cross CTA boundaries and leave a partial final tile.
        width = 2051
        mapping = mmap.mmap(-1, 32768, flags=mmap.MAP_SHARED)
        host = torch.frombuffer(mapping, dtype=torch.uint8)
        result = torch.frombuffer(mapping, dtype=torch.float32, count=3 * width)
        host_completed = torch.frombuffer(
            mapping, dtype=torch.int32, count=1, offset=32764
        )
        result.copy_(torch.arange(3 * width, dtype=torch.float32))
        module = mailbox_module()
        address = module.mailbox_register(host)
        try:
            gpu = torch.full((3, width), 0.5, device="cuda", dtype=torch.float32)
            ids = torch.full((3, 2), -1, device="cuda", dtype=torch.int32)
            completed = torch.zeros(1, device="cuda", dtype=torch.int32)
            output = torch.empty((3, width), device="cuda", dtype=torch.bfloat16)
            hybrid_merge(ids, gpu, completed, output, address, 1)
            torch.testing.assert_close(output, gpu.to(torch.bfloat16), rtol=0, atol=0)
            ids[-1, -1] = 2
            completed.fill_(1)
            hybrid_merge(ids, gpu, completed, output, address, 1)
            expected = (gpu.cpu() + result.reshape(3, width)).to(torch.bfloat16)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            # B12X publishes directly into registered host memory. The CUDA
            # flag deliberately stays stale: waiting on it would hang forever.
            completed.zero_()
            host_completed.fill_(2)
            result.add_(1)
            hybrid_merge(
                ids,
                gpu,
                completed,
                output,
                address,
                2,
                completed_address=address + 32764,
            )
            expected = (gpu.cpu() + result.reshape(3, width)).to(torch.bfloat16)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        finally:
            torch.cuda.synchronize()
            module.mailbox_unregister(host)
            del host_completed, result, host
            mapping.close()


if __name__ == "__main__":
    unittest.main()
