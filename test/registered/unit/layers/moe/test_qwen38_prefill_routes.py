"""Cache/arena routing preserves the original weighted expert sum."""

import threading
import unittest
from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.qwen38_prefill_routes import partition_prefill_routes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestPrefillRoutes(CustomTestCase):
    def test_kv_tail_release_waits_for_hot_expert_migration(self):
        """KV cannot reuse expert backing until payload migration completes."""
        from sglang.srt.mem_cache.qwen38_kv_stages import StagedKVLoan

        order = []
        loan = StagedKVLoan.__new__(StagedKVLoan)
        loan.policy = SimpleNamespace(index=0, capacities=[8, 16, 24])
        loan.allocator = SimpleNamespace(
            num_pages=12,
            page_size=2,
            backed_num_pages=4,
            free_pages=torch.tensor([1, 2], dtype=torch.int64),
            device="cpu",
        )
        loan.runtime = SimpleNamespace(
            staged_trim_enqueued=True,
            preserve_cache_on_resize=True,
            staged_trim_done=SimpleNamespace(query=lambda: True),
            complete_staged_trim=lambda: order.append("migrate completed"),
            staged_pool=SimpleNamespace(
                release_stage=lambda _: order.append("release backing")
            ),
            expert_loan_budgets=[6, 4, 2],
        )
        loan.pool = SimpleNamespace(
            _post_capture_owner=SimpleNamespace(
                finalize=lambda _: order.append("grow KV"),
            )
        )
        loan._return(1)
        self.assertEqual(order, ["migrate completed", "release backing", "grow KV"])
        self.assertEqual(loan.allocator.free_pages.tolist(), [1, 2, 5, 6, 7, 8])

    def test_decode_mailbox_reads_hits_while_miss_rows_are_still_unloaded(self):
        """Snapshot pre-plan hits; fill must join only after their computation."""
        from sglang.srt.layers.moe.qwen38_cache_mailbox import DecodeMissMailbox

        operations = []
        pending_copies = []

        class Event:
            def record(self, stream):
                pass

            def query(self):
                return True

        class Stream:
            def wait_event(self, event):
                if event is filled:
                    operations.append("join fill")
                    for copy in pending_copies:
                        copy()

        compute, fill_stream = Stream(), Stream()
        done, ready, filled = Event(), Event(), Event()
        ids = torch.tensor([[2, 1]], dtype=torch.int32)
        values = torch.tensor([[float("nan"), float("nan")], [4.0, 8.0]])
        state = {
            "lookup": torch.tensor([-1, 1, -1]),
            "counts": torch.zeros(4, dtype=torch.int32),
            "keys": torch.tensor([-1, 1]),
            "frequency": torch.zeros(2),
            "ages": torch.zeros(2, dtype=torch.int64),
            "generation": torch.zeros(2, dtype=torch.int64),
            "clock": torch.zeros(1, dtype=torch.int64),
            "plan": torch.zeros(12, dtype=torch.int32),
        }
        r = SimpleNamespace(
            device="cpu",
            layer_computed={},
            warm_ready={},
            jobs=SimpleNamespace(maxsize=4),
            device_states={0: state},
            cache_policy="lfu",
            cache_protected_fraction=0.7,
            caches={0: {"w13_weight": values}},
            layer_scales={0: {}},
            resident_alphas=False,
            fused_decode_plan_invocations=0,
            decode_device_gate_invocations=0,
            slots_for_layer=lambda _: 2,
        )
        mailbox = DecodeMissMailbox.__new__(DecodeMissMailbox)
        mailbox.runtime = r
        mailbox.last_computed = None
        mailbox.lock = threading.Lock()
        mailbox.pending = deque()
        mailbox.events = [done]
        mailbox.available = deque([done])
        mailbox.retained_external = {}
        mailbox.sampler = None
        mailbox.hit_stats = torch.zeros(1, 4, dtype=torch.int64)
        mailbox.layouts = {0: None}
        mailbox.failed_address, mailbox.fill_cta = 0, 1
        mailbox.fill_stream = fill_stream
        mailbox.overlap_events = {id(done): (ready, filled)}
        mailbox.overlap_invocations = mailbox.early_cache_invocations = 0

        def plan(route_ids, remapped, *_args):
            early_ids, late_ids = _args[-2:]
            early_ids.copy_(torch.tensor([[-1, 1]], dtype=torch.int32))
            late_ids.copy_(torch.tensor([[0, -1]], dtype=torch.int32))
            # Native boundary assigns the previously missing expert a slot.
            state["lookup"][2] = 0
            remapped.copy_(torch.tensor([[0, 1]], dtype=torch.int32))

        def enqueue_fill(*_args):
            pending_copies.append(lambda: values[0].copy_(torch.tensor([2.0, 3.0])))

        def early(route_ids, tensors):
            self.assertEqual(route_ids.tolist(), [[-1, 1]])
            self.assertTrue(torch.isnan(tensors["w13_weight"][0]).all())
            operations.append("cache hits")
            return values[1:2].clone()

        def execute(prepared):
            self.assertEqual(prepared.topk_ids.tolist(), [[0, -1]])
            result = prepared.early_output + values[0:1]
            return result

        native = SimpleNamespace(
            decode_plan_parallel_early=plan, mailbox_fill_grid=enqueue_fill
        )
        with (
            patch.object(torch.cuda, "current_stream", return_value=compute),
            patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
            patch(
                "sglang.srt.layers.moe.qwen38_cache_mailbox._module",
                return_value=native,
            ),
        ):
            result = mailbox.run(
                0,
                ids,
                execute,
                0,
                early_cache=early,
                during_fill=lambda: operations.append("shared"),
            )
        torch.testing.assert_close(result, torch.tensor([[6.0, 11.0]]))
        self.assertEqual(operations, ["shared", "cache hits", "join fill"])
        self.assertEqual(mailbox.early_cache_invocations, 1)

    def test_stage_shrink_retains_hot_nonprefix_rows_and_growth_keeps_them(self):
        """Changing layer offsets must not silently discard surviving experts."""
        from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        r = Qwen38ThreadedExpertRuntime.__new__(Qwen38ThreadedExpertRuntime)
        r.device = torch.device("cpu")
        r.manifest = SimpleNamespace(num_layers=2, num_experts=8)
        r.layer_cache_slots = [3, 3]
        storage = {"weight": torch.tensor([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])}
        r.cache_pool = storage
        r.staged_pool = SimpleNamespace(
            pool_views=lambda n: {k: v[:n] for k, v in storage.items()}
        )
        r.migration_scratch = {"weight": torch.empty(())}
        r.adaptive_metadata = AdaptiveMetadataBuffers(2, 8, 3, r.device)
        r.device_states = {}
        r.slot_ready, r.slot_last_use = {}, {}
        r.warm_ready, r.layer_computed, r.caches = {}, {}, {}
        r.policies = {}
        r.borrowed_arena = False
        r.cache_policy = "lfu"
        r.gate_warmed_layers = set()
        r.miss_mailbox = r.packed_transport = None
        stream = SimpleNamespace(wait_stream=lambda _: None)
        r.transfer = r.metadata_stream = stream
        r.adaptive_migrated_experts = r.adaptive_evicted_experts = 0
        records = {
            0: [(0, 1.0, 1), (1, 5.0, 2), (2, 9.0, 3)],
            1: [(3, 1.0, 1), (4, 8.0, 2), (5, 6.0, 3)],
        }
        for layer in range(2):
            state = r._state(layer)
            r.adaptive_metadata.restore(layer, state, records[layer])

        def snapshot():
            return [
                tuple(
                    r.device_states[layer][name].tolist()
                    for name in ("keys", "frequency", "ages")
                )
                for layer in range(2)
            ]

        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
        ):
            r._preserve_staged_layout([2, 2], 4, snapshot())
            self.assertEqual(r.adaptive_evicted_experts, 2)
            self.assertEqual(r.device_states[0]["keys"].tolist(), [2, 1])
            self.assertEqual(r.device_states[1]["keys"].tolist(), [4, 5])
            torch.testing.assert_close(
                storage["weight"][:4], torch.tensor([12.0, 11.0, 21.0, 22.0])
            )
            r._preserve_staged_layout([3, 3], 6, snapshot())
        self.assertEqual(r.adaptive_evicted_experts, 2)
        self.assertEqual(r.device_states[0]["keys"].tolist(), [2, 1, -1])
        self.assertEqual(r.device_states[1]["keys"].tolist(), [4, 5, -1])
        self.assertEqual(r.device_states[1]["lookup"][4].item(), 0)
        torch.testing.assert_close(
            r.caches[0]["weight"][:2], torch.tensor([12.0, 11.0])
        )
        torch.testing.assert_close(
            r.caches[1]["weight"][:2], torch.tensor([21.0, 22.0])
        )

    def test_moe_boundary_combines_both_pools_with_reused_output_buffer(self):
        """Exercise the real dispatch wrapper; replace only GPU computation."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        ids = torch.tensor([[4, 1, 3], [1, 4, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
        hidden = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
        all_values = torch.tensor(
            [[1.0, 1.0], [5.0, -1.0], [1.0, 1.0], [-4.0, 3.0], [2.0, 7.0]]
        )
        cache_values = torch.full((4, 2), float("nan"))
        cache_values[1], cache_values[3] = all_values[4], all_values[1]
        arena_values = torch.full_like(all_values, float("nan"))
        arena_values[3] = all_values[3]
        tensor_names = (
            "w13_weight",
            "w2_weight",
            "w13_input_scale_quant",
            "w13_blockscale_swizzled",
            "g1_alphas",
            "w2_input_scale_quant",
            "w2_blockscale_swizzled",
            "g2_alphas",
        )
        cache_tensors = dict.fromkeys(tensor_names)
        arena_tensors = dict.fromkeys(tensor_names)
        cache_tensors["w13_weight"] = cache_values
        arena_tensors["w13_weight"] = arena_values
        output_buffer = torch.empty_like(hidden)
        expected = (all_values[ids.long()] * weights[..., None]).sum(1) * hidden

        def reference_compute(x, route_ids, route_weights, tensors, **_kwargs):
            values = torch.zeros((*route_ids.shape, x.shape[-1]))
            valid = route_ids >= 0
            values[valid] = tensors["w13_weight"][route_ids[valid].long()]
            contribution = (values * route_weights[..., None]).sum(1)
            output_buffer.copy_(contribution * x)
            return output_buffer

        def cutlass_compute(batch, quant):
            return StandardCombineInput(
                hidden_states=reference_compute(
                    batch.hidden_states,
                    batch.topk_output.topk_ids,
                    batch.topk_output.topk_weights,
                    {"w13_weight": quant.w13_weight},
                )
            )

        prepared = PreparedExperts(
            arena_tensors, ids, 5, True, cache_tensors, {4: 1, 1: 3}, [3]
        )

        def wait_for_weights():
            # Cache computation must already have happened when arena joins.
            cache_expected = (all_values[ids[:, :2].long()] * weights[:, :2, None]).sum(
                1
            ) * hidden
            torch.testing.assert_close(output_buffer, cache_expected)
            arena_values[3] = all_values[3]

        prepared.wait_for_weights = wait_for_weights
        method = Qwen38CachedNvFp4MoeMethod(None, "fixture")
        method.layer_id = 0
        method.moe_runner_config = MoeRunnerConfig()
        method.swiglu_buffers = SimpleNamespace(get=lambda _: (None, None, None))
        method.runners = {n: SimpleNamespace(run=cutlass_compute) for n in (4, 5)}
        method.runtime = SimpleNamespace(
            run=lambda _layer, _ids, execute: execute(prepared)
        )
        with patch(
            "sglang.srt.layers.quantization.qwen38_cached_w4a16.cached_w4a16_moe",
            side_effect=reference_compute,
        ):
            for w4a16 in (False, True):
                method.prefill_use_w4a16 = w4a16
                arena_values.fill_(float("nan"))
                actual = method._run_experts(
                    hidden, weights, ids, None, execution_phase="prefill"
                )
                torch.testing.assert_close(actual, expected)
                # A large route array may select arena even when every unique
                # expert is cached. Poison arena to expose an unwanted read.
                mixed_prepared = prepared
                prepared = replace(
                    prepared,
                    topk_ids=ids[:, :2],
                    arena_experts=[],
                    wait_for_weights=None,
                )
                actual = method._run_experts(
                    hidden, weights[:, :2], ids[:, :2], None, execution_phase="prefill"
                )
                all_hit_expected = (
                    all_values[ids[:, :2].long()] * weights[:, :2, None]
                ).sum(1) * hidden
                torch.testing.assert_close(actual, all_hit_expected)
                prepared = mixed_prepared

    def test_prefill_workspace_bound_preserves_tail_with_reused_backend_output(self):
        """MoE subchunks must preserve every token despite backend output reuse."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        x = torch.arange(4101 * 2, dtype=torch.bfloat16).view(4101, 2)
        ids = torch.zeros((4101, 1), dtype=torch.int32)
        weights = torch.linspace(0.1, 0.9, 4101, dtype=torch.bfloat16).view(-1, 1)
        names = (
            "w13_weight",
            "w2_weight",
            "w13_input_scale_quant",
            "w13_blockscale_swizzled",
            "g1_alphas",
            "w2_input_scale_quant",
            "w2_blockscale_swizzled",
            "g2_alphas",
        )
        prepared = PreparedExperts(dict.fromkeys(names), ids, 1, True)
        scratch = torch.empty(2048, 2, dtype=torch.bfloat16)

        def run(batch, quant):
            n = batch.hidden_states.shape[0]
            self.assertLessEqual(n, 2048)
            scratch[:n].copy_(batch.hidden_states * batch.topk_output.topk_weights)
            return StandardCombineInput(hidden_states=scratch[:n])

        method = Qwen38CachedNvFp4MoeMethod(None, "fixture")
        method.layer_id = 0
        method.prefill_use_w4a16 = False
        method.moe_runner_config = MoeRunnerConfig()
        method.swiglu_buffers = SimpleNamespace(get=lambda _: (None, None, None))
        method.runners = {1: SimpleNamespace(run=run)}
        method.runtime = SimpleNamespace(
            prefill_direct_cache=True,
            run=lambda _layer, _ids, execute: execute(prepared),
        )
        actual = method._run_experts(x, weights, ids, None, execution_phase="prefill")
        torch.testing.assert_close(actual, x * weights)

    def test_decode_early_output_is_added_once_after_fill(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        ids = torch.tensor([[-1, 0, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        values = torch.tensor([[3.0, 4.0], [7.0, 9.0]])
        early = torch.tensor([[4.9, 6.3]])
        prepared = PreparedExperts(
            {"w13_weight": values},
            ids,
            2,
            False,
            early_output=early,
        )
        method = Qwen38CachedNvFp4MoeMethod(None, "fixture")
        method.layer_id = 0
        method.use_w4a16 = True
        method.moe_runner_config = MoeRunnerConfig()
        method.runtime = SimpleNamespace(
            run=lambda _layer, _ids, execute: execute(prepared)
        )

        def reference_compute(x, route_ids, route_weights, tensors, **_kwargs):
            valid = route_ids >= 0
            selected = torch.zeros((*route_ids.shape, x.shape[-1]))
            selected[valid] = tensors["w13_weight"][route_ids[valid].long()]
            return (selected * route_weights[..., None]).sum(1)

        with patch(
            "sglang.srt.layers.quantization.qwen38_cached_w4a16.cached_w4a16_moe",
            side_effect=reference_compute,
        ):
            actual = method._run_experts(
                torch.ones(1, 2), weights, ids, None, execution_phase="decode"
            )
        torch.testing.assert_close(actual, torch.tensor([[5.8, 7.5]]))

    def test_partition_preserves_weighted_sum_and_only_reads_populated_rows(self):
        """Inactive routes use -1; active routes only read populated rows."""
        ids = torch.tensor([[4, 1, 3], [1, 4, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.4, 0.0]])
        cache = {4: 1, 1: 3}
        a, aw, b, bw = partition_prefill_routes(ids, weights, cache, [3])
        # Poison every unpopulated row: accidental masked reads produce NaN.
        cache_values = torch.full((4, 2), float("nan"))
        cache_values[1] = torch.tensor([2.0, 7.0])
        cache_values[3] = torch.tensor([5.0, -1.0])
        arena_values = torch.full((5, 2), float("nan"))
        arena_values[3] = torch.tensor([-4.0, 3.0])

        def contribution(values, routes, weights):
            selected = torch.zeros((*routes.shape, values.shape[-1]))
            valid = routes >= 0
            selected[valid] = values[routes[valid].long()]
            return (selected * weights[..., None]).sum(1)

        actual = contribution(cache_values, a, aw)
        actual += contribution(arena_values, b, bw)
        expected = torch.tensor([[-0.1, 2.6], [3.8, 2.2]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aw + bw, weights)
        self.assertTrue(set(a.flatten().tolist()) <= {-1, 1, 3})
        self.assertEqual(set(b.flatten().tolist()), {-1, 3})
        self.assertTrue(torch.equal(a[aw == 0], torch.full_like(a[aw == 0], -1)))
        self.assertTrue(torch.equal(b[bw == 0], torch.full_like(b[bw == 0], -1)))
        # All-hit calls execute only the cache contribution.
        a, aw, b, bw = partition_prefill_routes(ids[:, :2], weights[:, :2], cache, [])
        self.assertIsNone(b)
        self.assertIsNone(bw)
        torch.testing.assert_close(aw, weights[:, :2])


if __name__ == "__main__":
    unittest.main()
