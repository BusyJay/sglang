"""Exercise production boundary methods without a CUDA/PyTorch installation."""

import ast
import contextlib
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


def method(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class TestAdaptiveBoundary(unittest.TestCase):
    def test_borrowed_slots_do_not_switch_short_prefill_transport(self):
        fn = method(
            "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            "_use_prefill_arena",
            {},
        )
        runtime = SimpleNamespace(
            prefill_layer_slots=(133,),
            slots_for_layer=lambda _: 147,
            prefill_arena=True,
        )
        # A 14-token top-10 prefill used the batched arena before KV lending.
        self.assertTrue(fn(runtime, 0, 140))
        self.assertFalse(fn(runtime, 0, 30))
        runtime.prefill_arena = False
        self.assertFalse(fn(runtime, 0, 140))
        runtime.prefill_arena = True
        runtime.slots_for_layer = lambda _: 100
        self.assertTrue(fn(runtime, 0, 120))

    def test_existing_result_must_cover_snapshot_without_waiting(self):
        fn = method(
            "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            "adaptive_after_result",
            {},
        )
        runtime = SimpleNamespace(
            adaptive_pending=([1], object(), SimpleNamespace(query=lambda: False)),
            poisoned=False,
        )
        with self.assertRaisesRegex(RuntimeError, "result completion"):
            fn(runtime)
        self.assertTrue(runtime.poisoned)

    def test_migration_publishes_on_snapshot_stream_and_orders_transports(self):
        events = []

        class Host:
            def __getitem__(self, _):
                return self

            def tolist(self):
                events.append("read snapshot")
                return [1]

        stream = object()

        @contextlib.contextmanager
        def stream_context(value):
            self.assertIs(value, stream)
            events.append("enter stream")
            yield

        namespace = dict(
            time=time,
            torch=SimpleNamespace(
                no_grad=contextlib.nullcontext,
                cuda=SimpleNamespace(stream=stream_context),
            ),
            LayerExpertCachePolicy=lambda *_: object(),
            logger=SimpleNamespace(info=lambda *_: None),
        )
        fn = method(
            "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            "adaptive_after_result",
            namespace,
        )
        r = SimpleNamespace(
            adaptive_pending=([2], stream, SimpleNamespace(query=lambda: True)),
            adaptive_lock=threading.Lock(),
            lock=threading.Lock(),
            layer_cache_slots=[1],
            manifest=SimpleNamespace(num_experts=4, num_layers=1),
            adaptive_metadata=SimpleNamespace(
                host=[{k: Host() for k in ("keys", "frequency", "ages")}]
            ),
            _migrate_pool=lambda target, snapshot: (
                events.append("migrate") or {0: []},
                1,
                0,
            ),
            _assign_pool_views=lambda _: events.append("assign"),
            miss_mailbox=SimpleNamespace(
                refresh_targets=lambda **kw: events.append(("mailbox", kw))
            ),
            packed_transport=SimpleNamespace(
                refresh_targets=lambda **kw: events.append(("packed", kw))
            ),
            device_states={},
            slot_ready={},
            slot_last_use={},
            warm_ready={},
            layer_computed={},
            _state=lambda _: None,
            _restore_device_policy=lambda _: events.append("restore"),
            cache_policy="lru",
            metadata_stream=SimpleNamespace(
                wait_stream=lambda s: events.append(("metadata waits", s))
            ),
            transfer=SimpleNamespace(
                wait_stream=lambda s: events.append(("transfer waits", s))
            ),
            adaptive_rebalances=0,
            gate_warmed_layers=set(),
            adaptive_migrated_experts=0,
            adaptive_evicted_experts=0,
        )
        fn(r)
        self.assertEqual(r.layer_cache_slots, [2])
        self.assertIsNone(r.adaptive_pending)
        self.assertLess(events.index("read snapshot"), events.index("restore"))
        self.assertLess(events.index("migrate"), events.index("assign"))
        self.assertIn(("mailbox", {"non_blocking": True}), events)
        self.assertIn(("packed", {"non_blocking": True}), events)
        self.assertIn(("transfer waits", stream), events)

    def test_wrong_overlap_result_cannot_apply_maintenance(self):
        namespace = {}
        fn = method(
            "python/sglang/srt/mem_cache/allocator/paged.py",
            "cache_maintenance_after_result",
            namespace,
        )
        events = []
        result = object()
        allocator = SimpleNamespace(
            maintenance_result=result,
            staged_loan=SimpleNamespace(after_result=lambda: events.append("return")),
            expert_runtime=SimpleNamespace(
                adaptive_after_result=lambda: events.append("adaptive")
            ),
            cache_maintenance_needs_result=lambda: False,
        )
        fn(allocator, object())
        self.assertEqual(events, [])
        fn(allocator, result)
        self.assertEqual(events, ["return", "adaptive"])
        self.assertIsNone(allocator.maintenance_result)


class TestVmmRelease(unittest.TestCase):
    def test_exact_release_and_close_bookkeeping(self):
        events = []
        driver = SimpleNamespace(
            cuMemUnmap=lambda address, size: events.append(("unmap", address, size)),
            cuMemRelease=lambda handle: events.append(("release", handle)),
        )
        release = method(
            "python/sglang/srt/utils/cuda_vmm_utils.py",
            "release_mapping",
            dict(_get_cuda_driver=lambda: driver, check_drv=lambda *_: None),
        )
        reservation = SimpleNamespace(
            _closed=False, base=1000, _mappings=[(1000, 64, 1), (1064, 64, 2)]
        )
        with self.assertRaises(ValueError):
            release(reservation, 64, 32)
        self.assertEqual(events, [])
        release(reservation, 64, 64)
        self.assertEqual(events, [("unmap", 1064, 64), ("release", 2)])
        self.assertEqual(reservation._mappings, [(1000, 64, 1)])
        with self.assertRaises(ValueError):
            release(reservation, 64, 64)


class TestAdaptiveStages(unittest.TestCase):
    def test_stage_budgets_use_physical_bytes_and_allow_return_reborrow_cycles(self):
        events = []
        ns = {}
        configure = method(
            "python/sglang/srt/layers/moe/qwen38_staged_pool.py", "configure_loans", ns
        )
        grow = method(
            "python/sglang/srt/layers/moe/qwen38_staged_pool.py", "grow_stage", ns
        )
        release = method(
            "python/sglang/srt/layers/moe/qwen38_staged_pool.py", "release_stage", ns
        )
        mappings = {}

        def commit(offset, end):
            events.append(("commit", end))
            mappings[offset] = end

        def trim(offset, keep):
            events.append(("trim", keep))
            mappings[offset] = keep

        pool = SimpleNamespace(
            base_budget=4,
            maximum_budget=20,
            budgets=None,
            stage_index=0,
            loan_mappings=[],
            rows={"w": 10},
            offsets={"w": 0},
            physical_bytes=lambda n: ((n * 10 + 15) // 16) * 16,
            arena=SimpleNamespace(
                _align=lambda n: ((n + 15) // 16) * 16,
                commit_range=commit,
                release_tail=trim,
            ),
        )
        pool.grow_stage = lambda i, **kw: grow(pool, i, **kw)
        budgets = configure(pool, [96, 48, 0])
        self.assertEqual(budgets, [14, 9, 4])
        self.assertEqual(mappings[0], 144)
        for _ in range(3):
            release(pool, 2)
            self.assertEqual(mappings[0], 48)
            grow(pool, 0)
            self.assertEqual(mappings[0], 144)
        release(pool, 1)
        self.assertEqual(mappings[0], 96)
        grow(pool, 0)
        self.assertEqual(mappings[0], 144)

    def test_cached_tail_prevents_reborrow_even_with_low_total_usage(self):
        namespace = {}
        fn = method(
            "python/sglang/srt/mem_cache/qwen38_kv_stages.py", "_reborrow", namespace
        )
        # This path returns before torch.sort; supply a minimal import stub.
        import sys
        from unittest.mock import patch

        free = SimpleNamespace(tolist=lambda: [False, True, True, True, False])
        loan = SimpleNamespace(
            probe=SimpleNamespace(query=lambda: True),
            free_snapshot=free,
            allocator=SimpleNamespace(backed_num_pages=4, page_size=64),
            used_tokens=lambda: 64,
            reserve=1,
            policy=SimpleNamespace(capacities=[128, 192, 256], index=2),
            runtime=SimpleNamespace(kv_reborrow_probe=True),
        )
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            fn(loan)
        self.assertEqual(loan.policy.index, 2)
        self.assertIsNone(loan.probe)
        self.assertFalse(loan.runtime.kv_reborrow_probe)

    def test_return_then_short_demand_schedules_reborrow_probe(self):
        fn = method(
            "python/sglang/srt/mem_cache/qwen38_kv_stages.py", "before_forward", {}
        )
        r = SimpleNamespace(pending_kv_stage=None, kv_reborrow_probe=False)
        loan = SimpleNamespace(
            runtime=r,
            reserve=0,
            policy=SimpleNamespace(index=0, capacities=[80000, 160000, 220032]),
            used_tokens=lambda: 71808,
        )
        fn(loan, 8192)
        self.assertEqual(r.pending_kv_stage, 1)
        loan.policy.index = 1
        r.pending_kv_stage = None
        loan.used_tokens = lambda: 1024
        fn(loan, 8192)
        self.assertTrue(r.kv_reborrow_probe)
        self.assertIsNone(r.pending_kv_stage)

    def test_reborrow_releases_kv_before_backing_experts_and_removes_only_free_tail(
        self,
    ):
        import sys
        from unittest.mock import patch

        events = []

        class Pages(list):
            pass

        fn = method("python/sglang/srt/mem_cache/qwen38_kv_stages.py", "_reborrow", {})
        allocator = SimpleNamespace(
            backed_num_pages=4,
            page_size=64,
            get_all_free_pages=lambda: Pages([4, 2, 3]),
            staged_pages=[],
            num_staged_pages=0,
        )
        loan = SimpleNamespace(
            probe=SimpleNamespace(query=lambda: True),
            free_snapshot=SimpleNamespace(
                tolist=lambda: [False, False, True, True, True]
            ),
            allocator=allocator,
            used_tokens=lambda: 64,
            reserve=8,
            policy=SimpleNamespace(capacities=[128, 192, 256], index=2),
            pool=SimpleNamespace(
                _post_capture_owner=SimpleNamespace(
                    shrink_capacity=lambda n: events.append(("KV shrink", n))
                )
            ),
            runtime=SimpleNamespace(
                kv_reborrow_probe=True,
                staged_pool=SimpleNamespace(
                    grow_stage=lambda i: events.append(("expert grow", i))
                ),
                grow_staged_layout=lambda i: events.append(("publish", i)),
                expert_loan_budgets=[14, 9, 4],
            ),
        )
        fn.__globals__["logger"] = SimpleNamespace(info=lambda *_: None)
        with patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(sort=lambda p: (Pages(sorted(p)), None))},
        ):
            fn(loan)
        self.assertEqual(
            events, [("KV shrink", 128), ("expert grow", 0), ("publish", 0)]
        )
        self.assertEqual(allocator.free_pages, [2])
        self.assertEqual(allocator.backed_num_pages, 2)
        self.assertEqual(loan.policy.index, 0)
