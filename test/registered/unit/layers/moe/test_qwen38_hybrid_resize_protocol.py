# SPDX-License-Identifier: Apache-2.0
"""Execute runtime resize with byte-backed tensors; check DMA/publication order."""

import contextlib
import importlib.util
import queue
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[5]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Row:
    def __init__(self, storage, slot, operations, fail):
        self.storage, self.slot, self.operations, self.fail = (
            storage,
            slot,
            operations,
            fail,
        )

    def copy_(self, source):
        self.operations.append(("move", source.slot, self.slot))
        if self.fail:
            raise RuntimeError("injected transfer failure")
        self.storage[self.slot] = source.storage[source.slot]


class Tensor:
    def __init__(self, storage, operations, fail=False):
        self.storage, self.operations, self.fail = storage, operations, fail

    def __getitem__(self, slot):
        return Row(self.storage, slot, self.operations, self.fail)


class TestGlobalResizeProtocol(CustomTestCase):
    def setUp(self):
        names = (
            "sglang.srt.layers.moe.qwen38_hybrid_policy",
            "hybrid_resize_subject",
            "global_kv_resize_subject",
        )
        previous = {name: sys.modules.get(name) for name in names}

        def restore_modules():
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        self.addCleanup(restore_modules)
        self.operations = []
        operations = self.operations

        class Event:
            def record(self, stream):
                operations.append(("record",))

            def synchronize(self):
                operations.append(("dma_complete",))

        class Stream:
            def wait_stream(self, stream):
                operations.append(("dependency",))

        self.stream = Stream()
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(
            current_stream=lambda _: self.stream,
            stream=lambda _: contextlib.nullcontext(),
            Event=Event,
        )
        env_stub = ModuleType("sglang.srt.environ")
        env_stub.envs = SimpleNamespace()
        self.policy_module = load(
            "sglang.srt.layers.moe.qwen38_hybrid_policy",
            "python/sglang/srt/layers/moe/qwen38_hybrid_policy.py",
        )
        with patch.dict(
            sys.modules, {"torch": torch_stub, "sglang.srt.environ": env_stub}
        ):
            self.runtime_module = load(
                "hybrid_resize_subject", "python/sglang/srt/layers/moe/qwen38_hybrid.py"
            )

    def subject(self, fail=False):
        policy = self.policy_module.GlobalLfu(2, 8, [2, 2])
        policy.seed(0, {0: 0, 1: 1})
        policy.seed(1, {0: 2, 1: 3})
        policy.frequency[0][:2] = [1, 2]
        policy.frequency[1][:2] = [8, 9]
        storage = [bytes([key[0], key[1]]) * 32 for key in policy.keys] + [None] * 4
        tensor = Tensor(storage, self.operations, fail)
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.lookups = {}
        subject.readable = {}
        subject.b12x = {}
        subject.policy = policy
        subject._seed = lambda: None
        subject.drain = lambda: self.operations.append(("drain",))
        subject.fill_stream = self.stream
        subject.stats = __import__("collections").Counter()
        subject.failure = None
        subject.runtime = SimpleNamespace(
            staged_pool=SimpleNamespace(pool_views=lambda budget: {"w": tensor}),
            layer_computed={},
            device="cuda",
            transfer=self.stream,
            manifest=SimpleNamespace(num_layers=2),
            _refresh_staged_arena=lambda: self.operations.append(("arena_refresh",)),
            adaptive_migrated_experts=0,
            adaptive_evicted_experts=0,
            poisoned=False,
        )
        commit = policy.commit_resize

        def publish(plan):
            self.operations.append(("publish",))
            commit(plan)

        policy.commit_resize = publish
        return subject, storage

    def test_readback_failure_releases_cpu_result_waiter(self):
        from concurrent.futures import Future

        subject, _ = self.subject()
        result = Future()
        released = []
        subject.cpu = SimpleNamespace(release_worker=lambda: released.append(True))
        subject.cpu_jobs = queue.Queue()

        def failed_readback():
            raise RuntimeError("injected readback failure")

        ready = SimpleNamespace(synchronize=failed_readback)
        subject.cpu_jobs.put((0, None, None, None, ready, None, {}, result))
        subject.cpu_jobs.put(None)
        with patch.object(
            self.runtime_module.torch.cuda,
            "device",
            lambda _: contextlib.nullcontext(),
            create=True,
        ):
            subject._cpu_work()
        self.assertTrue(result.done())
        with self.assertRaisesRegex(RuntimeError, "readback failure"):
            result.result()
        self.assertEqual(released, [True])
        self.assertEqual(subject.cpu_jobs.unfinished_tasks, 0)

    def maintenance_subject(self):
        subject, _ = self.subject()
        subject.closed = False
        subject.timeout = 1
        subject.active = {}
        subject.jobs = queue.Queue()
        subject.pending = None
        subject.current = None
        subject.seeded = True
        subject.policy.frequency[0][:2] = [1, 1]
        subject.policy.frequency[0][3] = 3
        return subject

    def test_reader_and_copy_waits_do_not_block_other_layers(self):
        """A slow reader/copy must not stall another layer's publication."""
        subject = self.maintenance_subject()
        p = subject.policy
        p.frequency[1][3] = 12
        reader_ready = [False]
        copy_ready = [False]
        copies = []
        for layer in (0, 1):
            epoch, _ = p.begin(layer)
            event = SimpleNamespace(query=lambda: reader_ready[0])
            if layer == 1:
                event = SimpleNamespace(query=lambda: True)
            subject.pending = (layer, epoch, (3,), event)
            subject.release_previous()
        jobs = [subject.jobs.get_nowait() for _ in range(2)]
        subject._copy = lambda *args: SimpleNamespace(query=lambda: copy_ready[0])
        readers = subject._submit_ready(jobs, copies)
        self.assertEqual(len(readers), 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(p.layers[0].mapping, {0: 0, 1: 1})
        self.assertEqual(subject._begin_layer(0), (None, {}))
        self.assertEqual(subject._begin_layer(1), (None, {1: 3}))
        self.assertNotIn(3, p.layers[1].mapping)
        self.assertEqual(subject._complete_copies(copies), copies)
        copy_ready[0] = True
        self.assertEqual(subject._complete_copies(copies), [])
        self.assertTrue(subject.active[1].done())
        self.assertFalse(subject.active[0].done())
        self.assertIn(3, subject._begin_layer(1)[1])
        self.assertEqual(p.layers[0].mapping, {0: 0, 1: 1})
        reader_ready[0] = True
        copies = []
        self.assertEqual(subject._submit_ready(readers, copies), [])
        subject._complete_copies(copies)
        self.assertEqual(subject.jobs.unfinished_tasks, 0)

    def test_busy_layer_observations_count_without_new_mutation(self):
        subject = self.maintenance_subject()
        p = subject.policy
        epoch, _ = p.begin(0)
        subject.pending = (0, epoch, (3,), SimpleNamespace(query=lambda: True))
        subject.release_previous()
        copies = []
        subject._copy = lambda *args: SimpleNamespace(query=lambda: False)
        subject._submit_ready([subject.jobs.get_nowait()], copies)
        mapping = subject._begin_layer(0)[1]
        self.assertEqual(mapping, {1: 1})
        subject.pending = (0, None, (3,), None)
        subject.release_previous()
        observation = subject.jobs.get_nowait()
        self.assertEqual(observation, ("observe", 0, (3,)))
        p.observe(*observation[1:])
        subject.jobs.task_done()
        self.assertEqual(subject._begin_layer(0)[1], mapping)
        self.assertEqual(p.keys[0], (0, 3))
        self.assertNotIn(3, p.layers[0].mapping)
        copies[0][3].query = lambda: True
        subject._complete_copies(copies)
        self.assertEqual(subject.jobs.unfinished_tasks, 0)
        self.assertIn(3, subject._begin_layer(0)[1])

    def test_failed_copy_never_publishes_and_unblocks_drain(self):
        subject = self.maintenance_subject()
        epoch, _ = subject.policy.begin(0)
        subject.pending = (0, epoch, (3,), SimpleNamespace(query=lambda: True))
        subject.release_previous()

        def failed_query():
            raise RuntimeError("injected asynchronous copy failure")

        subject._copy = lambda *args: SimpleNamespace(query=failed_query)
        copies = []
        subject._submit_ready([subject.jobs.get_nowait()], copies)
        subject._complete_copies(copies)
        self.assertNotIn(3, subject.policy.layers[0].mapping)
        with self.assertRaisesRegex(RuntimeError, "asynchronous copy failure"):
            subject.active[0].result()
        self.assertEqual(subject.jobs.unfinished_tasks, 0)
        with self.assertRaisesRegex(RuntimeError, "background failure"):
            subject._begin_layer(0)

    def test_worker_publishes_ready_layer_while_another_reader_is_busy(self):
        import threading

        subject = self.maintenance_subject()
        subject.policy.frequency[1][3] = 12
        reader_ready = threading.Event()
        submitted = [threading.Event(), threading.Event()]
        copied = [threading.Event(), threading.Event()]

        def copy(layer, fills, done):
            submitted[layer].set()
            return SimpleNamespace(query=copied[layer].is_set)

        subject._copy = copy
        for layer in (0, 1):
            epoch, _ = subject.policy.begin(layer)
            event = SimpleNamespace(
                query=reader_ready.is_set if layer == 0 else lambda: True
            )
            subject.pending = (layer, epoch, (3,), event)
            subject.release_previous()
        with patch.object(
            self.runtime_module.torch.cuda,
            "device",
            lambda _: contextlib.nullcontext(),
            create=True,
        ):
            worker = threading.Thread(target=subject._work)
            worker.start()
            try:
                self.assertTrue(submitted[1].wait(2))
                self.assertFalse(submitted[0].is_set())
                self.assertEqual(subject._begin_layer(0), (None, {}))
                copied[1].set()
                subject.active[1].result(timeout=2)
                self.assertEqual(subject.policy.layers[1].mapping[3], 2)
                self.assertFalse(subject.active[0].done())
                reader_ready.set()
                self.assertTrue(submitted[0].wait(2))
                copied[0].set()
                subject._join_queue(subject.jobs)
                self.assertEqual(subject.policy.layers[0].mapping[3], 0)
                self.assertEqual(subject.jobs.unfinished_tasks, 0)
            finally:
                reader_ready.set()
                for event in copied:
                    event.set()
                subject.jobs.put(None)
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_borrowed_arena_has_no_persistent_cache_owners(self):
        subject = self.maintenance_subject()
        subject.runtime.borrowed_arena = True
        subject.runtime.manifest.num_experts = 2
        subject.b12x = {0: object()}
        reserved = subject._reserved_slots(4)
        self.assertEqual(reserved, {0, 2, 3})
        plan = subject.policy.plan_resize(4, reserved=reserved)
        subject.policy.commit_resize(plan)
        self.assertEqual(subject.policy.owner, [-1, 1, -1, -1])
        self.assertEqual(subject.policy.layers[1].mapping, {1: 1})

    def test_stop_keeps_source_alive_until_outstanding_dma_retires(self):
        import threading

        subject = self.maintenance_subject()
        stop_received = threading.Event()
        stop_polled = threading.Event()
        submitted = threading.Event()
        copied = threading.Event()

        class Jobs(queue.Queue):
            def get(self, *args, **kwargs):
                job = super().get(*args, **kwargs)
                if job is None:
                    stop_received.set()
                return job

        def query():
            if stop_received.is_set():
                stop_polled.set()
            return copied.is_set()

        def copy(*args):
            submitted.set()
            return SimpleNamespace(query=query)

        subject.jobs = Jobs()
        subject._copy = copy
        epoch, _ = subject.policy.begin(0)
        subject.pending = (0, epoch, (3,), SimpleNamespace(query=lambda: True))
        subject.release_previous()
        with patch.object(
            self.runtime_module.torch.cuda,
            "device",
            lambda _: contextlib.nullcontext(),
            create=True,
        ):
            worker = threading.Thread(target=subject._work)
            worker.start()
            try:
                self.assertTrue(submitted.wait(2))
                subject.jobs.put(None)
                self.assertTrue(stop_received.wait(2))
                self.assertTrue(stop_polled.wait(2))
                self.assertTrue(worker.is_alive())
                self.assertFalse(subject.active[0].done())
            finally:
                copied.set()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(subject.policy.layers[0].mapping[3], 0)
        self.assertEqual(subject.jobs.unfinished_tasks, 0)

    def test_all_moves_complete_before_mapping_and_tail_release(self):
        subject, storage = self.subject()
        subject.resize(2)
        self.operations.append(("unmap_tail",))
        self.assertEqual(storage[:2], [bytes([1, 1]) * 32, bytes([1, 0]) * 32])
        self.assertEqual(subject.policy.layers[0].mapping, {})
        self.assertEqual(set(subject.policy.layers[1].mapping.values()), {0, 1})
        names = [item[0] for item in self.operations]
        self.assertLess(names.index("drain"), names.index("move"))
        self.assertLess(
            max(i for i, n in enumerate(names) if n == "move"),
            names.index("dma_complete"),
        )
        self.assertLess(names.index("dma_complete"), names.index("publish"))
        self.assertLess(names.index("publish"), names.index("unmap_tail"))

    def test_copy_failure_poisoned_without_publication(self):
        subject, _ = self.subject(fail=True)
        before = list(subject.policy.keys)
        with self.assertRaises(RuntimeError):
            subject.resize(2)
        self.assertEqual(subject.policy.keys, before)
        self.assertTrue(subject.runtime.poisoned)
        self.assertIsNotNone(subject.failure)
        self.assertNotIn("publish", [item[0] for item in self.operations])

    def test_growth_never_moves_or_evicts(self):
        subject, storage = self.subject()
        before = list(storage)
        subject.resize(7)
        self.assertEqual(storage, before)
        self.assertEqual(subject.policy.owner[4:], [None] * 3)
        self.assertEqual(subject.stats["resize_moves"], 0)
        self.assertEqual(subject.stats["resize_evictions"], 0)
        self.assertEqual(subject.stats["resize_retained_experts"], 4)

    def test_seed_restores_legacy_arena_before_importing_keys(self):
        import collections
        import threading

        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        physical = [2]
        policy = self.policy_module.GlobalLfu(1, 4, [1])
        operations = self.operations
        runtime = SimpleNamespace(
            lock=threading.Lock(),
            _raise_error=lambda: operations.append(("old_errors",)),
            _drain_arena_readers=lambda: operations.append(("old_readers_drained",)),
            layer_cache_slots=[1],
            device_states={0: {"keys": Tensor([1]), "frequency": Tensor([2.0])}},
            manifest=SimpleNamespace(num_layers=1),
            arena_residents={2},
            arena_layer=0,
        )

        def return_arena():
            physical[0] = 1
            operations.append(("old_bytes_restored",))

        runtime._return_arena = return_arena
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.runtime = runtime
        subject.lookups = {}
        subject.policy = policy
        subject.seeded = False
        subject.pool = {}
        subject.b12x = {}
        subject.stats = collections.Counter()
        subject._seed()
        self.assertEqual(policy.layers[0].mapping, {1: 0})
        self.assertEqual(physical[policy.layers[0].mapping[1]], 1)
        self.assertEqual(runtime.arena_residents, set())
        self.assertIn("old_bytes_restored", [op[0] for op in operations])

    def test_seed_propagates_legacy_failure_without_publishing(self):
        import threading

        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.seeded = False

        def fail():
            raise RuntimeError("legacy DMA failure")

        subject.runtime = SimpleNamespace(lock=threading.Lock(), _raise_error=fail)
        with self.assertRaisesRegex(RuntimeError, "legacy DMA failure"):
            subject._seed()
        self.assertFalse(subject.seeded)

    def test_b12x_dispatch_preserves_masked_route_outputs(self):
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.gpu_backend = "b12x"
        subject.pool = {}
        subject.runtime = SimpleNamespace(layer_scales={0: {}})
        subject.b12x = {
            0: SimpleNamespace(
                run=lambda x, ids, weights, tensors, **kwargs: sum(
                    (slot + 1) * weight
                    for slot, weight in zip(ids, weights)
                    if slot >= 0
                )
            )
        }
        self.runtime_module.torch.float32 = "float32"
        # Noncontiguous global slots and inactive routes cover all/mixed/no hits.
        for ids, expected in (([600, 1024], 1626), ([600, -1], 601), ([-1, -1], 0)):
            with self.subTest(ids=ids):
                self.assertEqual(subject._gpu(None, ids, [1, 1], 0), expected)

    def test_missing_or_failed_b12x_raises_without_baseline(self):
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.gpu_backend = "b12x"
        subject.pool = {}
        subject.runtime = SimpleNamespace(layer_scales={0: {}})
        subject.b12x = {}
        with self.assertRaisesRegex(RuntimeError, "B12X was selected"):
            subject._gpu(None, [-1], [1], 0)

        def fail(*args, **kwargs):
            raise RuntimeError("injected B12X launch failure")

        subject.b12x[0] = SimpleNamespace(run=fail)
        self.runtime_module.torch.float32 = "float32"
        with self.assertRaisesRegex(RuntimeError, "injected B12X launch failure"):
            subject._gpu(None, [700], [1], 0)

    def test_global_kv_return_compacts_even_when_legacy_preserve_is_false(self):
        module = load(
            "global_kv_resize_subject",
            "python/sglang/srt/mem_cache/qwen38_kv_stages.py",
        )
        operations = self.operations

        class Torch:
            @staticmethod
            def arange(*args, **kwargs):
                return []

            @staticmethod
            def cat(items):
                return []

            int64 = "int64"

        loan = module.StagedKVLoan.__new__(module.StagedKVLoan)
        loan.allocator = SimpleNamespace(
            backed_num_pages=2, page_size=1, device="cpu", free_pages=[]
        )
        loan.policy = SimpleNamespace(capacities=[2, 4, 6], index=0)
        loan.pool = SimpleNamespace(
            _post_capture_owner=SimpleNamespace(
                finalize=lambda _: operations.append(("kv_grow",))
            )
        )
        loan.runtime = SimpleNamespace(
            staged_trim_enqueued=True,
            staged_trim_done=SimpleNamespace(query=lambda: True),
            hybrid=object(),
            preserve_cache_on_resize=False,
            complete_staged_trim=lambda: operations.append(("compact_complete",)),
            staged_pool=SimpleNamespace(
                release_stage=lambda _: operations.append(("expert_unmap",))
            ),
            expert_loan_budgets=[8, 6, 4],
            pending_kv_stage=1,
        )
        with patch.dict(sys.modules, {"torch": Torch}):
            loan._return(1)
        self.assertEqual(
            [item[0] for item in operations],
            ["compact_complete", "expert_unmap", "kv_grow"],
        )


if __name__ == "__main__":
    unittest.main()
