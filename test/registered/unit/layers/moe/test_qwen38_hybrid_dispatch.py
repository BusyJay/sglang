# SPDX-License-Identifier: Apache-2.0
"""Execute hybrid dispatch with CPU torch and controlled CUDA boundaries."""

import collections
import contextlib
import importlib.util
import queue
import sys
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[5]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_experts(x, ids, weights, values):
    """Independent dense reference, preserving duplicate and inactive routes."""
    result = torch.zeros_like(x, dtype=torch.float32)
    for token, row in enumerate(ids.tolist()):
        for route, expert in enumerate(row):
            if expert >= 0:
                result[token] += (
                    x[token].float() * values[expert] * weights[token, route].float()
                )
    return result


class TestHybridDispatch(CustomTestCase):
    def setUp(self):
        self.operations = []
        self.events = []
        self.submitted = []
        self.cpu_calls = []
        self.gpu_calls = []
        self.decisions = []
        self.cpu_selection = set()
        self.admitted_experts = set()
        self.cache_copies = []
        self.preparations = []
        self.cpu_error = None
        self.arena_error = None
        operations = self.operations
        owner = self

        class Stream:
            def __init__(self, name):
                self.name = name
                self.cuda_stream = name

            def wait_stream(self, stream):
                operations.append(("wait_stream", self.name, stream.name))

            def wait_event(self, event):
                operations.append(("wait_event", self.name, event))

            def synchronize(self):
                operations.append(("stream_sync", self.name))

        class Event:
            def __init__(self):
                self.stream = None
                self.synchronized = False
                owner.events.append(self)

            def record(self, stream):
                self.stream = stream.name
                operations.append(("record", self.stream, self))

            def synchronize(self):
                self.synchronized = True
                operations.append(("sync", self))

            def query(self):
                return self.synchronized

        self.compute = Stream("compute")
        self.transfer = Stream("transfer")
        self.Event = Event
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name, replacement in {
            "current_stream": lambda *_: self.compute,
            "stream": lambda _: contextlib.nullcontext(),
            "device": lambda _: contextlib.nullcontext(),
            "Event": Event,
            "is_current_stream_capturing": lambda: False,
        }.items():
            stack.enter_context(patch.object(torch.cuda, name, replacement))
        # CPU CI has no pinned allocator or CUDA stream ownership.
        for name in ("empty_like", "full"):
            original = getattr(torch, name)

            def unpinned(*args, _original=original, **kwargs):
                kwargs.pop("pin_memory", None)
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(torch, name, unpinned))
        stack.enter_context(
            patch.object(
                torch.Tensor,
                "record_stream",
                lambda tensor, stream: operations.append(
                    ("tensor_stream", stream.name)
                ),
            )
        )
        self.hybrid = load_source(
            "hybrid_dispatch_subject", "python/sglang/srt/layers/moe/qwen38_hybrid.py"
        )
        self.partition = load_source(
            "hybrid_dispatch_partition",
            "python/sglang/kernels/ops/moe/qwen38_expert_cache.py",
        )

        def partition(ids, lookup, cache, arena, limit):
            active = ids >= 0
            slots = lookup[ids.clamp_min(0).long()]
            hits = active & (slots >= 0) & (slots < limit)
            cache.copy_(torch.where(hits, slots, -1))
            arena.copy_(torch.where(active & ~hits, ids, -1))

        stack.enter_context(
            patch.object(
                self.partition,
                "_partition_module",
                return_value=SimpleNamespace(partition=partition),
            )
        )
        stack.enter_context(
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.moe.qwen38_expert_cache": self.partition},
            )
        )
        self.expert_values = torch.tensor(
            [1.0, 3.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0]
        )
        self.mapping = {0: 3, 1: 5}
        self.pool = {"value": torch.full((16,), float("nan"))}
        for expert, slot in self.mapping.items():
            self.pool["value"][slot] = self.expert_values[expert]
        self.arena = {"value": torch.full((12,), float("nan"))}
        self.input_scale = torch.tensor([2.0])
        subject = self.hybrid.HybridCache.__new__(self.hybrid.HybridCache)
        self.subject = subject
        subject.runtime = SimpleNamespace(
            device="cpu",
            manifest=SimpleNamespace(num_layers=2, num_experts=12),
            layer_scales={
                0: {"input_scale": self.input_scale},
                1: {"input_scale": self.input_scale},
            },
            arena=self.arena,
            arena_last_use=None,
            transfer=self.transfer,
            layer_computed={},
            finish_layer=lambda layer: operations.append(("finish", layer)),
            _record_invocation=lambda arena, ids, layer: operations.append(
                ("invocation", arena, ids.clone(), layer)
            ),
        )

        def fill(layer, misses):
            operations.append(("fill", layer, tuple(misses)))
            for expert in misses:
                self.arena["value"][expert] = self.expert_values[expert]

        subject.runtime._fill_arena_selected = fill
        subject.policy = self.hybrid.GlobalLfu(2, 12, [8, 8])
        subject.policy.layers[0].mapping = self.mapping.copy()
        for expert, slot in self.mapping.items():
            subject.policy.keys[slot] = (0, expert)
            subject.policy.set_owner(slot, 0)
        subject.seeded = True
        subject.closed = False
        subject.failure = None
        subject.current = None
        subject.pending = None
        subject.active = {}
        subject.jobs = queue.Queue()
        subject.prepared_windows = set()
        subject.readable = {}
        subject.lookups = {}
        subject.host_buffers = {}
        subject.mapped_batch_pages = []
        subject.stats = collections.Counter()
        subject.pool = self.pool
        subject.gpu_backend = "b12x"
        subject.readback = Stream("readback")
        subject.result_stream = Stream("result")
        subject.timeout = 3

        def copy(layer, fills, done):
            self.cache_copies.append((layer, tuple(fills)))
            operations.append(("cache_copy", layer, tuple(fills)))
            for expert, slot in fills:
                self.pool["value"][slot] = self.expert_values[expert]
            copied = self.Event()
            copied.synchronized = True
            return copied

        subject._copy = copy

        def prepare(layer, epoch, requested, compute):
            self.preparations.append((layer, epoch, requested))
            operations.append(("prepare", layer, requested))
            policy = subject.policy
            policy.release(layer, epoch)
            state = policy.layers[layer]
            admitted = self.admitted_experts & (set(requested) - state.mapping.keys())
            slots = iter(slot for slot, key in enumerate(policy.keys) if key is None)
            fills = [(expert, next(slots)) for expert in sorted(admitted)]
            for expert, slot in fills:
                policy.keys[slot] = (layer, expert)
                policy.set_owner(slot, layer)
            if fills:
                subject._copy(layer, fills, self.Event())
            policy.commit(layer, epoch, fills)
            epoch, mapping = policy.begin(layer)
            subject.prepared_windows.add((layer, epoch))
            return epoch, mapping

        subject._prepare_window = prepare

        def select_cpu(tokens, rows, mapping, rejected):
            self.decisions.append((tokens, rows, dict(mapping), frozenset(rejected)))
            operations.append(("select_cpu", frozenset(rejected)))
            return frozenset(self.cpu_selection & rejected)

        subject.cost_model = SimpleNamespace(select_cpu=select_cpu)

        def gpu(x, ids, weights, tensors, output_dtype):
            self.assertEqual(output_dtype, torch.float32)
            self.assertIs(tensors["input_scale"], self.input_scale)
            is_arena = tensors["value"] is self.arena["value"]
            self.gpu_calls.append((is_arena, ids.clone()))
            operations.append(("gpu", is_arena))
            output = weighted_experts(x, ids, weights, tensors["value"])
            if is_arena and self.arena_error is not None:
                raise self.arena_error
            return output

        subject.b12x = {0: SimpleNamespace(run=gpu)}

        class Jobs(queue.Queue):
            def put_nowait(self, job):
                owner.submitted.append(job)
                super().put_nowait(job)

        subject.cpu_jobs = Jobs()

        def cpu(layer, x, ids, weights, result):
            self.assertTrue(self.submitted[-1][4].synchronized)
            operations.append(("cpu",))
            self.cpu_calls.append((ids.clone(), x.data_ptr(), result.data_ptr()))
            if self.cpu_error is not None:
                raise self.cpu_error
            result.copy_(weighted_experts(x, ids, weights, self.expert_values))

        subject.cpu = SimpleNamespace(
            run=cpu, release_worker=lambda: operations.append(("release_worker",))
        )

    def start_worker(self):
        worker = threading.Thread(target=self.subject._cpu_work, daemon=True)
        worker.start()

        def stop():
            self.subject.cpu_jobs.put(None)
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.subject.cpu_jobs.unfinished_tasks, 0)

        self.addCleanup(stop)

    def start_manager(self):
        worker = threading.Thread(target=self.subject._work, daemon=True)
        worker.start()

        def stop():
            self.subject.jobs.put(None)
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.subject.jobs.unfinished_tasks, 0)

        self.addCleanup(stop)

    def run_routes(self, ids, weights=None, x=None):
        ids = torch.tensor(ids, dtype=torch.int32)
        weights = (
            torch.ones(ids.shape)
            if weights is None
            else torch.tensor(weights, dtype=torch.float32)
        )
        x = torch.ones((ids.shape[0], 2), dtype=torch.bfloat16) if x is None else x
        before = ids.clone()
        expected = weighted_experts(x, ids, weights, self.expert_values).to(x.dtype)
        # Simulate completed, empty maintenance without changing the snapshot.
        state = self.subject.policy.layers[0]
        if state.phase == "computing":
            self.subject.prepared_windows.discard((0, state.epoch))
            self.subject.policy.release(0, state.epoch)
            self.subject.policy.commit(0, state.epoch, [])
        epoch, mapping = self.subject.policy.begin(0)
        self.subject.current = (0, epoch, mapping)
        output = self.subject.run(
            0, x, ids, weights, shared=lambda: self.operations.append(("shared",))
        )
        torch.testing.assert_close(ids, before)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.dtype, x.dtype)
        self.assertIs(self.subject.pending[-1], self.subject.runtime.layer_computed[0])
        return output

    def test_device_gpu_only_skips_mailbox_and_next_layer_publishes(self):
        device = load_source(
            "hybrid_device_dispatch_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = self.subject
        planner.valid = True
        planner.dirty = False
        planner.sequence = 0
        planner.has_cpu_samples = True
        planner.compute_stream = None
        planner.lookup = torch.empty((2, 12), dtype=torch.int32)
        planner.frequency = torch.empty((2, 12))
        planner.owners = torch.empty(16, dtype=torch.int32)
        planner.keys = torch.empty(16, dtype=torch.int32)
        planner.costs = torch.empty((0, 70), dtype=torch.float64)
        planner.device_stats = torch.zeros(7, dtype=torch.int64)
        planner.flag = torch.zeros(1, dtype=torch.int32)
        planner.heap_index = torch.empty(1, dtype=torch.int32)
        planner.layout = lambda layer: layer
        self.subject.device_planner = planner
        self.subject.runtime.miss_mailbox = SimpleNamespace(
            failed_address=1,
            fill_cta=8,
            lock=threading.Lock(),
            pending=collections.deque(),
        )

        class Page:
            def __init__(page, hidden, top_k, device):
                page.batch = SimpleNamespace(
                    head=torch.zeros(1, dtype=torch.int32),
                    layer=torch.zeros(1, dtype=torch.int32),
                    tokens=torch.zeros(1, dtype=torch.int32),
                    device_addresses=(12, 20, 28, 36, 0x1000, 0x100C),
                )

        def plan(
            ids,
            lookup,
            frequency,
            owners,
            keys,
            costs,
            cached,
            cpu,
            cache_plan,
            stats,
            layer,
            decay,
            admission,
        ):
            cached.copy_(ids + 1)
            cpu.fill_(-1)
            if layer == 1:
                cpu[0, 0] = ids[0, 0]
            self.operations.append(("plan", layer))

        def plan_stage(*args):
            plan(*args[:13])
            x, weights = args[13:15]
            ids = args[7]
            if bool((ids >= 0).any()):
                batch = self.subject.mapped_batch_pages[0].batch
                batch.layer.fill_(args[10])
                batch.tokens.fill_(x.shape[0])
                batch.head.fill_(args[-2])
                self.operations.append(("publish", args[10], args[-2]))

        self.subject._gpu = lambda x, ids, weights, layer, tensors=None, hybrid=None: (
            self.operations.append(("gpu", layer))
            or torch.zeros_like(
                x, dtype=torch.bfloat16 if hybrid is not None else torch.float32
            )
        )

        def merge(
            cpu_ids,
            gpu,
            completed,
            output,
            host_result,
            sequence,
            failed_address,
            completed_address=0,
        ):
            self.assertEqual(completed_address, 0x100C)
            output.copy_(gpu + (3 if bool((cpu_ids >= 0).any()) else 0))

        outputs = []
        with (
            patch.object(device, "hybrid_plan_stage_indexed", plan_stage),
            patch.object(device, "_MappedBatchPage", Page),
            patch.object(
                device,
                "hybrid_merge",
                side_effect=merge,
            ),
            patch.object(
                device,
                "fill_module",
                return_value=SimpleNamespace(mailbox_fill_grid=lambda *args: None),
            ),
        ):
            for layer in (0, 1):
                if layer == 1:
                    first_done = planner.last_compute_done
                    self.compute = type(self.compute)("graph-warmup")
                outputs.append(
                    planner.run(
                        layer,
                        torch.ones((1, 2)),
                        torch.tensor([[2, 3]], dtype=torch.int32),
                        torch.ones((1, 2)),
                        None,
                    )
                )
        torch.testing.assert_close(outputs[0], torch.zeros((1, 2)))
        torch.testing.assert_close(outputs[1], torch.full((1, 2), 3.0))
        self.assertLess(
            self.operations.index(("wait_event", "graph-warmup", first_done)),
            self.operations.index(("gpu", 1)),
        )
        self.assertEqual(
            [op for op in self.operations if op[0] == "publish"], [("publish", 1, 2)]
        )
        self.assertEqual(
            [op for op in self.operations if op[0] == "gpu"], [("gpu", 0), ("gpu", 1)]
        )
        self.assertTrue(self.subject.cpu_jobs.empty())
        self.assertEqual(self.subject.mapped_batch_pages[0].batch.head.item(), 2)
        self.assertEqual(self.subject.mapped_batch_pages[0].batch.layer.item(), 1)

    def test_device_cpu_result_is_written_before_completion_publication(self):
        device = load_source(
            "hybrid_device_cpu_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = self.subject
        published = []
        planner.publish = lambda sequence: published.append(
            (sequence, self.subject.mapped_batch_pages[0].batch.result[0].clone())
        )
        self.subject.cpu = SimpleNamespace(
            run=lambda layer, x, ids, weights, output: output.fill_(7)
        )
        batch = SimpleNamespace(
            x=torch.ones((3, 2)),
            ids=torch.tensor([[2, -1], [-1, -1], [-1, -1]], dtype=torch.int32),
            weights=torch.ones((3, 2)),
            result=torch.zeros((3, 2), dtype=torch.float32),
            head=torch.tensor([2], dtype=torch.int32),
            layer=torch.tensor([1], dtype=torch.int32),
            tokens=torch.tensor([1], dtype=torch.int32),
        )
        self.subject.mapped_batch_pages = [SimpleNamespace(batch=batch)]
        planner.cpu_task(1, batch, 1, 2)
        torch.testing.assert_close(batch.result[0], torch.full((2,), 7.0))
        self.assertEqual(published[0][0], 2)
        torch.testing.assert_close(published[0][1], torch.full((2,), 7.0))
        self.assertEqual(self.subject.stats["cpu_calls"], 1)

    def test_b12x_completion_publishes_to_mapped_page(self):
        device = load_source(
            "hybrid_device_complete_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = self.subject
        self.subject.gpu_backend = "b12x"
        completed = torch.zeros(1, dtype=torch.int32)
        self.subject.mapped_batch_pages = [
            SimpleNamespace(batch=SimpleNamespace(completed=completed))
        ]
        published = []

        def complete(target, sequence):
            self.assertIs(target, completed)
            target.fill_(sequence)
            published.append(sequence)

        with patch.object(
            device, "_module", return_value=SimpleNamespace(complete=complete)
        ):
            planner.publish(7)
        self.assertEqual(published, [7])
        self.assertEqual(completed.item(), 7)

    def test_cpu_poll_silently_skips_gpu_only_layer(self):
        device = load_source(
            "hybrid_device_poll_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = self.subject
        batch = SimpleNamespace(
            head=torch.zeros(1, dtype=torch.int32),
            layer=torch.zeros(1, dtype=torch.int32),
            tokens=torch.zeros(1, dtype=torch.int32),
        )
        self.subject.mapped_batch_pages = [SimpleNamespace(batch=batch)]

        def poll(head, seen):
            return 0 if head.item() != seen.item() else -1

        with patch.object(device, "_module", return_value=SimpleNamespace(poll=poll)):
            self.assertIsNone(planner.poll())
            batch.layer.fill_(1)
            batch.tokens.fill_(3)
            batch.head.fill_(2)
            self.assertEqual(planner.poll(), (1, batch, 3, 2))
            self.assertIsNone(planner.poll())
            batch.layer.fill_(3)
            batch.tokens.fill_(1)
            batch.head.fill_(4)
            self.assertEqual(planner.poll(), (3, batch, 1, 4))

    def test_mapped_batch_reuses_one_region_for_one_and_three_tokens(self):
        device = load_source(
            "hybrid_device_mapping_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        released = []
        module = SimpleNamespace(
            mailbox_register=lambda tensor: 0x1000,
            mailbox_unregister=lambda tensor: released.append(tensor.data_ptr()),
        )
        with patch.object(self.partition, "_module", return_value=module):
            page = device._MappedBatchPage(hidden=8, top_k=2, device="cpu")
            try:
                batch = page.batch
                self.assertEqual(len(page.mapping) % device.mmap.PAGESIZE, 0)
                self.assertEqual(
                    batch.device_addresses,
                    (0x100C, 0x103C, 0x1054, 0x106C, 0x1000, 0x10CC),
                )
                self.assertEqual(batch.head.item(), 0)
                self.assertEqual(batch.completed.item(), 0)
                batch.x[:3].fill_(5)
                self.assertEqual(batch.x[2].tolist(), [5] * 8)
            finally:
                del batch
                page.close("cpu")
        self.assertEqual(len(released), 1)

    def test_device_capacity_rejects_small_pool_before_gpu_submission(self):
        device = load_source(
            "hybrid_device_capacity_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = SimpleNamespace(
            policy=SimpleNamespace(experts=64, owner=[None] * 31),
            runtime=SimpleNamespace(manifest=SimpleNamespace(top_k=10)),
            _reserved_slots=lambda budget: {0},
        )
        planner._validate_capacity()
        with self.assertRaisesRegex(ValueError, "30 usable slots.*has 29"):
            planner._validate_capacity(30)
        self.assertFalse(self.gpu_calls)

    def test_device_cpu_failure_poison_and_releases_result_consumer(self):
        device = load_source(
            "hybrid_device_failure_subject",
            "python/sglang/srt/layers/moe/qwen38_hybrid_device.py",
        )
        planner = device.DeviceHybridPlanner.__new__(device.DeviceHybridPlanner)
        planner.cache = self.subject
        published, aborted = [], []
        planner.publish = published.append
        self.subject.runtime.miss_mailbox = SimpleNamespace(_abort=aborted.append)
        error = RuntimeError("native failure")

        def fail(*args):
            raise error

        self.subject.cpu = SimpleNamespace(run=fail)
        batch = SimpleNamespace(
            x=torch.ones((3, 2)),
            ids=torch.tensor([[2, -1], [-1, -1], [-1, -1]], dtype=torch.int32),
            weights=torch.ones((3, 2)),
            result=torch.zeros((3, 2)),
        )
        planner.cpu_task(0, batch, 1, 1)
        self.assertIs(self.subject.failure, error)
        self.assertTrue(self.subject.runtime.poisoned)
        self.assertEqual(aborted, [error])
        self.assertEqual(published, [1])

    def test_prefill_borrow_is_temporary_and_returns_all_tail_slots(self):
        subject = self.subject
        subject.runtime.borrowed_arena = True
        subject.policy = self.hybrid.GlobalLfu(2, 12, [8, 8])
        subject.policy.seed(0, {e: e for e in range(8)})
        subject.policy.seed(1, {e: e + 8 for e in range(8)})
        subject.drain = lambda: None
        subject._borrow_prefill_slots()
        self.assertEqual(subject.prefill_reserved, set(range(4, 16)))
        self.assertEqual(subject.policy.layers[0].mapping, {e: e for e in range(4)})
        self.assertFalse(subject.policy.layers[1].mapping)
        subject.runtime.arena_last_use = self.Event()
        subject._release_prefill_slots()
        self.assertFalse(subject.prefill_reserved)
        self.assertFalse(subject.policy.reserved)
        self.assertEqual(subject.policy.free_slots, set(range(4, 16)))
        self.assertTrue(subject.runtime.arena_last_use.synchronized)

    def test_hits_and_all_masked_never_allocate_or_submit_cpu(self):
        empty_like = torch.empty_like

        def no_host_buffers(*args, **kwargs):
            if kwargs.get("device") == "cpu" or kwargs.get("pin_memory"):
                raise AssertionError("host allocation on hit")
            return empty_like(*args, **kwargs)

        with patch.object(torch, "empty_like", side_effect=no_host_buffers):
            self.run_routes([[1, 0, 1, -1]])
            self.run_routes([[-1, -1, -1, -1]])
        self.assertEqual(
            self.decisions,
            [
                (1, [[1, 0, 1, -1]], self.mapping, frozenset()),
                (1, [[-1, -1, -1, -1]], self.mapping, frozenset()),
            ],
        )
        self.assertFalse(self.preparations)
        self.assertEqual(self.gpu_calls[0][1].tolist(), [[5, 3, 5, -1]])
        self.assertEqual(len(self.gpu_calls), 1)
        self.assertFalse(self.submitted)
        self.assertFalse(self.subject.host_buffers)
        self.assertEqual(self.subject.pending[2], ())

    def test_invalid_routes_poison_acquired_compute_window(self):
        for expert in (-2, self.subject.runtime.manifest.num_experts):
            with self.subTest(expert=expert):
                self.subject.failure = None
                state = self.subject.policy.layers[0]
                if state.phase == "computing":
                    self.subject.policy.release(0, state.epoch)
                    self.subject.policy.commit(0, state.epoch, [])
                epoch, mapping = self.subject.policy.begin(0)
                self.subject.current = (0, epoch, mapping)
                ids = torch.tensor([[expert]], dtype=torch.int32)
                with self.assertRaisesRegex(
                    ValueError, "outside expert range"
                ) as error:
                    self.subject.run(0, torch.ones((1, 2)), ids, torch.ones((1, 1)))
                self.assertIs(self.subject.failure, error.exception)
                with self.assertRaisesRegex(
                    RuntimeError, "background failure"
                ) as poisoned:
                    self.subject._check()
                self.assertIs(poisoned.exception.__cause__, error.exception)
                self.assertFalse(self.submitted)
                self.assertFalse(self.decisions)
                self.assertFalse(self.gpu_calls)
                self.assertIsNone(self.subject.pending)

    def test_gpu_misses_use_disjoint_arena_and_reader_fences(self):
        prior = self.Event()
        self.subject.runtime.arena_last_use = prior
        keys = list(self.subject.policy.keys)
        pool = self.pool["value"].clone()
        self.run_routes([[1, 4, 4, -1], [0, 2, 4, -1]])
        self.assertEqual(
            self.decisions,
            [(2, [[1, 4, 4, -1], [0, 2, 4, -1]], self.mapping, frozenset({2, 4}))],
        )
        self.assertEqual(
            [(arena, ids.tolist()) for arena, ids in self.gpu_calls],
            [
                (False, [[5, -1, -1, -1], [3, -1, -1, -1]]),
                (True, [[-1, 4, 4, -1], [-1, 2, 4, -1]]),
            ],
        )
        order = self.operations
        fill_index = next(i for i, op in enumerate(order) if op[0] == "fill")
        self.assertLess(order.index(("wait_event", "transfer", prior)), fill_index)
        self.assertLess(order.index(("wait_stream", "transfer", "compute")), fill_index)
        self.assertLess(fill_index, order.index(("gpu", False)))
        self.assertLess(
            order.index(("gpu", False)),
            order.index(("wait_stream", "compute", "transfer")),
        )
        self.assertLess(
            order.index(("wait_stream", "compute", "transfer")),
            order.index(("gpu", True)),
        )
        self.assertEqual(self.subject.runtime.arena_last_use.stream, "compute")
        self.assertEqual(self.subject.policy.keys, keys)
        torch.testing.assert_close(self.pool["value"], pool, equal_nan=True)
        self.assertFalse(self.subject.host_buffers)
        self.assertFalse(self.submitted)
        self.assertEqual(self.subject.stats["gpu_miss_experts"], 2)
        self.assertTrue(next(op[1] for op in order if op[0] == "invocation"))

    def test_gpu_no_hits_passes_only_arena_routes(self):
        # No resident mapping, so partitioning must not launch a cache pass.
        self.subject.policy = self.hybrid.GlobalLfu(2, 12, [8, 8])
        self.run_routes([[4, 4, 2, -1]])
        self.assertEqual(len(self.gpu_calls), 1)
        self.assertTrue(self.gpu_calls[0][0])
        self.assertFalse(self.subject.lookups)
        self.assertEqual(self.subject.stats["gpu_miss_experts"], 2)
        self.assertFalse(self.submitted)

    def test_ten_misses_all_admitted_use_one_cache_pass_without_arena(self):
        """Admission replaces the original miss snapshot before GPU routing."""
        self.admitted_experts = set(range(2, 12))
        # Even a CPU preference cannot select experts admitted into the cache.
        self.cpu_selection = self.admitted_experts.copy()
        self.run_routes([list(range(2, 12))])
        mapping = self.subject.policy.layers[0].mapping
        self.assertEqual(
            self.decisions, [(1, [list(range(2, 12))], mapping, frozenset())]
        )
        self.assertEqual(len(self.preparations), 1)
        self.assertEqual(len(self.cache_copies), 1)
        self.assertEqual({e for e, _ in self.cache_copies[0][1]}, set(range(2, 12)))
        self.assertEqual(
            [(arena, ids.tolist()) for arena, ids in self.gpu_calls],
            [(False, [[mapping[e] for e in range(2, 12)]])],
        )
        self.assertFalse(self.submitted)
        self.assertFalse(self.subject.host_buffers)
        self.assertFalse(any(op[0] == "fill" for op in self.operations))
        self.assertEqual(self.subject.stats["gpu_admitted_experts"], 10)
        self.assertEqual(self.subject.stats["selected_cpu_experts"], 0)
        self.assertEqual(self.subject.stats["gpu_miss_experts"], 0)
        self.assertFalse(next(op[1] for op in self.operations if op[0] == "invocation"))
        self.assertLess(
            next(i for i, op in enumerate(self.operations) if op[0] == "cache_copy"),
            next(i for i, op in enumerate(self.operations) if op[0] == "select_cpu"),
        )
        self.subject.release_previous()
        self.assertEqual(len(self.cache_copies), 1)
        self.assertTrue(self.subject.jobs.empty())
        self.assertFalse(self.subject.prepared_windows)
        self.assertEqual(self.subject.policy.layers[0].phase, "locked")

    def test_ten_misses_five_admitted_one_cpu_leave_four_arena_experts(self):
        """A CPU subset must preserve admitted and remaining rejected GPU work."""
        self.admitted_experts = set(range(2, 7))
        self.cpu_selection = {9}
        self.start_worker()
        self.run_routes([list(range(2, 12))])
        mapping = self.subject.policy.layers[0].mapping
        self.assertEqual(
            self.decisions,
            [(1, [list(range(2, 12))], mapping, frozenset(range(7, 12)))],
        )
        self.assertEqual(len(self.cpu_calls), 1)
        self.assertEqual(self.cpu_calls[0][0].tolist(), [[-1] * 7 + [9, -1, -1]])
        self.assertEqual(
            [(arena, ids.tolist()) for arena, ids in self.gpu_calls],
            [
                (False, [[mapping[e] for e in range(2, 7)] + [-1] * 5]),
                (True, [[-1] * 5 + [7, 8, -1, 10, 11]]),
            ],
        )
        self.assertEqual(sum(int((ids >= 0).sum()) for _, ids in self.gpu_calls), 9)
        self.assertEqual(
            [op for op in self.operations if op[0] == "fill"],
            [("fill", 0, (7, 8, 10, 11))],
        )
        self.assertEqual(len(self.cache_copies), 1)
        self.assertEqual({e for e, _ in self.cache_copies[0][1]}, set(range(2, 7)))
        self.assertEqual(self.subject.stats["gpu_admitted_experts"], 5)
        self.assertEqual(self.subject.stats["lfu_rejected_experts"], 5)
        self.assertEqual(self.subject.stats["selected_cpu_experts"], 1)
        self.assertEqual(self.subject.stats["gpu_miss_experts"], 4)
        self.subject.release_previous()
        self.assertEqual(len(self.cache_copies), 1)
        self.assertTrue(self.subject.jobs.empty())
        self.assertFalse(self.subject.prepared_windows)
        self.assertEqual(self.subject.policy.layers[0].phase, "locked")

    def test_real_prepare_publishes_first_touch_only_after_copy_completion(self):
        """First-touch free-slot admission cannot publish weights still copying."""
        subject = self.subject
        subject.policy = self.hybrid.GlobalLfu(2, 12, [2, 1])
        subject.policy.seed(1, {11: 0})
        reader_done = self.Event()
        reader_done.synchronized = True
        subject.runtime.layer_computed[0] = reader_done
        copy_started = threading.Event()
        copy_done = threading.Event()
        copy = subject._copy

        def pending_copy(layer, fills, done):
            self.assertIs(done, reader_done)
            copy(layer, fills, done)
            copy_started.set()
            return SimpleNamespace(query=copy_done.is_set)

        subject._copy = pending_copy
        self.start_manager()
        epoch, snapshot = subject.policy.begin(0)
        result = Future()

        def prepare():
            try:
                result.set_result(
                    self.hybrid.HybridCache._prepare_window(
                        subject, 0, epoch, (2,), self.compute
                    )
                )
            except BaseException as error:
                result.set_exception(error)

        caller = threading.Thread(target=prepare, daemon=True)
        caller.start()

        def finish_caller():
            copy_done.set()
            caller.join(3)
            self.assertFalse(caller.is_alive())

        self.addCleanup(finish_caller)
        self.assertTrue(copy_started.wait(2))
        self.assertEqual(subject.policy.layers[0].mapping, {})
        self.assertEqual(subject.readable[0], {})
        self.assertEqual(subject.policy.layers[0].phase, "managing")
        self.assertFalse(result.done())
        self.assertFalse(subject.active[0].done())
        self.assertEqual(subject.policy.keys[0], (0, 2))
        self.assertEqual(subject.policy.layers[1].mapping, {11: 2})
        copy_done.set()
        prepared_epoch, mapping = result.result(timeout=2)
        subject._join_queue(subject.jobs)
        self.assertEqual(prepared_epoch, epoch + 1)
        self.assertEqual(mapping, {2: 0})
        self.assertEqual(snapshot, {})
        self.assertEqual(subject.policy.layers[0].phase, "computing")
        self.assertFalse(subject.active)
        self.assertEqual(subject.prepared_windows, {(0, prepared_epoch)})
        self.assertEqual(self.cache_copies, [(0, ((2, 0),))])
        self.assertEqual(subject.policy.frequency[0][2], 1.0)
        self.assertLess(subject.policy.frequency[0][2], subject.policy.admission)
        torch.testing.assert_close(self.pool["value"][0], self.expert_values[2])
        subject.pending = (0, prepared_epoch, (2,), reader_done)
        subject.release_previous()
        self.assertEqual(self.cache_copies, [(0, ((2, 0),))])
        self.assertEqual(subject.policy.frequency[0][2], 1.0)
        self.assertTrue(subject.jobs.empty())
        self.assertFalse(subject.prepared_windows)
        self.assertEqual(subject.policy.layers[0].phase, "locked")

    def test_real_prepare_rejects_cold_replacement_then_admits_second_touch(self):
        """Only the released layer's resident can be replaced after probation."""
        subject = self.subject
        subject.policy = self.hybrid.GlobalLfu(2, 12, [1, 1])
        subject.policy.seed(0, {0: 0})
        subject.policy.seed(1, {11: 0})
        subject.policy.frequency[0][0] = 1.0
        done = self.Event()
        done.synchronized = True
        subject.runtime.layer_computed[0] = done
        self.start_manager()
        for attempt in range(2):
            epoch, snapshot = subject.policy.begin(0)
            prepared_epoch, mapping = self.hybrid.HybridCache._prepare_window(
                subject, 0, epoch, (2,), self.compute
            )
            subject._join_queue(subject.jobs)
            self.assertEqual(snapshot, {0: 0})
            self.assertEqual(mapping, {0: 0} if attempt == 0 else {2: 0})
            self.assertEqual(subject.policy.layers[1].mapping, {11: 1})
            self.assertEqual(len(self.cache_copies), attempt)
            self.assertAlmostEqual(subject.policy.frequency[0][2], 1 + 0.95 * attempt)
            frequency = [row[:] for row in subject.policy.frequency]
            subject.pending = (0, prepared_epoch, (2,), done)
            subject.release_previous()
            self.assertEqual(subject.policy.frequency, frequency)
            self.assertEqual(len(self.cache_copies), attempt)
            self.assertFalse(subject.prepared_windows)
            self.assertTrue(subject.jobs.empty())
        self.assertEqual(self.cache_copies, [(0, ((2, 0),))])
        self.assertEqual(subject.stats["evictions"], 1)
        self.assertEqual(subject.stats["admission_prepared_calls"], 2)

    def test_gpu_resident_and_arena_sum_before_bf16_conversion(self):
        self.pool["value"][3] = self.expert_values[0] = 256.0
        self.pool["value"][5] = self.expert_values[1] = 1.0
        self.expert_values[2] = -256.0
        output = self.run_routes([[0, 1, 2, 2, -1]], [[1, 1, 0.5, 0.5, 7]])
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
        self.assertEqual([arena for arena, _ in self.gpu_calls], [False, True])
        self.assertFalse(self.submitted)

    def test_cpu_duplicate_misses_fp32_merge_and_buffer_reuse_fences(self):
        self.cpu_selection = {2}
        self.start_worker()
        # A BF16 partition merge would lose the unit contribution before cancellation.
        self.pool["value"][3] = self.expert_values[0] = 256.0
        self.pool["value"][5] = self.expert_values[1] = 1.0
        self.expert_values[2] = -256.0
        self.run_routes([[0, 1, 2, 2, -1]], [[1, 1, 0.5, 0.5, 7]])
        pointers = tuple(
            t.data_ptr() for t in next(iter(self.subject.host_buffers.values()))
        )
        first_upload = next(
            op[2] for op in self.operations if op[:2] == ("record", "result")
        )
        boundary = len(self.operations)
        self.run_routes([[0, 1, 2, 2, -1]], [[1, 1, 0.25, 0.75, 9]])
        self.assertEqual(
            tuple(t.data_ptr() for t in next(iter(self.subject.host_buffers.values()))),
            pointers,
        )
        self.assertEqual(len(self.subject.host_buffers), 1)
        self.assertEqual(
            [job[2].tolist() for job in self.submitted], [[[-1, -1, 2, 2, -1]]] * 2
        )
        self.assertTrue(all(len(job) == 7 and job[-1].done() for job in self.submitted))
        self.assertTrue(
            all(job[3].dtype == job[5].dtype == torch.float32 for job in self.submitted)
        )
        self.assertEqual(
            self.decisions,
            [(1, [[0, 1, 2, 2, -1]], self.mapping, frozenset({2}))] * 2,
        )
        self.assertEqual(self.subject.stats["cpu_miss_experts"], 2)
        self.assertEqual(self.subject.stats["cpu_calls"], 2)
        self.assertIn(
            ("wait_event", "compute", first_upload), self.operations[:boundary]
        )
        second = self.operations[boundary:]
        readback = second.index(("wait_stream", "readback", "compute"))
        ready = self.submitted[1][4]
        self.assertLess(readback, second.index(("record", "readback", ready)))
        self.assertLess(second.index(("sync", ready)), second.index(("cpu",)))
        upload = next(op[2] for op in second if op[:2] == ("record", "result"))
        self.assertLess(
            second.index(("cpu",)), second.index(("record", "result", upload))
        )
        self.assertLess(
            second.index(("wait_event", "compute", upload)), second.index(("finish", 0))
        )

    def test_cpu_all_misses_and_all_masked_skips_cpu(self):
        self.cpu_selection = {2}
        self.admitted_experts = set()
        self.start_worker()
        self.run_routes([[2, 2, -1]])
        self.run_routes([[-1, -1, -1]])
        self.assertFalse(self.gpu_calls)
        self.assertEqual(len(self.cpu_calls), 1)
        self.assertEqual(self.cpu_calls[0][0].tolist(), [[2, 2, -1]])
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(self.decisions[-1][-1], frozenset())
        self.assertEqual(len(self.preparations), 1)
        self.assertEqual(self.subject.stats["cpu_miss_experts"], 1)
        self.assertEqual(len(self.subject.host_buffers), 1)

    def test_cpu_future_failure_propagates_without_upload_or_finish(self):
        self.cpu_selection = {2}
        self.cpu_error = RuntimeError("native CPU failure")
        self.start_worker()
        with self.assertRaisesRegex(RuntimeError, "native CPU failure"):
            self.run_routes([[0, 2, -1]])
        self.assertIs(self.subject.failure, self.cpu_error)
        self.assertIs(self.submitted[0][-1].exception(), self.cpu_error)
        self.assertEqual(self.subject.stats["cpu_miss_experts"], 0)
        self.assertFalse(
            any(
                op[0] == "finish" or op[:2] == ("record", "result")
                for op in self.operations
            )
        )
        self.assertIsNone(self.subject.pending)

    def test_arena_launch_failure_still_records_last_reader(self):
        self.arena_error = RuntimeError("arena launch failed after enqueue")
        with self.assertRaisesRegex(RuntimeError, "after enqueue"):
            self.run_routes([[0, 2]])
        done = self.subject.runtime.arena_last_use
        self.assertIsNotNone(done)
        self.assertEqual(done.stream, "compute")
        self.assertGreater(
            self.operations.index(("record", "compute", done)),
            self.operations.index(("gpu", True)),
        )
        self.assertIsNone(self.subject.pending)
        self.arena_error = None
        boundary = len(self.operations)
        self.run_routes([[2, -1]])
        later = self.operations[boundary:]
        self.assertLess(
            later.index(("wait_event", "transfer", done)),
            next(i for i, op in enumerate(later) if op[0] == "fill"),
        )

    def test_lookup_lru_is_per_layer_and_snapshot_immutable(self):
        subject = self.subject
        other = subject._lookup(1, {4: 7})
        a = subject._lookup(0, {0: 3})
        b = subject._lookup(0, {1: 5})
        self.assertIs(subject._lookup(0, {0: 3}), a)
        subject._lookup(0, {2: 6})
        self.assertEqual(list(subject.lookups[0]), [((0, 3),), ((2, 6),)])
        self.assertIs(subject._lookup(1, {4: 7}), other)
        self.assertIsNot(subject._lookup(0, {1: 5}), b)
        torch.testing.assert_close(a, torch.tensor([3] + [-1] * 11, dtype=torch.int32))
        self.assertEqual(len(subject.lookups[0]), 2)
        self.assertEqual(len(subject.lookups[1]), 1)

    def test_seed_zeros_padding_without_mutating_legacy_arena(self):
        subject = self.subject
        subject.seeded = False
        subject.policy = self.hybrid.GlobalLfu(2, 12, [8, 8])
        r = subject.runtime
        r.lock = threading.Lock()
        r.layer_cache_slots = [8, 8]
        r.device_states = {
            layer: {
                "keys": torch.tensor([0, 1] + [-1] * 6),
                "frequency": torch.ones(8),
            }
            for layer in (0, 1)
        }
        r._raise_error = lambda: None
        r._drain_arena_readers = lambda: self.operations.append(("drain",))
        r._return_arena = lambda: self.operations.append(("return",))
        self.pool["value"][0] = float("nan")
        self.arena["value"].fill_(float("nan"))
        subject._seed()
        self.assertEqual(self.pool["value"][0].item(), 0)
        self.assertTrue(torch.isnan(self.arena["value"]).all())
        self.assertTrue(torch.isnan(self.arena["value"][1:]).all())
        self.assertEqual(subject.policy.keys[0], None)
        self.assertNotIn(0, subject.policy.layers[0].mapping)
        self.assertTrue(subject.seeded)
        self.assertIn(("stream_sync", "compute"), self.operations)
        self.assertLess(
            self.operations.index(("return",)),
            self.operations.index(("stream_sync", "compute")),
        )


if __name__ == "__main__":
    unittest.main()
