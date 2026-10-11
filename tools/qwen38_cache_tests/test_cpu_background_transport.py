# SPDX-License-Identifier: Apache-2.0
"""Worker lifetime/pipelining tests without CUDA or model weights."""

import contextlib
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class FakeCudaEvent:
    """Stand-in for torch.cuda.Event; records which stream it landed on."""

    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream

    def synchronize(self):
        return None


class TestBackgroundTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import against lightweight stand-ins so the worker protocol can run
        # on hosts without PyTorch/SGLang runtime dependencies.
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                device=lambda _: contextlib.nullcontext(), Event=FakeCudaEvent
            ),
            no_grad=lambda: contextlib.nullcontext(),
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "sglang.kernels.ops.moe.qwen38_expert_cache": SimpleNamespace(
                    cache_plan=None, cache_route=None
                ),
                "sglang.srt.layers.moe.qwen38_cache_device": SimpleNamespace(
                    Qwen38DeviceExpertRuntime=object
                ),
                "sglang.srt.layers.moe.qwen38_expert_cache": SimpleNamespace(
                    LayerExpertCachePolicy=None
                ),
                "sglang.srt.model_loader.qwen38_nvfp4_pack.format": SimpleNamespace(
                    REQUIRED_TENSORS=()
                ),
                "sglang.srt.layers.moe.qwen38_cache_runtime": SimpleNamespace(
                    PreparedExperts=None
                ),
            },
        ):
            cls.module = load_source(
                "qwen38_threaded_worker_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            )

    def _runtime(self, permits):
        runtime = self.module.Qwen38ThreadedExpertRuntime.__new__(
            self.module.Qwen38ThreadedExpertRuntime
        )
        runtime.device = "fake"
        runtime.jobs = queue.Queue(maxsize=permits)
        runtime.retired = queue.Queue()
        runtime.capacity = threading.BoundedSemaphore(permits)
        runtime.worker_error = None
        runtime.poisoned = False
        runtime.warm_decode_cache = False
        runtime.cache_policy = "lfu"
        runtime.slots_for_layer = lambda _: 2
        runtime.arena_residents = None
        runtime.manifest = SimpleNamespace(num_experts=3)
        runtime.arena_layer = 0
        runtime.arena_last_use = None
        runtime.transfer = SimpleNamespace(
            wait_event=lambda _: None, synchronize=lambda: None
        )
        return runtime

    def _job(self, planned, completed):
        launched = threading.Event()
        launched.set()
        # Arena jobs carry the pinned overflow counter; 0 means no bad route.
        plan = SimpleNamespace(tolist=lambda: [0, 0, 0, 0])
        return self.module.TransferJob(0, True, plan, planned, completed, launched)

    def _threads(self, runtime):
        worker = threading.Thread(target=runtime._work)
        reaper = threading.Thread(target=runtime._retire)
        reaper.start()
        worker.start()
        return worker, reaper

    def test_transport_reports_ready_events_while_compute_pending(self):
        # Slot readiness must be reported by CUDA events recorded on the
        # transfer stream; it must not wait on prior compute completion.
        runtime = self._runtime(2)
        completion = threading.Event()
        planned = SimpleNamespace(synchronize=lambda: None)
        worker, reaper = self._threads(runtime)
        try:
            jobs = []
            for _ in range(2):
                runtime.capacity.acquire()
                completed = SimpleNamespace(
                    synchronize=lambda: completion.wait(5)
                )
                jobs.append(self._job(planned, completed))
                runtime.jobs.put(jobs[-1])
            for job in jobs:
                self.assertTrue(
                    job.ready_set.wait(3), "worker did not report slot readiness"
                )
                self.assertIsNone(job.error)
                self.assertIsInstance(job.ready, FakeCudaEvent)
                self.assertIs(job.ready.recorded_on, runtime.transfer)
            self.assertIs(runtime.arena_last_use, jobs[-1].computed)
        finally:
            completion.set()
            runtime.jobs.join()
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reaper.is_alive())
        self.assertIsNone(runtime.worker_error)

    def test_transfer_failure_propagates_to_consumer_and_retires(self):
        runtime = self._runtime(1)
        planned = SimpleNamespace(synchronize=lambda: None)
        completed = SimpleNamespace(synchronize=lambda: None)
        worker, reaper = self._threads(runtime)
        bad_plan = SimpleNamespace(tolist=lambda: [1, 0, 0, 0])
        runtime.capacity.acquire()
        job = self.module.TransferJob(
            0, False, bad_plan, planned, completed, threading.Event()
        )
        job.launched.set()
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3), "worker did not report failure")
            self.assertIsInstance(job.error, ValueError)
            self.assertIsNone(job.ready)
            self.assertIsNotNone(runtime.worker_error)
        finally:
            runtime.jobs.join()
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reaper.is_alive())

    def test_poisoned_runtime_rejects_new_work(self):
        runtime = self._runtime(1)
        runtime.poisoned = True
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            runtime._raise_error()

    def test_cross_layer_arena_waits_before_bulk_fill(self):
        runtime = self._runtime(1)
        previous = object()
        runtime.arena_last_use = previous
        runtime.arena_loads = 0
        operations = []
        runtime.transfer.wait_event = lambda event: operations.append(("wait", event))
        runtime._ensure_arena_experts = (
            lambda layer, counts: operations.append(("fill", layer)) or True
        )
        planned = SimpleNamespace(synchronize=lambda: None)
        computed = SimpleNamespace(synchronize=lambda: None)
        job = self._job(planned, computed)
        job.layer = 1
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsNone(job.error)
            self.assertEqual(operations, [("wait", planned), ("wait", previous), ("fill", 1)])
            self.assertEqual(runtime.arena_loads, 1)
            self.assertIs(runtime.arena_last_use, computed)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)

    def test_warmup_dispatch_precedes_ready_notice(self):
        runtime = self._runtime(1)
        runtime.warm_decode_cache = True
        warm_calls = []
        runtime._warm_from_arena = lambda layer, counts: warm_calls.append((layer, counts))
        job = self._job(SimpleNamespace(synchronize=lambda: None), SimpleNamespace(synchronize=lambda: None))
        job.plan = SimpleNamespace(tolist=lambda: [5, 2, 0, 0])
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsNone(job.error)
            self.assertEqual(warm_calls, [(0, [5, 2, 0])])
            self.assertIs(job.ready.recorded_on, runtime.transfer)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)

    def test_warmup_failure_propagates_without_ready_event(self):
        runtime = self._runtime(1)
        runtime.warm_decode_cache = True
        runtime._warm_from_arena = mock.Mock(side_effect=RuntimeError("D2D failed"))
        job = self._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: None),
        )
        job.plan = SimpleNamespace(tolist=lambda: [4, 2, 0])
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsInstance(job.error, RuntimeError)
            self.assertIsNone(job.ready)
            self.assertIsNotNone(runtime.worker_error)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
