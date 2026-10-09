"""Worker lifetime/pipelining tests without CUDA or model weights."""

import contextlib
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class TestBackgroundTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import against lightweight stand-ins so the worker protocol can run
        # on hosts without PyTorch/SGLang runtime dependencies.
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
            no_grad=lambda: contextlib.nullcontext(),
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "sglang.kernels.ops.moe.qwen38_expert_cache": SimpleNamespace(
                    cache_plan=None, cache_route=None, cache_ready_guard=None
                ),
                "sglang.srt.layers.moe.qwen38_cache_device": SimpleNamespace(
                    Qwen38DeviceExpertRuntime=object
                ),
                "sglang.srt.layers.moe.qwen38_cache_runtime": SimpleNamespace(
                    PreparedExperts=None
                ),
                "sglang.srt.layers.moe.qwen38_stream_memops": SimpleNamespace(
                    StreamMemoryOps=None
                ),
            },
        ):
            cls.module = load_source(
                "qwen38_threaded_worker_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            )

    def test_transport_advances_before_previous_compute_completes(self):
        runtime = self.module.Qwen38ThreadedExpertRuntime.__new__(
            self.module.Qwen38ThreadedExpertRuntime
        )
        runtime.device = "fake"
        runtime.jobs = queue.Queue()
        runtime.retired = queue.Queue()
        runtime.capacity = threading.BoundedSemaphore(2)
        runtime.worker_error = None
        runtime.arena_layer = 0
        runtime.arena_last_use = None
        runtime.transfer = SimpleNamespace(
            wait_event=lambda _: None, synchronize=lambda: None
        )
        signaled = threading.Event()
        signals = []

        def signal(stream, gate, failed=False):
            signals.append((gate, failed))
            if len(signals) == 2:
                signaled.set()

        runtime.memops = SimpleNamespace(signal=signal)
        completion = threading.Event()
        event = SimpleNamespace(synchronize=lambda: completion.wait(5))
        planned = SimpleNamespace(synchronize=lambda: None)
        zero = SimpleNamespace(item=lambda: 0)
        launched = threading.Event()
        launched.set()
        worker = threading.Thread(target=runtime._work)
        reaper = threading.Thread(target=runtime._retire)
        reaper.start()
        worker.start()
        try:
            for index in range(2):
                runtime.capacity.acquire()
                runtime.jobs.put(
                    self.module.TransferJob(
                        0, True, index, zero, planned, event, launched
                    )
                )
            self.assertTrue(
                signaled.wait(3), "worker blocked on previous compute retirement"
            )
            self.assertEqual(signals, [(0, False), (1, False)])
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

    def test_transfer_failure_signals_guard_and_retires(self):
        runtime = self.module.Qwen38ThreadedExpertRuntime.__new__(
            self.module.Qwen38ThreadedExpertRuntime
        )
        runtime.device = "fake"
        runtime.jobs = queue.Queue()
        runtime.retired = queue.Queue()
        runtime.capacity = threading.BoundedSemaphore(1)
        runtime.worker_error = None
        runtime.transfer = SimpleNamespace(
            wait_event=lambda _: None, synchronize=lambda: None
        )
        signals = []
        runtime.memops = SimpleNamespace(
            signal=lambda stream, gate, failed=False: signals.append(failed)
        )
        planned = SimpleNamespace(synchronize=lambda: None)
        event = SimpleNamespace(synchronize=lambda: None)
        launched = threading.Event()
        launched.set()
        invalid = SimpleNamespace(item=lambda: 1)
        worker = threading.Thread(target=runtime._work)
        reaper = threading.Thread(target=runtime._retire)
        reaper.start()
        worker.start()
        runtime.capacity.acquire()
        runtime.jobs.put(
            self.module.TransferJob(0, True, 0, invalid, planned, event, launched)
        )
        runtime.jobs.join()
        runtime.jobs.put(None)
        worker.join(3)
        runtime.retired.put(None)
        reaper.join(3)
        self.assertEqual(signals, [True])
        self.assertIsInstance(runtime.worker_error, ValueError)
