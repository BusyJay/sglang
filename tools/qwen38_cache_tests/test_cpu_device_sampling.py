"""Host-side ownership and ordered handoff of asynchronous GPU route samples."""

import contextlib
import math
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class Tensor:
    def __init__(self, shape):
        self.nbytes = math.prod(shape) * 4
        self.rows = {}
        self.copies = 0

    def __getitem__(self, key):
        bank, layer, index = key
        indices = (
            range(index.start, index.stop) if isinstance(index, slice) else [index]
        )
        tensor = self

        class View:
            def data_ptr(self):
                return 1 + bank * 1000 + layer * 100 + index * 10

            def copy_(self, source, non_blocking):
                assert non_blocking
                tensor.copies += 1
                for i, row in zip(indices, source.tolist()):
                    tensor.rows[bank, layer, i] = list(row)

            def tolist(self):
                return [list(tensor.rows[bank, layer, i]) for i in indices]

        return View()


class Event:
    def __init__(self):
        self.allow = threading.Event()
        self.allow.set()
        self.waiting = threading.Event()

    def record(self, stream):
        pass

    def synchronize(self):
        self.waiting.set()
        if not self.allow.wait(5):
            raise TimeoutError("test collector did not get released")


class TestDeviceRouteSampler(unittest.TestCase):
    def setUp(self):
        self.stream = mock.Mock()
        fake_torch = SimpleNamespace(
            int32="int32",
            empty=lambda shape, **kw: Tensor(shape),
            cuda=SimpleNamespace(
                Stream=lambda **kw: self.stream,
                Event=Event,
                current_stream=lambda device: self.stream,
                device=lambda device: contextlib.nullcontext(),
                stream=lambda stream: contextlib.nullcontext(),
            ),
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            module = load_source(
                "device_sampling_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_sampling.py",
            )
        self.runtime = SimpleNamespace(
            manifest=SimpleNamespace(num_layers=2, num_experts=8, top_k=2),
            adaptive_cache_window=2,
            device="fake",
            adaptive_cache=True,
            adaptive_samples=queue.Queue(),
            adaptive_sample_drops=0,
        )
        self.abort = mock.Mock()
        self.sampler = module.DeviceRouteSampler(self.runtime, self.abort)
        self.addCleanup(self.sampler.close)

    def record(self, layer, cycle, index, active):
        pointer = self.sampler.begin(layer, (cycle, index), self.stream)
        if pointer:
            bank = self.sampler.current[layer][0]
            self.sampler.device.rows[bank, layer, index] = [len(active), *active, -999][
                :3
            ]
            self.sampler.commit(layer, (cycle, index), self.stream)
        return pointer

    def test_window_and_partial_flush_preserve_counts_ids_and_indices(self):
        self.record(0, 0, 0, [1, 5])
        self.assertEqual(self.sampler.host.copies, 0)
        self.record(0, 0, 1, [])
        self.sampler.synchronize()
        self.assertEqual(
            self.runtime.adaptive_samples.get_nowait(), (0, 0, 0, 2, [[1, 5], []])
        )
        self.assertEqual(self.sampler.host.copies, 1)
        # A transport switch can continue the same observation at a later index.
        self.record(1, 3, 1, [7])
        self.sampler.synchronize()
        self.assertEqual(
            self.runtime.adaptive_samples.get_nowait(), (1, 3, 1, 2, [[7]])
        )
        self.abort.assert_not_called()
        self.assertEqual(self.sampler.device_bytes, 2 * 2 * 2 * 3 * 4)
        self.assertEqual(self.sampler.pinned_bytes, self.sampler.device_bytes)

    def test_banks_are_not_reused_while_collector_waits_for_cuda(self):
        event = self.sampler.copied[0][0]
        event.allow.clear()
        self.addCleanup(event.allow.set)
        self.record(0, 0, 0, [1])
        self.record(0, 0, 1, [2])
        self.assertTrue(event.waiting.wait(2))
        self.assertFalse(self.sampler.free[0][0].is_set())
        self.record(0, 1, 0, [3])
        self.record(0, 1, 1, [4])
        self.assertFalse(self.sampler.free[1][0].is_set())
        self.assertEqual(self.record(0, 2, 0, [5]), 0)
        self.assertEqual(self.record(0, 2, 1, [6]), 0)
        self.assertEqual(self.runtime.adaptive_sample_drops, 2)
        event.allow.set()
        self.sampler.synchronize()
        self.assertFalse(self.sampler.free[0][0].is_set())
        consumed = [self.runtime.adaptive_samples.get_nowait() for _ in range(2)]
        for batch in consumed:
            batch.release()
        self.assertNotEqual(self.record(0, 3, 0, [7]), 0)
        self.sampler.synchronize()
        cycles = [batch[1] for batch in consumed]
        cycles.append(self.runtime.adaptive_samples.get_nowait()[1])
        self.assertEqual(cycles, [0, 1, 3])

    def test_queue_pressure_reports_entire_batch_and_releases_bank(self):
        self.runtime.adaptive_samples = queue.Queue(maxsize=1)
        self.runtime.adaptive_samples.put("occupied")
        self.record(0, 0, 0, [1])
        self.record(0, 0, 1, [2])
        self.sampler.synchronize()
        self.assertEqual(self.runtime.adaptive_sample_drops, 2)
        self.assertTrue(self.sampler.free[0][0].is_set())
        self.abort.assert_not_called()

    def test_invalid_sample_is_rejected_before_analysis(self):
        self.record(0, 0, 0, [8])
        self.sampler.synchronize()
        self.assertTrue(self.runtime.adaptive_samples.empty())
        self.assertIsInstance(self.abort.call_args.args[0], ValueError)
