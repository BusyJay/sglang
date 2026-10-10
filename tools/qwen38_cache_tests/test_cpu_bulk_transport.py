# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual staging implementation without torch, CUDA or weights."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class ByteTensor:
    def __init__(self, data):
        self.data = memoryview(data)

    def reshape(self, *_):
        return self

    def view(self, *_):
        return self

    def numel(self):
        return len(self.data)

    def numpy(self):
        return self.data

    def __getitem__(self, key):
        return ByteTensor(self.data[key])

    def copy_(self, source, non_blocking=False):
        self.data[:] = source.data


class Event:
    def __init__(self, blocking=False):
        self.blocking = blocking
        self.records = self.waits = 0

    def record(self, stream):
        self.records += 1

    def synchronize(self):
        self.waits += 1


class FailingTensor(ByteTensor):
    def __getitem__(self, key):
        return self

    def copy_(self, source, non_blocking=False):
        raise RuntimeError("H2D failure")


class TestBulkTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict("sys.modules", {
            "torch": SimpleNamespace(
                uint8="uint8",
                cuda=SimpleNamespace(stream=lambda _: contextlib.nullcontext(), Event=Event),
            ),
            "sglang.kernels.ops.moe.qwen38_expert_cache": SimpleNamespace(cache_route=None),
            "sglang.srt.layers.moe.qwen38_cache_runtime": SimpleNamespace(
                Qwen38SynchronousExpertRuntime=object
            ),
            "sglang.srt.model_loader.qwen38_nvfp4_pack.format": SimpleNamespace(
                REQUIRED_TENSORS=("a", "b")
            ),
        }):
            cls.module = load_source(
                "qwen38_bulk_under_test", "python/sglang/srt/layers/moe/qwen38_cache_async.py"
            )

    def runtime(self, size=8):
        runtime = self.module.Qwen38AsyncExpertRuntime.__new__(
            self.module.Qwen38AsyncExpertRuntime
        )
        runtime.ring = [ByteTensor(bytearray(size)) for _ in range(2)]
        runtime.ring_events = [None, None]
        runtime.ring_index = 0
        runtime.transfer = object()
        runtime.ready = None
        runtime.transferred_bytes = runtime.h2d_copy_calls = runtime.staging_batches = 0
        runtime._order_transfer = lambda: None
        return runtime

    def test_contiguous_tensor_chunks_and_ring_reuse(self):
        runtime = self.runtime()
        source = memoryview(bytes(range(35)))
        output = bytearray(35)
        runtime._copy(source, ByteTensor(output))
        self.assertEqual(output, bytes(range(35)))
        self.assertEqual(runtime.h2d_copy_calls, 5)
        self.assertEqual(runtime.staging_batches, 5)
        self.assertEqual(runtime.transferred_bytes, 35)
        self.assertEqual(sum(e.waits for e in runtime.ring_events), 3)
        self.assertTrue(all(e.blocking for e in runtime.ring_events))
        self.assertNotIn(runtime.ready, runtime.ring_events)
        with self.assertRaises(ValueError):
            len(source)  # Consumed pack views must not keep mmap alive.

    def test_small_tensors_share_page_and_event(self):
        runtime = self.runtime()
        outputs = [bytearray(3), bytearray(4)]
        runtime._copy_batch([
            (memoryview(b"abc"), ByteTensor(outputs[0])),
            (memoryview(b"defg"), ByteTensor(outputs[1])),
        ])
        self.assertEqual(outputs, [b"abc", b"defg"])
        self.assertEqual(runtime.h2d_copy_calls, 2)
        self.assertEqual(runtime.staging_batches, 1)

    def test_ready_event_not_rerecorded_when_ring_wraps(self):
        runtime = self.runtime()
        runtime._copy(memoryview(b"abcdefgh"), ByteTensor(bytearray(8)))
        first_ready = runtime.ready
        for _ in range(3):
            runtime._copy(memoryview(b"ijklmnop"), ByteTensor(bytearray(8)))
        self.assertEqual(first_ready.records, 1)
        self.assertIsNot(first_ready, runtime.ready)
        self.assertEqual(sum(e.records for e in runtime.ring_events), 4)

    def test_mismatched_size_releases_view_without_dispatch(self):
        runtime = self.runtime()
        view = memoryview(b"abc")
        with self.assertRaisesRegex(ValueError, "byte sizes differ"):
            runtime._copy(view, ByteTensor(bytearray(2)))
        self.assertEqual(runtime.h2d_copy_calls, 0)
        with self.assertRaises(ValueError):
            len(view)

    def test_arena_uses_whole_tensor_views_not_expert_views(self):
        runtime = self.runtime()
        runtime.pack = SimpleNamespace(
            tensor_bytes=lambda layer, name: memoryview(b"a" * 17 if name == "a" else b"b" * 5)
        )
        runtime.arena = {"a": ByteTensor(bytearray(17)), "b": ByteTensor(bytearray(5))}
        runtime._fill_arena(0)
        self.assertEqual(bytes(runtime.arena["a"].data), b"a" * 17)
        self.assertEqual(bytes(runtime.arena["b"].data), b"b" * 5)
        self.assertEqual(runtime.staging_batches, 3)
        self.assertEqual(runtime.transferred_bytes, 22)

    def test_partial_dispatch_failure_records_completion_and_releases_views(self):
        runtime = self.runtime()
        first, second = memoryview(b"abc"), memoryview(b"def")
        bad = FailingTensor(bytearray(3))
        with self.assertRaisesRegex(RuntimeError, "H2D failure"):
            runtime._copy_batch([(first, ByteTensor(bytearray(3))), (second, bad)])
        self.assertEqual(runtime.ring_events[0].records, 1)
        self.assertIsNotNone(runtime.ready)
        for view in (first, second):
            with self.assertRaises(ValueError):
                len(view)
