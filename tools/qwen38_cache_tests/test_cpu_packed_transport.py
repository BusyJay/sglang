# SPDX-License-Identifier: Apache-2.0
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

m = load_source("packed_transport_cpu", "python/sglang/srt/layers/moe/qwen38_packed_transport.py")


class TestPackedLayout(unittest.TestCase):
    def test_tail_segments_reconstruct_payload_without_gaps(self):
        sizes = [13, 16, 69, 1]
        pages = m.page_layout(sizes, 32)
        actual = [bytearray(size) for size in sizes]
        reference = [bytes([i + 1]) * size for i, size in enumerate(sizes)]
        visited = [0] * len(sizes)
        for page in pages:
            packed = bytearray(32)
            end = 0
            for i, offset, start, count in page:
                self.assertEqual(start, end)
                packed[start:start + count] = reference[i][offset:offset + count]
                end += count
                visited[i] += count
            self.assertLessEqual(end, 32)
            for i, offset, start, count in page:
                actual[i][offset:offset + count] = packed[start:start + count]
        self.assertEqual(visited, sizes)
        self.assertEqual(actual, list(reference))

    def test_threshold_validation_and_direct_fallback(self):
        for args in [("bad", "direct"), ("direct", "packed", 0), ("packed", "direct", 8, 4)]:
            with self.assertRaises(ValueError):
                m.validate_packed_config(*args)
        t = m.PackedExpertTransport.__new__(m.PackedExpertTransport)
        t.runtime = SimpleNamespace(packed_min_bytes=8, packed_max_bytes=32, packed_min_copies=2)
        for sizes, expected in [([8, 8], True), ([32], False), ([1, 1], False), ([32, 32], False)]:
            t.layouts = {"decode": ((), sizes, m.page_layout(sizes, 32))}
            self.assertIs(t.eligible("decode"), expected)


class TestPackedLifecycle(unittest.TestCase):
    def fixture(self):
        import torch
        from collections import Counter
        t = m.PackedExpertTransport.__new__(m.PackedExpertTransport)
        t.torch = torch
        t.counters = Counter()
        t.scratch = mock.MagicMock()
        t.scratch.__getitem__.return_value = mock.Mock()
        actions = []
        event = mock.Mock()
        event.synchronize.side_effect = lambda: actions.append("host_wait")
        event.record.side_effect = lambda _: actions.append("completion")
        r = SimpleNamespace(ring=[torch.zeros(32, dtype=torch.uint8)], ring_events=[event], ring_index=0,
                            transfer=object(), transferred_bytes=0, h2d_copy_calls=0, staging_batches=0)
        views = []
        def source(layer, expert, name):
            v = memoryview(bytes([expert + 1]) * (17 if name == "a" else 9))
            views.append(v)
            return v
        r.pack = SimpleNamespace(expert_tensor_bytes=source)
        t.runtime = r
        pages = m.page_layout((17, 9), 32)
        t.layouts = {"decode": (("a", "b"), (17, 9), pages)}
        t.tables = {("decode", 0, 0): (object(),)}
        return t, r, views, actions

    def test_ring_completion_covers_scatter_and_reuse(self):
        import torch
        t, r, views, actions = self.fixture()
        with mock.patch("sglang.kernels.ops.moe.qwen38_expert_cache.packed_scatter", side_effect=lambda *_: actions.append("scatter")), \
             mock.patch.object(torch.cuda, "stream", lambda _: contextlib.nullcontext()):
            t.fill(0, 2, 0, "decode")
            t.fill(0, 3, 0, "decode")
        self.assertEqual(actions, ["host_wait", "scatter", "completion"] * 2)
        self.assertEqual(r.h2d_copy_calls, 2)
        self.assertEqual(r.transferred_bytes, 52)
        for view in views:
            with self.assertRaises(ValueError):
                len(view)

    def test_scatter_failure_protects_ring_and_releases_views(self):
        import torch
        t, r, views, actions = self.fixture()
        with mock.patch("sglang.kernels.ops.moe.qwen38_expert_cache.packed_scatter", side_effect=RuntimeError("launch")), \
             mock.patch.object(torch.cuda, "stream", lambda _: contextlib.nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "launch"):
                t.fill(0, 0, 0, "decode")
        self.assertEqual(actions, ["host_wait", "completion"])
        self.assertEqual(r.staging_batches, 1)
        for view in views:
            with self.assertRaises(ValueError):
                len(view)

    def test_address_tables_refresh_after_layout_relocation(self):
        import torch
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
        rows = {name: torch.zeros((2, 16), dtype=torch.uint8) for name in REQUIRED_TENSORS}
        records = {name: SimpleNamespace(nbytes=32) for name in REQUIRED_TENSORS}
        r = SimpleNamespace(ring=[torch.zeros(64, dtype=torch.uint8)], device="cpu", caches={0: rows},
                            arena=None, packed_decode="packed", packed_prefill="direct", decode_copy_names=tuple(REQUIRED_TENSORS), manifest=SimpleNamespace(tensors={0: records}, num_experts=2), transfer=mock.Mock())
        with mock.patch.object(torch.cuda, "current_stream", return_value=mock.Mock()):
            t = m.PackedExpertTransport(r)
            t.refresh_targets()
            first = int(t.tables["decode", 0, 1][0][0, 0])
            self.assertEqual(first, rows[REQUIRED_TENSORS[0]][1].data_ptr())
            r.caches = {0: {name: torch.zeros((3, 16), dtype=torch.uint8) for name in REQUIRED_TENSORS}}
            t.refresh_targets()
            self.assertEqual(int(t.tables["decode", 0, 2][0][0, 0]), r.caches[0][REQUIRED_TENSORS[0]][2].data_ptr())
            self.assertNotEqual(first, int(t.tables["decode", 0, 1][0][0, 0]))
            self.assertEqual(t.counters["target_refreshes"], 2)

    def test_runtime_direct_fallback_and_prefill_fragment_selection(self):
        import torch
        from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
        from sglang.srt.layers.moe.qwen38_cache_device import Qwen38DeviceExpertRuntime
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
        r = Qwen38ThreadedExpertRuntime.__new__(Qwen38ThreadedExpertRuntime)
        r.miss_mailbox = None
        r.caches, r.arena = {0: {}}, {}
        r.packed_transport = None
        with mock.patch.object(Qwen38DeviceExpertRuntime, "_fill") as direct:
            r._fill(0, 0, r.caches[0], 0)
            direct.assert_called_once()
        r.profiler = None
        names = tuple(REQUIRED_TENSORS)
        transport = mock.Mock(layouts={"decode": (names, (), ()), "prefill": (names, (), ())})
        r.packed_transport = transport
        r.packed_decode = r.packed_prefill = "packed"
        transport.eligible.return_value = False
        transport.counters = __import__("collections").Counter()
        with mock.patch.object(Qwen38DeviceExpertRuntime, "_fill") as direct:
            r._fill(0, 0, r.caches[0], 0)
            direct.assert_called_once()
            transport.fill.assert_not_called()
        transport.eligible.return_value = True
        with mock.patch.object(Qwen38DeviceExpertRuntime, "_fill") as direct:
            r._fill(0, 0, r.caches[0], 0)
            direct.assert_not_called()
            transport.fill.assert_called_once_with(0, 0, 0, "decode")
        transport.fill.reset_mock()
        r._fill_arena_selected(0, [0, 2])
        self.assertEqual(transport.fill.call_count, 2)
        # A contiguous run must take the original bulk path.
        r.manifest = SimpleNamespace(tensors={0: {name: SimpleNamespace(nbytes=64) for name in names}}, num_experts=4)
        r.arena = {name: torch.zeros((4, 16), dtype=torch.uint8) for name in names}
        r.pack = SimpleNamespace(tensor_bytes=lambda *_: memoryview(bytes(64)))
        r._copy_batch = mock.Mock(side_effect=lambda entries: [view.release() for view, _ in entries])
        transport.fill.reset_mock()
        r._fill_arena_selected(0, [0, 1])
        transport.fill.assert_not_called()
        r._copy_batch.assert_called_once()
