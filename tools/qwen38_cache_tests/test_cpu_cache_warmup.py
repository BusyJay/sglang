# SPDX-License-Identifier: Apache-2.0
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

warmup = load_source(
    "sglang.srt.layers.moe.qwen38_cache_warmup",
    "python/sglang/srt/layers/moe/qwen38_cache_warmup.py",
)


class TestWarmup(unittest.TestCase):
    def test_frequency_ranking_and_deterministic_ties(self):
        self.assertEqual(warmup.select_warm_experts([3, 9, 9, 0, 5], 3), [1, 2, 4])

    def test_padding_only_and_small_active_set(self):
        self.assertEqual(warmup.select_warm_experts([0, 0], 2), [])
        self.assertEqual(warmup.select_warm_experts([0, 5, 0], 2), [1])

    def test_invalid_counts_and_capacity(self):
        for counts, capacity in [([1], 0), ([-1, 2], 2)]:
            with self.assertRaises(ValueError):
                warmup.select_warm_experts(counts, capacity)

    def test_actual_warmup_dispatch_and_policy_initialization(self):
        names = ("a", "b")
        indices = mock.MagicMock(name="indices")
        slots = mock.MagicMock(name="slots")
        ready = mock.Mock()
        cuda = SimpleNamespace(stream=lambda _: contextlib.nullcontext(), Event=lambda: ready)
        fake_torch = SimpleNamespace(
            cuda=cuda, int64="int64", int32="int32",
            tensor=mock.Mock(return_value=indices),
            arange=mock.Mock(return_value=slots), index_select=mock.Mock(),
        )
        with mock.patch.dict("sys.modules", {
            "torch": fake_torch,
            "sglang.kernels.ops.moe.qwen38_expert_cache": SimpleNamespace(cache_plan=None, cache_route=None),
            "sglang.srt.layers.moe.qwen38_cache_async": SimpleNamespace(Qwen38AsyncExpertRuntime=object),
            "sglang.srt.layers.moe.qwen38_cache_runtime": SimpleNamespace(PreparedExperts=None),
            "sglang.srt.model_loader.qwen38_nvfp4_pack.format": SimpleNamespace(REQUIRED_TENSORS=names),
        }):
            module = load_source("qwen38_warm_device_under_test", "python/sglang/srt/layers/moe/qwen38_cache_device.py")
        runtime = module.Qwen38DeviceExpertRuntime.__new__(module.Qwen38DeviceExpertRuntime)
        runtime.device = "cuda"
        runtime.slots_per_layer = 2
        runtime.slots_for_layer = lambda _: 2
        runtime.transfer = mock.Mock()
        old_reader = object()
        runtime.slot_last_use = {0: [old_reader, None]}
        state = {name: mock.MagicMock(name=name) for name in (
            "lookup", "keys", "frequency", "ages", "generation", "clock"
        )}
        runtime.device_states = {0: state}
        runtime.arena = {name: mock.MagicMock() for name in names}
        runtime.caches = {0: {name: mock.MagicMock() for name in names}}
        runtime.warm_ready = {}
        runtime.slot_ready = {0: [None, None]}
        runtime.warmup_runs = runtime.warmed_experts = runtime.warmup_d2d_bytes = 0
        runtime.manifest = SimpleNamespace(
            num_experts=4, tensors={0: {name: SimpleNamespace(nbytes=32) for name in names}}
        )
        runtime._warm_from_arena(0, [1, 7, 6, 0])
        fake_torch.tensor.assert_called_once_with([1, 2], dtype="int64", device="cuda")
        runtime.transfer.wait_event.assert_called_once_with(old_reader)
        self.assertEqual(fake_torch.index_select.call_count, 2)
        state["lookup"].fill_.assert_called_once_with(-1)
        state["frequency"].zero_.assert_called_once()
        state["frequency"].__getitem__.return_value.fill_.assert_called_once_with(1)
        ready.record.assert_called_once_with(runtime.transfer)
        self.assertEqual(runtime.slot_ready[0], [ready, ready])
        self.assertEqual(runtime.warmed_experts, 2)
        self.assertEqual(runtime.warmup_d2d_bytes, 32)
        runtime._warm_from_arena(0, [0, 0, 0, 0])
        self.assertEqual(runtime.warmup_runs, 1)
