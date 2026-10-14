import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class TestCacheBuffers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch unavailable")
        cls.torch = torch
        cls.buffers = load_source(
            "qwen38_cache_buffers_test",
            "python/sglang/srt/layers/moe/qwen38_cache_buffers.py",
        )

    def test_metadata_reset_and_restore_reuse_storage(self):
        torch = self.torch
        buffers = self.buffers.AdaptiveMetadataBuffers(2, 16, 10, torch.device("cpu"))
        pointers = [
            {name: t.data_ptr() for name, t in layer.items()}
            for layer in buffers.device
        ]
        with (
            mock.patch.object(
                torch, "empty", side_effect=AssertionError("new allocation")
            ),
            mock.patch.object(
                torch, "tensor", side_effect=AssertionError("new tensor")
            ),
        ):
            for capacity in (8, 9, 4, 10):
                state = buffers.reset(0, capacity)
                buffers.restore(0, state, [(7, 3.0, 9), (2, 5.0, 12)])
                self.assertEqual(state["lookup"][7].item(), 0)
                self.assertEqual(state["keys"][:2].tolist(), [7, 2])
                self.assertEqual(state["frequency"][:2].tolist(), [3.0, 5.0])
                self.assertEqual(state["ages"][:2].tolist(), [9, 12])
                self.assertEqual(state["clock"].item(), 12)
                self.assertEqual(state["plan"].numel(), 4 + 4 * capacity)
                self.assertEqual(state["prefill_plan"].numel(), 17 + capacity)
                for name, t in state.items():
                    self.assertEqual(t.data_ptr(), pointers[0][name])
            state = buffers.reset(0, 8)
            buffers.restore(0, state, [])
            self.assertTrue((state["keys"] == -1).all())
            self.assertTrue((state["lookup"] == -1).all())
            self.assertEqual(state["clock"].item(), 0)

    def test_swiglu_grows_only_at_high_water_mark(self):
        torch = self.torch
        calls = []

        def materialize(config, capacity, device):
            calls.append(capacity)
            return tuple(
                torch.full((capacity,), v, device=device) for v in (1.0, 0.0, 7.0)
            )

        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass": SimpleNamespace(
                    materialize_swiglu_params_for_cutlass=materialize
                ),
            },
        ):
            buffers = self.buffers.GrowableSwigluBuffers(None, torch.device("cpu"), 512)
            first = buffers.get(105)
            self.assertEqual(calls, [105])
            smaller = buffers.get(80)
            self.assertEqual(smaller[0].data_ptr(), first[0].data_ptr())
            grown = buffers.get(106)
            self.assertEqual(calls, [105, 210])
            for capacity in (107, 128, 80, 210):
                view = buffers.get(capacity)
                self.assertEqual(view[0].data_ptr(), grown[0].data_ptr())
                self.assertEqual(view[0].numel(), capacity)
                self.assertTrue((view[2] == 7).all())
            buffers.get(512)
            self.assertEqual(calls, [105, 210, 512])

    def test_swiglu_without_clamp_allocates_no_tensor(self):
        materialize = mock.Mock(return_value=(None, None, None))
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass": SimpleNamespace(
                    materialize_swiglu_params_for_cutlass=materialize
                ),
            },
        ):
            buffers = self.buffers.GrowableSwigluBuffers(
                None, self.torch.device("cpu"), 512
            )
            self.assertEqual(buffers.get(80), (None, None, None))
            self.assertEqual(buffers.get(40), (None, None, None))
            materialize.assert_called_once()
