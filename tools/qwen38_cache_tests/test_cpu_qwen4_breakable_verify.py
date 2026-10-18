"""Verify graphs must not enter helpers that require a prefill runner context."""
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from sglang.srt.models import qwen4_exp as model
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestBreakableVerify(unittest.TestCase):
    def test_verify_qsa_uses_batch_without_prefill_context(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        layer = SimpleNamespace(_compute_qsa_topk_indices_eager=mock.Mock(return_value='indices'))
        with mock.patch.object(model, 'is_in_breakable_cuda_graph', return_value=True), mock.patch.object(model, '_breakable_qsa_indexer') as prefill:
            result = model.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices(layer, 'hidden', 'positions', batch)
        self.assertEqual(result, 'indices')
        prefill.assert_not_called()
        layer._compute_qsa_topk_indices_eager.assert_called_once_with(hidden_states='hidden', positions='positions', forward_batch=batch)

    def test_verify_ple_uses_original_batch(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        ple_batch = object()
        layer = SimpleNamespace(_ple_forward_batch=batch, _ple_batch=ple_batch, ple=mock.Mock(return_value=torch.ones(2)))
        residual = torch.zeros(2)
        with mock.patch.object(model, 'is_in_breakable_cuda_graph', return_value=True), mock.patch.object(model, '_breakable_ple_forward') as prefill, mock.patch.object(model, '_shards_rows_over_cp', return_value=False):
            result = model.Qwen4ExpLayerExtensionMixin._add_ple_embedding(layer, residual)
        torch.testing.assert_close(result, torch.ones(2))
        prefill.assert_not_called()
        layer.ple.assert_called_once_with(residual, batch, ple_batch)

    def test_prefill_retains_eager_bridge(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with mock.patch.object(model, 'is_in_breakable_cuda_graph', return_value=True), mock.patch.object(model, '_breakable_qsa_indexer', return_value='prefill') as prefill:
            self.assertEqual(model.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices(object(), 'hidden', 'positions', batch), 'prefill')
        prefill.assert_called_once()

    def test_verify_prefetch_survives_eager_growth(self):
        layer = SimpleNamespace(_graph_prefetch_buffers={}, _eager_prefetch_buffer=None,
            _allocate_prefetch_buffer=lambda n, ids: torch.empty(n, 16))
        lookup = model.Qwen4ExpPLELayer._get_prefetch_buffer
        with mock.patch.object(model, 'is_in_breakable_cuda_graph', return_value=True), mock.patch.object(model, 'get_tc_piecewise_forward_context', return_value=None):
            with mock.patch.object(model, 'get_is_capture_mode', return_value=True):
                captured = lookup(layer, 3, None)
            with mock.patch.object(model, 'get_is_capture_mode', return_value=False):
                eager = lookup(layer, 8192, None)
            with mock.patch.object(model, 'get_is_capture_mode', return_value=True):
                replay = lookup(layer, 3, None)
        self.assertEqual(captured.data_ptr(), replay.data_ptr())
        self.assertNotEqual(captured.data_ptr(), eager.data_ptr())
        self.assertIs(layer._graph_prefetch_buffers[3], captured)

    def test_prefill_bridge_uses_eager_prefetch_storage(self):
        layer = SimpleNamespace(_graph_prefetch_buffers={}, _eager_prefetch_buffer=None,
            _allocate_prefetch_buffer=lambda n, ids: torch.empty(n, 16))
        with mock.patch.object(model, 'is_in_breakable_cuda_graph', return_value=True), mock.patch.object(model, 'get_tc_piecewise_forward_context', return_value=object()), mock.patch.object(model, 'get_is_capture_mode', return_value=True):
            result = model.Qwen4ExpPLELayer._get_prefetch_buffer(layer, 4, None)
        self.assertFalse(layer._graph_prefetch_buffers)
        self.assertEqual(result.data_ptr(), layer._eager_prefetch_buffer.data_ptr())


if __name__ == '__main__':
    unittest.main()
