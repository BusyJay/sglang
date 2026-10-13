"""CPU checks for loading BF16 GDN weights directly into final packed storage."""
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.qwen38_gdn_preallocated import preallocated_gdn_weight
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedModelOptConfig
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet


class TestPreallocatedGdn(unittest.TestCase):
    def projections(self, enabled=True, tp_size=1):
        root = Path(__file__).resolve().parents[4]
        config = json.loads((root / 'flash-next-plan/model-metadata/config.json').read_text())
        delegate = ModelOptMixedPrecisionConfig.from_config(config['quantization_config'])
        delegate.packed_modules_mapping = {
            'in_proj_qkvz': ['in_proj_qkv', 'in_proj_z'],
            'in_proj_ba': ['in_proj_b', 'in_proj_a'],
        }
        wrapper = Qwen38CachedModelOptConfig(delegate, None, gdn_value_heads=2 if enabled else None)
        layers = []
        for name, sizes in [('in_proj_qkvz', [2, 2, 2, 2]), ('in_proj_ba', [2, 2])]:
            layers.append(MergedColumnParallelLinear(
                4, sizes, bias=False, params_dtype=torch.bfloat16,
                quant_config=wrapper, prefix='model.layers.0.linear_attn.' + name,
                tp_rank=0, tp_size=tp_size,
            ))
        gdn = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(gdn)
        gdn.in_proj_qkvz, gdn.in_proj_ba = layers
        gdn._fused_in_proj_weight = None
        for layer in layers:
            gdn._bind_packed_weight_loaders(layer)
        self.assertEqual(wrapper._pending_gdn_weights, {})
        return gdn

    def finalize(self, gdn):
        with patch('sglang.srt.models.qwen3_5._is_cuda', True), patch(
            'sglang.srt.models.qwen3_5._use_aiter', False
        ), patch('sglang.srt.models.qwen3_5.get_lora', return_value=types.SimpleNamespace(
            enable_lora=False, lora_paths=None
        )):
            gdn.finalize_fused_in_proj()

    def test_checkpoint_shards_load_into_shared_storage_without_repacking(self):
        gdn = self.projections()
        qkvz, ba = gdn.in_proj_qkvz, gdn.in_proj_ba
        qkv = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        z = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + 32
        b = torch.full((2, 4), 3, dtype=torch.bfloat16)
        a = torch.full((2, 4), 7, dtype=torch.bfloat16)
        qkvz.weight.weight_loader(qkvz.weight, qkv, (0, 1, 2))
        qkvz.weight.weight_loader(qkvz.weight, z, 3)
        ba.weight.weight_loader(ba.weight, b, 0)
        ba.weight.weight_loader(ba.weight, a, 1)
        expected = torch.cat([qkv, z, b, a])
        packed = preallocated_gdn_weight(qkvz, ba)
        self.assertTrue(torch.equal(packed, expected))
        with patch('torch.cat', side_effect=AssertionError('unnecessary repacking')):
            self.finalize(gdn)
            self.finalize(gdn)
        self.assertIs(gdn._fused_in_proj_weight, packed)
        self.assertEqual(qkvz.weight.untyped_storage().data_ptr(), ba.weight.untyped_storage().data_ptr())
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        self.assertTrue(torch.equal(torch.nn.functional.linear(x, packed), torch.cat([
            torch.nn.functional.linear(x, qkvz.weight), torch.nn.functional.linear(x, ba.weight)
        ], dim=-1)))

    def test_tensor_parallel_shards_keep_their_offsets(self):
        gdn = self.projections(tp_size=2)
        qkvz, ba = gdn.in_proj_qkvz, gdn.in_proj_ba
        expected = []
        for shard in range(4):
            value = torch.full((2, 4), shard + 1, dtype=torch.bfloat16)
            qkvz.weight.weight_loader(qkvz.weight, value, shard)
            expected.append(value[:1])
        for shard in range(2):
            value = torch.full((2, 4), shard + 5, dtype=torch.bfloat16)
            ba.weight.weight_loader(ba.weight, value, shard)
            expected.append(value[:1])
        self.assertTrue(torch.equal(preallocated_gdn_weight(qkvz, ba), torch.cat(expected)))

    def test_disabled_path_still_repacks_loaded_values(self):
        gdn = self.projections(enabled=False)
        self.assertIsNone(preallocated_gdn_weight(gdn.in_proj_qkvz, gdn.in_proj_ba))
        with torch.no_grad():
            gdn.in_proj_qkvz.weight.fill_(1)
            gdn.in_proj_ba.weight.fill_(2)
        with patch('torch.cat', wraps=torch.cat) as cat:
            self.finalize(gdn)
        self.assertEqual(cat.call_count, 1)
        self.assertTrue(torch.equal(gdn._fused_in_proj_weight[:8], gdn.in_proj_qkvz.weight))

    def test_non_bf16_projection_keeps_original_allocation(self):
        gdn = self.projections()
        wrapper = gdn.in_proj_qkvz.quant_config
        linear = MergedColumnParallelLinear(
            4, [2, 2, 2, 2], bias=False, params_dtype=torch.float32,
            quant_config=wrapper, prefix='model.layers.1.linear_attn.in_proj_qkvz',
            tp_rank=0, tp_size=1,
        )
        self.assertFalse(hasattr(linear, '_qwen38_gdn_packed_weight'))
        self.assertEqual(wrapper._pending_gdn_weights, {})

    def test_quantized_projection_keeps_delegate_without_resolving_partner(self):
        method = object()
        def unexpected_resolve(prefix):
            raise AssertionError('quantized fallback should not resolve a BF16 partner')
        delegate = types.SimpleNamespace(
            packed_modules_mapping={},
            get_quant_method=lambda layer, prefix: method,
            resolve_quant_algo=unexpected_resolve,
        )
        wrapper = Qwen38CachedModelOptConfig(delegate, None, gdn_value_heads=2)
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        self.assertIs(wrapper.get_quant_method(
            layer, 'model.layers.0.linear_attn.in_proj_qkvz'
        ), method)

    def test_unrelated_linear_keeps_original_allocation(self):
        gdn = self.projections()
        wrapper = gdn.in_proj_qkvz.quant_config
        linear = MergedColumnParallelLinear(
            4, [2, 2], bias=False, params_dtype=torch.bfloat16,
            quant_config=wrapper, prefix='model.layers.0.linear_attn.out_proj',
            tp_rank=0, tp_size=1,
        )
        self.assertFalse(hasattr(linear, '_qwen38_gdn_packed_weight'))
        self.assertEqual(wrapper._pending_gdn_weights, {})

    def test_rebound_parameter_does_not_reuse_stale_storage(self):
        gdn = self.projections()
        with torch.no_grad():
            gdn.in_proj_qkvz.weight.fill_(1)
            gdn.in_proj_ba.weight.fill_(2)
        gdn.in_proj_ba.weight.data = torch.full_like(gdn.in_proj_ba.weight, 9)
        self.assertIsNone(preallocated_gdn_weight(gdn.in_proj_qkvz, gdn.in_proj_ba))
        self.finalize(gdn)
        self.assertTrue(torch.all(gdn._fused_in_proj_weight[8:] == 9))


if __name__ == '__main__':
    unittest.main()
