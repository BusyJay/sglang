from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

common_module = load_source(
    "qwen38_common_under_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_common.py",
)
filtered_safetensors_weights_iterator = (
    common_module.filtered_safetensors_weights_iterator
)
should_skip_checkpoint_tensor = common_module.should_skip_checkpoint_tensor
validate_qwen38_model_config = common_module.validate_qwen38_model_config


class _FakeSafeOpen:
    def __init__(self, names):
        self.names = names
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self):
        return self.names

    def get_tensor(self, name):
        self.requested.append(name)
        return f"tensor:{name}"


class TestLoaderFilter(unittest.TestCase):
    def test_name_filter(self):
        self.assertTrue(
            should_skip_checkpoint_tensor(
                "model.language_model.layers.0.mlp.experts.7.gate_proj.weight",
                language_model_only=True,
            )
        )
        self.assertFalse(
            should_skip_checkpoint_tensor(
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                language_model_only=True,
            )
        )
        self.assertTrue(
            should_skip_checkpoint_tensor(
                "model.visual.blocks.0.attn.qkv.weight", language_model_only=True
            )
        )

    def test_get_tensor_is_never_called_for_skipped_names(self):
        names = [
            "model.language_model.layers.0.mlp.experts.7.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.embed_tokens.weight",
            "model.mtp.layers.0.mlp.experts.7.gate_proj.weight",
            "model.mtp.embed_tokens.weight",
            "model.mtp.lm_head.weight",
            "model.language_model.layers.0.rotary_emb.inv_freq",
        ]
        handle = _FakeSafeOpen(names)
        with tempfile.TemporaryDirectory() as value:
            filename = Path(value) / "fixture.safetensors"
            filename.touch()
            fake_safetensors = types.SimpleNamespace(
                safe_open=mock.Mock(return_value=handle)
            )
            with mock.patch.dict("sys.modules", {"safetensors": fake_safetensors}):
                audit = {}
                loaded = list(
                    filtered_safetensors_weights_iterator([str(filename)], audit=audit)
                )
        expected = [names[1], names[3]]
        self.assertEqual(
            audit,
            {"mtp": 3, "rotary": 1, "routed_experts": 1, "visual": 1, "read_tensor": 2},
        )
        self.assertEqual(handle.requested, expected)
        self.assertEqual([name for name, _ in loaded], expected)

    def test_exact_qwen38_shape_contract(self):
        text = SimpleNamespace(
            num_hidden_layers=48,
            num_experts=512,
            num_experts_per_tok=10,
            hidden_size=2560,
            moe_intermediate_size=640,
        )
        config = SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"], text_config=text
        )
        self.assertEqual(validate_qwen38_model_config(config), [])
        text.num_experts = 511
        self.assertIn(
            "num_experts must be 512", validate_qwen38_model_config(config)[0]
        )


if __name__ == "__main__":
    unittest.main()
