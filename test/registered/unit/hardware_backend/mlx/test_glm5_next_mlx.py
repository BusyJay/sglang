"""GLM5-Next MLX must preserve the reference's KDA, pooled DSA and mHC.

Small random reference weights exercise sparse selection above the top-k budget
and cached continuation across incomplete k-pools. This catches treating the
checkpoint as ordinary GLM DSA, dropping the gate bound, or transposing mHC.
"""

import importlib.util
import os
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=6, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestGlm5NextMlx(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # M5 defaults to reduced-precision FP32 GEMM. The oracle here is
        # full FP32 PyTorch CPU; set this before the first MLX operation.
        precision_env = patch.dict(os.environ, {"MLX_ENABLE_TF32": "0"})
        precision_env.start()
        cls.addClassCleanup(precision_env.stop)
        import mlx.core as mx
        import numpy as np
        import torch
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model, ModelArgs

        torch.manual_seed(42)
        cls.config = Glm5NextTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=0,
            v_head_dim=16,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=8,
            index_kpool=4,
            linear_num_heads=2,
            linear_head_dim=32,
            linear_conv_kernel_dim=4,
            hc_mult=2,
            hc_sinkhorn_iters=5,
            swiglu_limit=0.05,
            pad_token_id=None,
        )
        cls.config._attn_implementation = "eager"
        cls.reference = Glm5NextTextModel(cls.config).eval()
        cfg = cls.config.to_dict()
        cfg.pop("architectures", None)
        cfg.update(mhc=True, scoring_func="sigmoid", max_position_embeddings=128)
        cfg["linear_attn_config"] = dict(
            num_heads=2, head_dim=32, short_conv_kernel_size=4, gate_lower_bound=-5.0
        )
        cls.model = Model(ModelArgs(model_type="glm5_next", text_config=cfg))
        weights = {}
        for key, tensor in cls.reference.state_dict().items():
            value = tensor.detach().float().numpy()
            key = "model." + key
            if ".attn_hc." in key or ".ffn_hc." in key:
                key = key.replace(".attn_hc.", ".hc_attn_").replace(
                    ".ffn_hc.", ".hc_ffn_"
                )
            if ".self_attn.forget_gate." in key:
                key = key.replace(".forget_gate.", ".")
            if key.endswith(".self_attn.conv1d.weight"):
                for name, part in zip(("q", "k", "v"), np.split(value, 3, axis=0)):
                    weights[key.replace("conv1d.weight", name + "_conv1d.weight")] = (
                        mx.array(part)
                    )
                continue
            if ".mlp.experts.gate_up_proj" in key:
                gate, up = np.split(value, 2, axis=1)
                for i in range(4):
                    prefix = key.replace("experts.gate_up_proj", f"experts.{i}.")
                    weights[prefix + "gate_proj.weight"] = mx.array(gate[i])
                    weights[prefix + "up_proj.weight"] = mx.array(up[i])
                continue
            if ".mlp.experts.down_proj" in key:
                for i in range(4):
                    weights[
                        key.replace(
                            "experts.down_proj", f"experts.{i}.down_proj.weight"
                        )
                    ] = mx.array(value[i])
                continue
            weights[key] = mx.array(value)
        # Test only the headless text trunk; the lm_head is a separate checkpoint tensor.
        sanitized = cls.model.sanitize(weights)
        cls.model.model.load_weights(
            [(k.removeprefix("model."), v) for k, v in sanitized.items()]
        )
        cls.tokens = list(range(1, 14))

    def test_mlx_startup_resolves_without_cuda_dsa_defaults(self):
        import json
        import tempfile
        from pathlib import Path

        from sglang.srt.arg_groups.overrides import resolved_view
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx
        from sglang.srt.server_args import prepare_server_args

        with tempfile.TemporaryDirectory() as directory:
            config = dict(
                model_type="glm5_next",
                architectures=["Glm5NextForConditionalGeneration"],
                text_config=self.model.args.text_config,
            )
            Path(directory, "config.json").write_text(json.dumps(config))
            use_mlx.cache_clear()
            try:
                with patch.dict(os.environ, {"SGLANG_USE_MLX": "1"}):
                    args = prepare_server_args(
                        [
                            "--model-path",
                            directory,
                            "--device",
                            "mps",
                            "--context-length",
                            "128",
                            "--cuda-graph-backend-decode=disabled",
                            "--cuda-graph-backend-prefill=disabled",
                        ]
                    )
                    args.resolve_once()
                    resolved = resolved_view(args)
                    self.assertTrue(resolved.disable_radix_cache)
                    self.assertTrue(resolved.disable_overlap_schedule)
                    self.assertEqual(resolved.page_size, 1)
                    self.assertNotEqual(resolved.attention_backend, "dsa")
                    self.assertFalse(ModelConfig.from_server_args(args).is_multimodal)
            finally:
                use_mlx.cache_clear()

    def test_sparse_reference_and_cached_continuation(self):
        import mlx.core as mx
        import numpy as np
        import torch

        with torch.no_grad():
            expected = self.reference(
                torch.tensor([self.tokens]), use_cache=False
            ).last_hidden_state.numpy()
        full = self.model.model(mx.array([self.tokens]), self.model.make_cache())
        np.testing.assert_allclose(np.array(full), expected, rtol=2e-4, atol=2e-5)
        cache = self.model.make_cache()
        parts = []
        # Split across complete and incomplete k-pools, then decode token by token.
        for start, end in ((0, 3), (3, 9), (9, 10), (10, 11), (11, 12), (12, 13)):
            parts.append(self.model.model(mx.array([self.tokens[start:end]]), cache))
        np.testing.assert_allclose(
            np.array(mx.concatenate(parts, axis=1)), expected, rtol=2e-4, atol=2e-5
        )
        self.assertTrue(all(c.offset == len(self.tokens) for c in cache))

    def test_quantized_loader_and_request_cache_isolation(self):
        import copy
        import json
        import tempfile
        from pathlib import Path

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        model = copy.deepcopy(self.model)

        def quantize_module(path, module):
            if not hasattr(module, "to_quantized") or module.weight.shape[-1] % 32:
                return False
            return {"group_size": 32, "bits": 5 if "up_proj" in path else 4}

        nn.quantize(model, class_predicate=quantize_module, group_size=32, bits=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = dict(
                model_type="glm5_next",
                text_config=self.model.args.text_config,
                quantization={"group_size": 32, "bits": 4},
            )
            (path / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(
                str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
            )
            runner = MlxModelRunner(str(path), disable_radix_cache=True, pool_size=128)
            self.addCleanup(runner.clear)
            runner.init_cache_pools(None)
            prompts = {"a": self.tokens[:9], "b": self.tokens[:3]}
            expected = {}
            for rid, prompt in prompts.items():
                cache = model.make_cache()
                token = int(
                    mx.argmax(
                        model(mx.array([prompt]), cache)[:, -1, :], axis=-1
                    ).item()
                )
                expected[rid] = int(
                    mx.argmax(
                        model(mx.array([[token]]), cache)[:, -1, :], axis=-1
                    ).item()
                )
                if rid == "a":
                    runner.prefill(
                        rid, prompt[:3], prompt[:3], [], [], 0, needs_logits=False
                    )
                    actual = runner.extend(rid, prompt[3:], [])
                else:
                    actual = runner.prefill(rid, prompt, prompt, [], [], 0)
                self.assertEqual(actual, token)
            pending = runner.decode_batch_start(["b", "a"])
            runner.eval_pending(pending)
            self.assertEqual(
                runner.decode_batch_finalize(pending), [expected["b"], expected["a"]]
            )


if __name__ == "__main__":
    unittest.main()
