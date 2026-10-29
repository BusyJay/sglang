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
            # Exercise the serving dispatch rather than only the model helper.
            with patch.object(
                runner.model,
                "forward_last_logits",
                wraps=runner.model.forward_last_logits,
            ) as forward:
                runner.prefill("last-only", self.tokens[:4], self.tokens[:4], [], [], 0)
                self.assertEqual(forward.call_count, 1)
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

    @staticmethod
    def _expanded_mla_reference(attn, x, cache):
        """Original expanded-KV equations, deliberately including dense masking."""
        import mlx.core as mx

        a = attn.args
        b, t, _ = x.shape
        qr = attn.q_a_layernorm(attn.q_a_proj(x))
        q = (
            attn.q_b_proj(qr)
            .reshape(b, t, a.num_attention_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        kv = attn.kv_a_layernorm(attn.kv_a_proj_with_mqa(x))
        if cache[0] is not None:
            kv = mx.concatenate([cache[0], kv], axis=1)
        cache[0] = kv
        selected = attn.indexer(x, qr, cache, cache.offset)
        cache.offset += t
        positions = mx.arange(kv.shape[1])
        if selected is None:
            mask = positions[None, :] <= (cache.offset - t + mx.arange(t))[:, None]
            mask = mask[None, None]
        else:
            mask = (
                (positions == selected[..., None]) & (selected[..., None] >= 0)
            ).any(-2)[:, None]
        expanded = (
            attn.kv_b_proj(kv)
            .reshape(b, -1, a.num_attention_heads, a.qk_nope_head_dim + a.v_head_dim)
            .transpose(0, 2, 1, 3)
        )
        k, v = mx.split(expanded, [a.qk_nope_head_dim], axis=-1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=a.qk_nope_head_dim**-0.5, mask=mask
        )
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))

    def test_absorbed_sparse_mla_matches_expanded_kv(self):
        """Reassociation must preserve scale, head layout and query-specific KV selection."""
        import copy

        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLA, NativeCache

        for rank, dtype, quantized, select_tail in (
            (16, mx.float32, False, True),
            (512, mx.float32, False, True),
            (512, mx.bfloat16, False, True),
            (32, mx.float32, True, True),
            (16, mx.float32, False, False),
        ):
            with self.subTest(
                rank=rank, dtype=dtype, quantized=quantized, select_tail=select_tail
            ):
                args = copy.deepcopy(self.model.args)
                args.kv_lora_rank = rank
                args.index_kpool_always_select_tail = select_tail
                attn = MLA(args)
                attn.set_dtype(dtype)
                if quantized:
                    attn.kv_b_proj = nn.QuantizedLinear.from_linear(
                        attn.kv_b_proj, group_size=32, bits=5
                    )
                # Different rows and queries select different pools. The latter
                # chunks cross the sparse budget. Single-token steps exercise
                # every tail length, including no tail at a pool boundary;
                # the final chunk must still mask future pools per query.
                mx.random.seed(19)
                inputs = mx.random.normal((2, 17, args.hidden_size)).astype(dtype)
                actual_cache, expected_cache = NativeCache(3), NativeCache(3)
                chunks = [(0, 3), (3, 7), (7, 10)]
                chunks += [(start, start + 1) for start in range(10, 14)]
                chunks += [(14, 17)]
                for start, end in chunks:
                    # Bootstrap with a visible tail: the expanded SDPA oracle
                    # has undefined all-masked rows before the first full pool.
                    args.index_kpool_always_select_tail = select_tail or start == 0
                    x = inputs[:, start:end]
                    actual = attn(x, actual_cache)
                    expected = self._expanded_mla_reference(attn, x, expected_cache)
                    mx.eval(actual, expected)
                    tolerance = (0.03, 0.003) if dtype == mx.bfloat16 else (2e-4, 2e-5)
                    np.testing.assert_allclose(
                        np.array(actual.astype(mx.float32)),
                        np.array(expected.astype(mx.float32)),
                        rtol=tolerance[0],
                        atol=tolerance[1],
                    )
                self.assertEqual(actual_cache[0].shape, (2, 17, rank))

    def test_quantized_expert_decode_checkpoints_preserve_switch_glu(self):
        """Projection checkpoints preserve clamping, mixed bits and batch layout."""
        import copy

        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.switch_layers import SwitchGLU

        from sglang.srt.hardware_backend.mlx.models.glm5_next import ClampedSwitchGLU

        args = copy.deepcopy(self.model.args)
        args.moe_intermediate_size = 32
        for dtype in (mx.float32, mx.bfloat16):
            switch = ClampedSwitchGLU(args)
            switch.set_dtype(dtype)
            for name, bits in (("up_proj", 4), ("gate_proj", 4), ("down_proj", 5)):
                projection = getattr(switch, name)
                setattr(switch, name, projection.to_quantized(group_size=32, bits=bits))
            for batch, tokens in ((1, 1), (2, 1), (1, 40)):
                with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                    x = (
                        mx.random.normal((batch, tokens, args.hidden_size)) * 4
                    ).astype(dtype)
                    indices = (
                        mx.arange(batch * tokens * 2) % args.n_routed_experts
                    ).reshape(batch, tokens, 2)
                    actual = switch(x, indices)
                    expected = SwitchGLU.__call__(switch, x, indices)
                    mx.eval(actual, expected)
                    np.testing.assert_array_equal(
                        np.array(actual.astype(mx.float32)),
                        np.array(expected.astype(mx.float32)),
                    )

    def test_compiled_mhc_matches_reference_with_twenty_sinkhorn_rounds(self):
        """Fusing reductions must preserve stream orientation and BF16 cast points."""
        import copy

        import mlx.core as mx
        import numpy as np
        import torch
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextHyperConnection,
        )

        from sglang.srt.hardware_backend.mlx.models.glm5_next import hyper_connection

        config = copy.deepcopy(self.config)
        config.hc_mult, config.hc_sinkhorn_iters = 4, 20
        reference = Glm5NextTextHyperConnection(config).eval()
        torch.manual_seed(29)
        with torch.no_grad():
            reference.fn.normal_(std=0.1)
            reference.base.normal_(std=0.2)
            reference.scale.fill_(1.0)
        for dtype, torch_dtype in (
            (mx.float32, torch.float32),
            (mx.bfloat16, torch.bfloat16),
        ):
            inputs = torch.randn(2, 7, 4, config.hidden_size).to(torch_dtype)
            with torch.no_grad():
                post, comb, collapsed = reference(inputs)
                residual = comb.to(torch_dtype).transpose(-1, -2) @ inputs
            actual = hyper_connection(
                mx.array(inputs.float().numpy()).astype(dtype),
                mx.array(reference.fn.detach().numpy()),
                mx.array(reference.base.detach().numpy()),
                mx.array(reference.scale.detach().numpy()),
                config,
            )
            mx.eval(actual)
            for result, expected in zip(
                actual, (collapsed, post.to(torch_dtype), residual)
            ):
                np.testing.assert_allclose(
                    np.array(result.astype(mx.float32)),
                    expected.float().numpy(),
                    rtol=0.03 if dtype == mx.bfloat16 else 2e-4,
                    atol=0.02 if dtype == mx.bfloat16 else 2e-5,
                )

    def test_last_logits_preserves_full_forward_and_cache_continuation(self):
        """Serving must project only the final row while updating every layer's cache."""
        import mlx.core as mx
        import numpy as np

        expected_cache, actual_cache = self.model.make_cache(), self.model.make_cache()
        for start, end in ((0, 7), (7, 12), (12, 13)):
            inputs = mx.array([self.tokens[start:end]])
            expected = self.model(inputs, expected_cache)[:, -1:, :]
            with patch.object(self.model, "lm_head", wraps=self.model.lm_head) as head:
                actual = self.model.forward_last_logits(inputs, actual_cache)
                self.assertEqual(head.call_count, 1)
                self.assertEqual(head.call_args.args[0].shape[1], 1)
            mx.eval(expected, actual)
            np.testing.assert_allclose(
                np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5
            )
            self.assertTrue(all(c.offset == end for c in actual_cache))

    def test_sparse_indices_match_reference_pool_and_tail_selection(self):
        """Invalid future pools and incomplete tails must never become real KV reads."""
        import mlx.core as mx
        import numpy as np
        import torch
        from transformers.cache_utils import DynamicCache

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        attn = self.model.layers[3].self_attn
        reference = self.reference.layers[3].self_attn.indexer
        cache, ref_cache = NativeCache(3), DynamicCache(config=self.config)
        rng = np.random.default_rng(21)
        values = rng.normal(size=(2, 19, self.config.hidden_size)).astype(np.float32)
        for start, end in ((0, 5), (5, 9), (9, 10), (10, 11), (11, 19)):
            x = mx.array(values[:, start:end])
            qr = attn.q_a_layernorm(attn.q_a_proj(x))
            indices = attn.indexer(x, qr, cache, start)
            # The Transformers indexer uses MLA's KV length/current offset.
            latent = torch.zeros((2, 1, end - start, self.config.kv_lora_rank))
            ref_cache.update(latent, torch.zeros((2, 1, end - start, 0)), 3)
            with torch.no_grad():
                expected = reference(
                    torch.from_numpy(values[:, start:end]),
                    torch.from_numpy(np.array(qr)),
                    torch.ones((2, end - start), dtype=torch.bool),
                    ref_cache,
                )
            positions = np.arange(end)
            expected_mask = (
                (positions == expected.numpy()[..., None])
                & (expected.numpy()[..., None] >= 0)
            ).any(-2)
            if indices is None:
                actual_mask = np.broadcast_to(
                    positions <= np.arange(start, end)[:, None], expected_mask.shape
                )
            else:
                actual_indices = np.array(indices)
                self.assertLessEqual(
                    actual_indices.shape[-1],
                    self.config.index_topk + self.config.index_kpool - 1,
                )
                actual_mask = (
                    (positions == actual_indices[..., None])
                    & (actual_indices[..., None] >= 0)
                ).any(-2)
            np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_incremental_indexer_pool_growth_and_state_roundtrip(self):
        """Completed pools survive growth/restoration; raw history stays bounded."""
        import mlx.core as mx
        import numpy as np

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        indexer = self.model.layers[3].self_attn.indexer
        cache = NativeCache(4)
        values = (
            np.random.default_rng(21)
            .normal(size=(2, 275, self.config.hidden_size))
            .astype(np.float32)
        )
        for start, end in (
            (0, 3),
            (3, 5),
            (5, 255),
            (255, 259),
            (259, 260),
            (260, 275),
        ):
            x = mx.array(values[:, start:end])
            qr = mx.zeros((2, end - start, self.config.q_lora_rank))
            indices = indexer(x, qr, cache, start)
            mx.eval(cache.state, *([] if indices is None else [indices]))
            tail_length = end % self.config.index_kpool
            self.assertEqual(
                cache[1].shape[1] if cache[1] is not None else 0, tail_length
            )
            self.assertEqual(
                cache[2].shape[1] if cache[2] is not None else 0, tail_length
            )
            # Reusing old pools must preserve their original weighted means,
            # including after moving them into a larger allocation.
            n = end // self.config.index_kpool
            if n == 0:
                self.assertIsNone(cache[3])
                continue
            all_x = mx.array(values[:, : n * self.config.index_kpool])
            keys = indexer.k_norm(indexer.wk(all_x))
            gates = all_x @ indexer.index_kpool_compress_gate.T
            shape = (2, n, self.config.index_kpool, self.config.index_head_dim)
            probabilities = mx.softmax(
                gates.reshape(shape).astype(mx.float32)
                + indexer.index_kpool_compress_ape.astype(mx.float32),
                axis=2,
            ).astype(keys.dtype)
            pooled = (probabilities * keys.reshape(shape)).sum(axis=2)
            np.testing.assert_allclose(
                np.array(cache[3][:, :n]), np.array(pooled), rtol=2e-5, atol=2e-6
            )
            restored = NativeCache(4)
            restored.state = cache.state
            restored.offset = end
            cache = restored

    def test_async_layer_groups_preserve_chunked_state(self):
        """Group boundaries and a final partial group must preserve every layer's state."""
        import mlx.core as mx
        import numpy as np

        expected = self.model.model(mx.array([self.tokens]), self.model.make_cache())
        for group_size in (1, 3, 8):
            with (
                self.subTest(group_size=group_size),
                patch.object(self.model.model, "_eval_group_size", group_size),
            ):
                cache = self.model.make_cache()
                parts = [self.model.model(mx.array([self.tokens[:9]]), cache)]
                for token in self.tokens[9:]:
                    parts.append(self.model.model(mx.array([[token]]), cache))
                result = mx.concatenate(parts, axis=1)
                np.testing.assert_allclose(
                    np.array(result), np.array(expected), rtol=2e-4, atol=2e-5
                )
                self.assertTrue(
                    all(state.offset == len(self.tokens) for state in cache)
                )

    def test_kda_qkv_fusion_preserves_quantization_and_canonical_weights(self):
        """Load-time fusion must preserve mixed-bit fallback, state and save format."""
        import copy

        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        for policy in ("fp32", "bf16", "q4", "mixed"):
            with self.subTest(policy=policy):
                reference = copy.deepcopy(self.model.layers[0].self_attn)
                if policy == "bf16":
                    reference.set_dtype(mx.bfloat16)
                elif policy in ("q4", "mixed"):
                    for name in ("q_proj", "k_proj", "v_proj"):
                        bits = 5 if policy == "mixed" and name == "k_proj" else 4
                        setattr(
                            reference,
                            name,
                            nn.QuantizedLinear.from_linear(
                                getattr(reference, name), group_size=32, bits=bits
                            ),
                        )
                prepared = copy.deepcopy(reference)
                expected_weights = dict(tree_flatten(reference.parameters()))
                self.assertEqual(prepared.prepare_for_inference(), policy != "mixed")
                actual_weights = dict(tree_flatten(prepared.parameters()))
                self.assertEqual(set(actual_weights), set(expected_weights))
                for key, expected in expected_weights.items():
                    actual = actual_weights[key]
                    if actual.dtype == mx.bfloat16:
                        actual, expected = (
                            actual.astype(mx.float32),
                            expected.astype(mx.float32),
                        )
                    np.testing.assert_array_equal(np.array(actual), np.array(expected))
                x = mx.random.normal((2, 9, self.config.hidden_size)).astype(
                    reference.q_conv.conv.weight.dtype
                )
                actual_cache, expected_cache = NativeCache(4), NativeCache(4)
                for start, end in ((0, 7), (7, 8), (8, 9)):
                    actual = prepared(x[:, start:end], actual_cache)
                    expected = reference(x[:, start:end], expected_cache)
                    tolerance = (0.03, 0.003) if policy == "bf16" else (2e-4, 2e-5)
                    np.testing.assert_allclose(
                        np.array(actual.astype(mx.float32)),
                        np.array(expected.astype(mx.float32)),
                        rtol=tolerance[0],
                        atol=tolerance[1],
                    )
                # A later parameter replacement must not serve stale fused data.
                if policy in ("q4", "mixed"):
                    prepared.q_proj.scales = prepared.q_proj.scales * 1.01
                    reference.q_proj.scales = reference.q_proj.scales * 1.01
                else:
                    prepared.q_proj.weight = prepared.q_proj.weight + 0.01
                    reference.q_proj.weight = reference.q_proj.weight + 0.01
                np.testing.assert_allclose(
                    np.array(prepared(x[:, :1], NativeCache(4)).astype(mx.float32)),
                    np.array(reference(x[:, :1], NativeCache(4)).astype(mx.float32)),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_kda_fusion_does_not_retain_duplicate_projection_storage(self):
        """Canonical Q/K/V parameters must share the fused buffer after preparation."""
        import gc

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import KDA

        attention = KDA(self.model.args)
        mx.eval(attention.parameters())
        gc.collect()
        before = mx.get_active_memory()
        self.assertTrue(attention.prepare_for_inference())
        mx.eval(attention.parameters())
        gc.collect()
        # Three FP32 64x32 matrices occupy 24 KiB. Keeping those plus a
        # concatenated copy would exceed this allocator bookkeeping allowance.
        self.assertLessEqual(mx.get_active_memory() - before, 4096)

    def test_moe_sorted_batch_matches_reference(self):
        """Cross the SwitchGLU sorting threshold while preserving clamp and row order."""
        import mlx.core as mx
        import numpy as np
        import torch

        values = (
            np.random.default_rng(17)
            .normal(size=(2, 40, self.config.hidden_size))
            .astype(np.float32)
        )
        actual = self.model.layers[3].mlp(mx.array(values))
        with torch.no_grad():
            expected = self.reference.layers[3].mlp(torch.from_numpy(values)).numpy()
        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
