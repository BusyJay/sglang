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

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestGlm5NextMlx(CustomTestCase):
    def test_missing_native_prefix_snapshot_keeps_prefill_bound(self):
        """A cache miss must not expand a small scheduler suffix into one giant forward."""
        import copy
        from types import SimpleNamespace

        import mlx.core as mx
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.runtime_context import get_context

        model = copy.deepcopy(self.model)
        runner = object.__new__(MlxModelRunner)
        runner.model = model
        runner._mtp = None
        runner._enable_sampling = False
        runner._req_to_token_pool = None
        runner._cache_layout = SimpleNamespace(has_auxiliary_state=False)
        lengths = []

        def trunk(inputs, cache):
            lengths.append(inputs.shape[1])
            return model.model(inputs, cache=cache)

        runner._trunk = trunk
        tokens = [10, 11, 12, 13] * 16 + [19, 23, 29]
        cache = model.make_cache()
        with get_context().override_server_args(
            model_path="dummy", chunked_prefill_size=16
        ):
            with patch.object(runner, "_acquire_cache", model.make_cache):
                pending = runner._prefill_native_prefix(
                    "replay",
                    tokens[64:],
                    tokens,
                    64,
                    [1, 2, 3],
                    1,
                    None,
                    True,
                    None,
                    None,
                )
                mx.eval(pending.lazy_token, [c.state for c in pending.cache])
        reference = None
        for start in range(0, len(tokens), 16):
            hidden = model.model(
                mx.array([tokens[start : start + 16]], mx.int32), cache
            )
            mx.eval(hidden, [c.state for c in cache])
            reference = mx.argmax(model.lm_head(hidden[:, -1:])[:, -1], -1)
        self.assertEqual(lengths, [16, 16, 16, 16])
        self.assertTrue(mx.array_equal(pending.lazy_token, reference).item())
        for actual, expected in zip(pending.cache, cache):
            self.assertEqual(actual.offset, len(tokens))
            for (_, a), (_, b) in zip(
                tree_flatten(actual.state), tree_flatten(expected.state)
            ):
                if a is None:
                    self.assertIsNone(b)
                else:
                    self.assertTrue(mx.array_equal(a, b).item())

    def test_latent_append_snapshots_restore_inside_and_across_blocks(self):
        """Rejected suffixes cannot overwrite a retained prefix or append block."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLACache
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        cache = MLACache(4)
        expected = mx.arange(259 * 128).reshape(1, 259, 128).astype(mx.float32)
        cache.append_latent(expected[:, :1])
        cache.offset = 1
        snapshots = []
        for start, stop in ((1, 255), (255, 256), (256, 257), (257, 259)):
            snapshots.append(clone_cache(cache))
            cache.append_latent(expected[:, start:stop])
            cache.offset = stop
            self.assertTrue(mx.array_equal(cache[0], expected[:, :stop]).item())
        for snapshot in snapshots:
            self.assertTrue(
                mx.array_equal(snapshot[0], expected[:, : snapshot.offset]).item()
            )
            restored = clone_cache(snapshot)
            suffix = mx.full((1, 2, 128), -7.0)
            restored.append_latent(suffix)
            restored.offset += 2
            self.assertTrue(mx.array_equal(restored[0][:, -2:], suffix).item())
            self.assertTrue(
                mx.array_equal(snapshot[0], expected[:, : snapshot.offset]).item()
            )
            roundtrip = MLACache(4)
            roundtrip.state = snapshot.state
            roundtrip.offset = snapshot.offset
            self.assertTrue(mx.array_equal(roundtrip[0], snapshot[0]).item())

    def test_segmented_mla_verification_seal_and_rejected_suffix(self):
        """Sealing an append block must hide later verified rows after rejection."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            MLACache,
            NativeCache,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        args = copy.copy(self.model.args)
        mx.random.seed(39)
        for rank in (256, 512):
            args.kv_lora_rank = rank
            attn = MLA(args)
            for length in (255, 511):
                native = NativeCache(4)
                mx.eval(
                    attn(mx.random.normal((1, length, args.hidden_size)), native),
                    native.cache,
                )
                segmented = MLACache(4)
                segmented.cache = [
                    v.view(v.dtype) if v is not None else None for v in native.cache
                ]
                segmented.offset = length
                if length > 256:
                    segmented.cache[0] = native[0][:, :256]
                    segmented.cache.append(
                        mx.pad(native[0][:, 256:], [(0, 0), (0, 1), (0, 0)])
                    )
                x = mx.random.normal((1, 5, args.hidden_size))
                expected, states = [], []
                for j in range(5):
                    expected.append(attn(x[:, j : j + 1], native))
                    states.append(clone_cache(native))
                actual, snapshots = attn.verify_rows(x, segmented)
                self.assertLess(
                    mx.max(mx.abs(actual - mx.concatenate(expected, 1))).item(), 2e-5
                )
                for snapshot, state in zip(snapshots, states):
                    self.assertEqual(snapshot.offset, state.offset)
                    for i in range(4):
                        if state[i] is None:
                            self.assertIsNone(snapshot[i])
                        else:
                            self.assertTrue(
                                mx.array_equal(snapshot[i], state[i]).item()
                            )
                    continuation = mx.random.normal((1, 1, args.hidden_size))
                    self.assertLess(
                        mx.max(
                            mx.abs(
                                attn(continuation, clone_cache(snapshot))
                                - attn(continuation, clone_cache(state))
                            )
                        ).item(),
                        2e-5,
                    )

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
        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model, ModelArgs
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

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

    def _capture_layer_oracle(self, model, inputs, cache, capture_layer_ids):
        """Execute singleton rows and explicit layers, bypassing target wrappers."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        hidden_rows, feature_rows, snapshots = [], [], []
        for j in range(inputs.shape[1]):
            x = model.model.embed_tokens(inputs[:, j : j + 1])
            if model.args.mhc:
                x = mx.repeat(x[..., None, :], model.args.hc_mult, axis=-2)
            completed = {}
            for i, layer in enumerate(model.layers):
                x = layer(x, cache[i])
                # The feature is the completed residual's arithmetic mHC mean,
                # not a branch projection, one stream, or the final RMSNorm.
                if i in capture_layer_ids:
                    completed[i] = (
                        mx.sum(x, axis=-2) / model.args.hc_mult
                        if model.args.mhc
                        else x
                    )
            contracted = (
                mx.sum(x, axis=-2) / model.args.hc_mult if model.args.mhc else x
            )
            hidden_rows.append(model.model.norm(contracted))
            feature_rows.append(
                mx.concatenate([completed[i] for i in capture_layer_ids], axis=-1)
            )
            snapshots.append([clone_cache(c) for c in cache])
        hidden = mx.concatenate(hidden_rows, axis=1)
        logits = mx.concatenate([model.lm_head(v) for v in hidden_rows], axis=1)
        return logits, hidden, snapshots, mx.concatenate(feature_rows, axis=1)

    def _assert_capture_array_close(self, actual, expected):
        import numpy as np

        self.assertEqual(actual.shape, expected.shape)
        # Use the existing small FP32 fixture's reference tolerance. Compiled
        # grouping may change reductions, but must preserve the target equation.
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5
        )

    def _assert_capture_cache_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertEqual(got.offset, want.offset)
            self.assertEqual(len(got.cache), len(want.cache))
            for a, b in zip(got.cache, want.cache):
                if a is None or b is None:
                    self.assertIs(a, b)
                else:
                    self._assert_capture_array_close(a, b)

    def test_target_capture_matches_layers_across_prefill_decode_groups(self):
        """Captures inside compiled groups must preserve layer order and caches."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        # Deliberately non-sorted: feature columns follow the caller's order.
        capture_ids = (3, 0, 1)
        for compiled, group_size in ((False, 4), (True, 4), (True, 2)):
            with self.subTest(compiled=compiled, group_size=group_size):
                model = copy.deepcopy(self.model)
                model.configure_fixed_groups(compiled)
                model.configure_eval_groups(
                    decode=group_size, prefill=group_size, verification=group_size
                )
                actual_cache, plain_cache, oracle_cache = (
                    model.make_cache() for _ in range(3)
                )
                # Warm the uncaptured group before requesting an internal tap.
                prefix = mx.array([[19, 23, 29]])
                for cache in (actual_cache, plain_cache, oracle_cache):
                    mx.eval(model.forward_with_hidden(prefix, cache))
                for tokens in ((5, 7, 11, 13, 17, 31), (37,), (41,)):
                    inputs = mx.array([tokens])
                    expected = self._capture_layer_oracle(
                        model, inputs, oracle_cache, capture_ids
                    )
                    actual = model.forward_with_capture(
                        inputs, actual_cache, capture_ids
                    )
                    plain = model.forward_with_hidden(inputs, plain_cache)
                    self.assertEqual(len(actual), 3)
                    for got, want in zip(actual[:2], expected[:2]):
                        self._assert_capture_array_close(got, want)
                    self._assert_capture_array_close(actual[2], expected[3])
                    self.assertEqual(actual[2].shape, (1, len(tokens), 96))
                    for got, want in zip(actual[:2], plain):
                        self._assert_capture_array_close(got, want)
                    self._assert_capture_cache_close(actual_cache, plain_cache)
                    self._assert_capture_cache_close(actual_cache, oracle_cache)
                    # The final layer tap precedes norm, so normalizing its
                    # feature slice must recover hidden, not another feature.
                    self._assert_capture_array_close(
                        model.model.norm(actual[2][..., :32]), actual[1]
                    )
                    self.assertGreater(
                        mx.max(mx.abs(actual[2][..., :32] - actual[1])).item(),
                        1e-3,
                    )
                token_cache = [clone_cache(c) for c in actual_cache]
                logits_cache = [clone_cache(c) for c in actual_cache]
                inputs = mx.array([[43]])
                token, hidden, features = model.forward_with_capture(
                    inputs, token_cache, capture_ids, token_only=True
                )
                logits, expected_hidden = model.forward_with_hidden(
                    inputs, logits_cache
                )
                self.assertTrue(mx.array_equal(token, mx.argmax(logits, -1)).item())
                self._assert_capture_array_close(hidden, expected_hidden)
                oracle = self._capture_layer_oracle(
                    model, inputs, [clone_cache(c) for c in actual_cache], capture_ids
                )
                self._assert_capture_array_close(features, oracle[3])
                self._assert_capture_cache_close(token_cache, logits_cache)

    def test_verification_capture_preserves_rejection_snapshots(self):
        """Every verified row tap and rejected-prefix continuation must agree."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        capture_ids = (2, 0, 3)
        for mode in ("serial", "head", "ffn"):
            for count in (1, 4):
                with self.subTest(mode=mode, count=count):
                    model = copy.deepcopy(self.model)
                    model.configure_mtp_verification(mode)
                    model.configure_fixed_groups(True)
                    model.configure_eval_groups(decode=4, prefill=4, verification=4)
                    base = model.make_cache()
                    mx.eval(model(mx.array([[1, 3, 5, 7, 9, 11]]), base))
                    inputs = mx.array([[13, 17, 19, 23][:count]])
                    oracle_cache = [clone_cache(c) for c in base]
                    expected = self._capture_layer_oracle(
                        model, inputs, oracle_cache, capture_ids
                    )
                    plain_cache = [clone_cache(c) for c in base]
                    plain = model.verify_mtp_rows(inputs, plain_cache)
                    self.assertEqual(len(plain), 3)
                    for token_only in (False, True):
                        with self.subTest(token_only=token_only):
                            actual_cache = [clone_cache(c) for c in base]
                            actual = model.verify_mtp_rows(
                                inputs,
                                actual_cache,
                                capture_layer_ids=capture_ids,
                                token_only=token_only,
                            )
                            self.assertEqual(len(actual), 4)
                            if token_only:
                                self.assertTrue(
                                    mx.array_equal(
                                        actual[0], mx.argmax(plain[0], -1)
                                    ).item()
                                )
                            else:
                                self._assert_capture_array_close(actual[0], plain[0])
                                self._assert_capture_array_close(actual[0], expected[0])
                            self._assert_capture_array_close(actual[1], expected[1])
                            self._assert_capture_array_close(actual[3], expected[3])
                            self._assert_capture_cache_close(actual_cache, plain_cache)
                            self._assert_capture_cache_close(actual_cache, oracle_cache)
                            self.assertEqual(len(actual[2]), count)
                            for j, snapshot in enumerate(actual[2]):
                                self._assert_capture_cache_close(snapshot, plain[2][j])
                                self._assert_capture_cache_close(
                                    snapshot, expected[2][j]
                                )
                                self.assertTrue(
                                    all(c.offset == 7 + j for c in snapshot)
                                )
                                # Reject the remaining draft rows and consume a
                                # different token from each retained boundary.
                                resumed = [clone_cache(c) for c in snapshot]
                                continuation = mx.array([[31 + j]])
                                got = model.forward_with_capture(
                                    continuation, resumed, capture_ids
                                )
                                want_cache = [clone_cache(c) for c in expected[2][j]]
                                want = self._capture_layer_oracle(
                                    model, continuation, want_cache, capture_ids
                                )
                                for a, b in zip(got[:2], want[:2]):
                                    self._assert_capture_array_close(a, b)
                                self._assert_capture_array_close(got[2], want[3])
                                self._assert_capture_cache_close(resumed, want_cache)
                                # Resuming a clone must not mutate its snapshot.
                                self._assert_capture_cache_close(
                                    snapshot, expected[2][j]
                                )

    def test_ragged_request_capture_keeps_features_and_prefix_owners_separate(self):
        """Ragged projection splits and rejected rows must stay request-local."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = copy.deepcopy(self.model)
        model.configure_mtp_verification("ffn")
        model.configure_fixed_groups(True)
        capture_ids = (3, 1, 0)
        # Singleton also exercises the requests API's delegation to verify_rows.
        for counts in ((3,), (1, 3, 5)):
            bases, inputs, expected = [], [], []
            for request, count in enumerate(counts):
                base = model.make_cache()
                prefix = mx.array([list(range(1 + request, 4 + 4 * request))])
                mx.eval(model(prefix, base))
                chain = mx.array(
                    [list(range(17 + 7 * request, 17 + 7 * request + count))]
                )
                bases.append(base)
                inputs.append(chain)
                expected.append(
                    self._capture_layer_oracle(
                        model, chain, [clone_cache(c) for c in base], capture_ids
                    )
                )
            plain_caches = [[clone_cache(c) for c in base] for base in bases]
            plain = model.verify_mtp_requests(inputs, plain_caches)
            self.assertEqual(len(plain), len(counts))
            self.assertTrue(all(len(entry) == 3 for entry in plain))
            for token_only in (False, True):
                with self.subTest(counts=counts, token_only=token_only):
                    caches = [[clone_cache(c) for c in base] for base in bases]
                    actual = model.verify_mtp_requests(
                        inputs,
                        caches,
                        capture_layer_ids=capture_ids,
                        token_only=token_only,
                    )
                    self.assertEqual(len(actual), len(counts))
                    for request, (got, want) in enumerate(zip(actual, expected)):
                        self.assertEqual(len(got), 4)
                        self.assertEqual(got[3].shape, (1, counts[request], 96))
                        if token_only:
                            self.assertTrue(
                                mx.array_equal(
                                    got[0], mx.argmax(plain[request][0], -1)
                                ).item()
                            )
                        else:
                            self._assert_capture_array_close(got[0], want[0])
                            self._assert_capture_array_close(got[0], plain[request][0])
                        self._assert_capture_array_close(got[1], want[1])
                        self._assert_capture_array_close(got[3], want[3])
                        self._assert_capture_cache_close(
                            caches[request], plain_caches[request]
                        )
                        self.assertEqual(len(got[2]), counts[request])
                        for j, snapshot in enumerate(got[2]):
                            self._assert_capture_cache_close(snapshot, want[2][j])
                            self._assert_capture_cache_close(
                                snapshot, plain[request][2][j]
                            )
                        # Use different rejection depths and continuation tokens
                        # per request, then check all owners again for aliasing.
                        retained = min(request, counts[request] - 1)
                        resumed = [clone_cache(c) for c in got[2][retained]]
                        oracle_cache = [clone_cache(c) for c in want[2][retained]]
                        continuation = mx.array([[47 + request]])
                        result = model.forward_with_capture(
                            continuation, resumed, capture_ids
                        )
                        oracle = self._capture_layer_oracle(
                            model, continuation, oracle_cache, capture_ids
                        )
                        for a, b in zip(result[:2], oracle[:2]):
                            self._assert_capture_array_close(a, b)
                        self._assert_capture_array_close(result[2], oracle[3])
                        self._assert_capture_cache_close(resumed, oracle_cache)
                    for got, want in zip(actual, expected):
                        for a, b in zip(got[2], want[2]):
                            self._assert_capture_cache_close(a, b)

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
                    self.assertFalse(resolved.disable_radix_cache)
                    self.assertTrue(resolved.uses_mamba_radix_cache)
                    self.assertEqual(resolved.mamba_radix_cache_strategy, "no_buffer")
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
            # Check the served token and consumed state across the head contract.
            reference_cache = model.make_cache()
            logits = model(mx.array([self.tokens[:4]]), reference_cache)
            expected_token = int(mx.argmax(logits[:, -1], -1).item())
            actual_token = runner.prefill("last-only", self.tokens[:4], self.tokens[:4], [], [], 0)
            self.assertEqual(actual_token, expected_token)
            self._assert_optimized_cache_equal(runner._req_caches["last-only"], reference_cache)
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

    def test_fp16_affine_ffn_preserves_bf16_residual_dtype(self):
        """Mixed-bit FP16 metadata must not promote dense/shared/routed FFNs."""
        import copy

        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.switch_layers import SwitchGLU

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLP,
            ClampedSwitchGLU,
            clamped_swiglu,
        )

        args = copy.deepcopy(self.model.args)
        args.moe_intermediate_size = 32
        for kind in ("dense", "shared", "routed"):
            block = (
                ClampedSwitchGLU(args)
                if kind == "routed"
                else MLP(args, 32 if kind == "shared" else 64)
            )
            block.match_affine_dtype = True
            block.set_dtype(mx.float16)
            for name, bits in (("up_proj", 4), ("gate_proj", 4), ("down_proj", 5)):
                proj = getattr(block, name)
                setattr(block, name, proj.to_quantized(group_size=32, bits=bits))
            for batch, tokens in ((1, 1), (2, 1), (1, 40)):
                with self.subTest(kind=kind, batch=batch, tokens=tokens):
                    x = mx.random.normal((batch, tokens, args.hidden_size)).astype(
                        mx.bfloat16
                    )
                    indices = (
                        mx.arange(batch * tokens * 2) % args.n_routed_experts
                    ).reshape(batch, tokens, 2)
                    fp16 = x.astype(mx.float16)
                    if kind == "routed":
                        actual = block(x, indices)
                        expected = SwitchGLU.__call__(block, fp16, indices)
                    else:
                        actual = block(x)
                        expected = block.down_proj(
                            clamped_swiglu(
                                block.gate_proj(fp16), block.up_proj(fp16), block.limit
                            )
                        )
                    expected = expected.astype(mx.bfloat16)
                    mx.eval(actual, expected)
                    self.assertEqual(actual.dtype, mx.bfloat16)
                    residual = x + (actual.sum(-2) if kind == "routed" else actual)
                    self.assertEqual(residual.dtype, mx.bfloat16)
                    np.testing.assert_array_equal(
                        np.array(actual.astype(mx.float32)),
                        np.array(expected.astype(mx.float32)),
                    )

        # Incompatible metadata retains the original mixed-dtype semantics.
        block = MLP(args, 64)
        block.match_affine_dtype = True
        block.set_dtype(mx.float16)
        for name in ("up_proj", "gate_proj", "down_proj"):
            proj = getattr(block, name)
            setattr(block, name, proj.to_quantized(group_size=32, bits=4))
        block.down_proj.scales = block.down_proj.scales.astype(mx.bfloat16)
        block.down_proj.biases = block.down_proj.biases.astype(mx.bfloat16)
        x = mx.ones((1, 1, args.hidden_size), mx.bfloat16)
        actual = block(x)
        expected = block.down_proj(
            clamped_swiglu(block.gate_proj(x), block.up_proj(x), block.limit)
        )
        mx.eval(actual, expected)
        self.assertEqual(actual.dtype, mx.float32)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_decode_ffn_candidates_preserve_updates_and_batch_fallback(self):
        """Compilation/fused gate-up preserve mixed bits and observe new arrays."""
        import copy

        import mlx.core as mx
        import numpy as np

        from sglang.srt.hardware_backend.mlx.models.glm5_next import ClampedSwitchGLU

        mx.random.seed(20261030)
        args = copy.deepcopy(self.model.args)
        args.moe_intermediate_size = 32
        original = ClampedSwitchGLU(args)
        original.set_dtype(mx.float16)
        for name, bits in (("gate_proj", 4), ("up_proj", 4), ("down_proj", 5)):
            p = getattr(original, name)
            setattr(original, name, p.to_quantized(group_size=32, bits=bits))
        for mode in ("compile", "gate_up"):
            block = copy.deepcopy(original)
            block.compile_decode = mode == "compile"
            block.fuse_gate_up = mode == "gate_up"
            canonical = tuple(
                value
                for proj in (block.gate_proj, block.up_proj)
                for value in (proj.weight, proj.scales, proj.biases)
            )
            block.prepare_for_inference()
            # Preparation must retain checkpoint arrays instead of replacing
            # them with slices of an eager full-expert concatenation.
            current = tuple(
                value
                for proj in (block.gate_proj, block.up_proj)
                for value in (proj.weight, proj.scales, proj.biases)
            )
            self.assertTrue(all(a is b for a, b in zip(canonical, current)))
            joined = block._fused_gate_up
            block.prepare_for_inference()
            self.assertIs(block._fused_gate_up, joined)
            reference = copy.deepcopy(original)
            for batch, tokens in ((1, 1), (2, 1), (1, 40)):
                x = mx.random.normal((batch, tokens, args.hidden_size)).astype(
                    mx.bfloat16
                )
                indices = (
                    mx.arange(batch * tokens * 2) % args.n_routed_experts
                ).reshape(batch, tokens, 2)
                actual, expected = block(x, indices), reference(x, indices)
                mx.eval(actual, expected)
                self.assertTrue(
                    mx.all(mx.isfinite(actual)).item(),
                    f"{mode=}, {batch=}, {tokens=}: candidate is non-finite",
                )
                self.assertTrue(
                    mx.all(mx.isfinite(expected)).item(),
                    f"{mode=}, {batch=}, {tokens=}: reference is non-finite",
                )
                np.testing.assert_allclose(
                    np.array(actual.astype(mx.float32)),
                    np.array(expected.astype(mx.float32)),
                    rtol=0.01,
                    atol=0.002,
                )
            block.gate_proj.scales = block.gate_proj.scales * 1.01
            reference.gate_proj.scales = reference.gate_proj.scales * 1.01
            x = x[:, :1]
            indices = indices[:, :1]
            actual, expected = block(x, indices), reference(x, indices)
            mx.eval(actual, expected)
            if mode == "gate_up":
                self.assertIsNone(block._fused_gate_up)
            np.testing.assert_allclose(
                np.array(actual.astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                rtol=0.01,
                atol=0.002,
            )

    def test_affine_profile_preserves_residual_with_quantized_mla(self):
        """Quantized MLA must not undo the default BF16 residual path."""
        import copy

        import mlx.core as mx

        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model

        args = copy.deepcopy(self.model.args)
        args.moe_intermediate_size = 32
        model = Model(args)
        model.set_dtype(mx.bfloat16)
        for layer in model.layers:
            if not layer.is_linear:
                for name in ("q_a_proj", "kv_a_proj_with_mqa", "o_proj"):
                    p = getattr(layer.self_attn, name)
                    p.set_dtype(mx.float16)
                    setattr(layer.self_attn, name, p.to_quantized(group_size=32))
        for profile, dtype in (
            ("base", mx.float32),
            ("affine_fp16", mx.bfloat16),
            (envs.SGLANG_MLX_GLM_OPTIMIZATION_PROFILE.default, mx.bfloat16),
        ):
            with self.subTest(profile=profile):
                model.configure_optimizations(profile)
                model.prepare_for_inference()
                cache = model.make_cache()
                for tokens in (self.tokens[:4], self.tokens[4:5]):
                    result = model.model(mx.array([tokens]), cache)
                    mx.eval(result, [c.state for c in cache])
                    self.assertEqual(result.dtype, dtype)
                    self.assertTrue(mx.all(mx.isfinite(result)).item())
                self.assertTrue(all(c.offset == 5 for c in cache))

    def test_router_preparation_preserves_scores_and_invalidates_updates(self):
        """Prepared FP32 router must preserve routing and track weight replacement."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MoE

        reference = MoE(self.model.args)
        reference.set_dtype(mx.bfloat16)
        prepared = copy.deepcopy(reference)
        prepared.prepare_for_inference()
        x = mx.random.normal((1, 1, self.model.args.hidden_size)).astype(mx.bfloat16)
        for update in (False, True):
            if update:
                reference.gate.weight = reference.gate.weight * 1.01
                prepared.gate.weight = prepared.gate.weight * 1.01
            actual, expected = prepared(x), reference(x)
            mx.eval(actual, expected)
            self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertIsNone(prepared._router_weight)

    def test_native_prefix_snapshot_restores_kda_and_mla_together(self):
        """A prefix fork must isolate recurrent, conv, latent KV and indexer state."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            _restore_cache,
            _snapshot_cache,
        )

        # Exercise copying warmed graph owners even when this test runs alone.
        source = copy.deepcopy(self.model)
        source.configure_fixed_groups(True)
        warm_cache = source.make_cache()
        mx.eval(source(mx.array([[1, 3, 5, 7]]), warm_cache))
        model = copy.deepcopy(source)
        self.assertTrue(model.model._fixed_groups)
        self.assertTrue(
            all(not owner.graphs for owner in model.model._fixed_groups.values())
        )
        cache = model.make_cache()
        tokens = mx.array([self.tokens[:8]])
        model.forward_last_logits(tokens[:, :4], cache)
        snapshots = [_snapshot_cache(c) for c in cache]
        expected = model.forward_last_logits(tokens[:, 4:], cache)
        mx.eval(expected, [c.state for c in cache])
        for _ in range(2):
            fork = model.make_cache()
            for c, snapshot in zip(fork, snapshots):
                _restore_cache(c, snapshot)
            self.assertTrue(all(c.offset == 4 for c in fork))
            actual = model.forward_last_logits(tokens[:, 4:], fork)
            mx.eval(actual, [c.state for c in fork])
            self.assertTrue(mx.array_equal(actual, expected).item())
            self.assertTrue(all(c.offset == 8 for c in fork))

    def test_compiled_mhc_matches_reference_with_twenty_sinkhorn_rounds(self):
        """Fusing reductions must preserve stream orientation and BF16 cast points."""
        import copy

        import mlx.core as mx
        import numpy as np
        import torch
        from sglang.srt.hardware_backend.mlx.models.glm5_next import hyper_connection
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextHyperConnection,
        )

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
        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache
        from transformers.cache_utils import DynamicCache

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
                patch.object(self.model.model, "_decode_eval_group_size", group_size),
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

    def test_verification_submission_has_independent_bounds_and_cache_scope(self):
        """Verification publishes only its current layer group, including a partial tail."""
        import mlx.core as mx

        trunk = self.model.model
        decode = trunk._decode_eval_group_size
        prefill = trunk._eval_group_size
        verification = trunk._verification_eval_group_size
        baseline = self.model.make_cache()
        mx.eval(self.model(mx.array([[1, 2, 3, 4]]), baseline))
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        expected = None
        real_eval = mx.async_eval
        try:
            for group in (1, 3, 8):
                submitted = []
                group_sizes = []
                cache = [clone_cache(c) for c in baseline]
                self.model.configure_eval_groups(
                    decode=8, prefill=2, verification=group
                )

                def capture(
                    value, states, group_sizes=group_sizes, submitted=submitted
                ):
                    group_sizes.append(len(states))
                    submitted.extend(states)
                    return real_eval(value, states)

                with patch.object(mx, "async_eval", capture):
                    result = self.model.verify_mtp_rows(mx.array([[5, 6]]), cache)
                mx.eval(result[:2], [c.cache for c in cache])
                self.assertEqual(
                    group_sizes,
                    [
                        min(group, len(cache) - start)
                        for start in range(0, len(cache), group)
                    ],
                )
                self.assertEqual(len(submitted), len(cache))
                for published, state in zip(submitted, cache):
                    self.assertIs(published, state.cache)
                if expected is None:
                    expected = result[:2]
                else:
                    for actual, want in zip(result[:2], expected):
                        self.assertTrue(mx.array_equal(actual, want).item())
        finally:
            self.model.configure_eval_groups(
                decode=decode, prefill=prefill, verification=verification
            )

    def test_compiled_joint_gate_up_observes_weight_and_limit_updates(self):
        """A cached producer graph must read replacement checkpoint arrays each call."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            ClampedSwitchGLU,
            _compiled_quantized_ffn,
            _compiled_routed_gate_up,
        )

        args = copy.copy(self.model.args)
        args.hidden_size = 512
        args.moe_intermediate_size = 512
        block = ClampedSwitchGLU(args)
        block.set_dtype(mx.float16)
        for name, bits in (("gate_proj", 4), ("up_proj", 4), ("down_proj", 5)):
            setattr(
                block, name, getattr(block, name).to_quantized(bits=bits, group_size=64)
            )
        x = mx.random.normal((1, 1, 512)).astype(mx.float16)
        indices = mx.array([[[0, 1]]], mx.uint32)
        for limit in (10.0, 0.0, None):
            block.activation.limit = limit
            actual = _compiled_routed_gate_up(block, x, indices)
            expected = _compiled_quantized_ffn(block, x, indices, gate_up_only=True)
            self.assertTrue(mx.array_equal(actual, expected).item())
            block.gate_proj.scales = block.gate_proj.scales * mx.array(0.5, mx.float16)
            actual = _compiled_routed_gate_up(block, x, indices)
            expected = _compiled_quantized_ffn(block, x, indices, gate_up_only=True)
            self.assertTrue(mx.array_equal(actual, expected).item())

    def test_phase_submission_groups_preserve_partial_group_cache(self):
        """A final partial group must publish all layer states at twelve layers."""
        import mlx.core as mx

        trunk = self.model.model
        original_decode = trunk._decode_eval_group_size
        original_prefill = trunk._eval_group_size
        variants = []
        try:
            with (
                patch.object(trunk, "layers", trunk.layers[:4] * 3),
                patch.object(trunk.args, "num_hidden_layers", 12),
            ):
                for group in (1, 4, 8):
                    self.model.configure_eval_groups(decode=group, prefill=4)
                    cache = self.model.make_cache()
                    outputs = []
                    for inputs in ([[1, 2, 3, 4]], [[5]], [[6]]):
                        outputs.append(self.model(mx.array(inputs), cache))
                        mx.eval(outputs[-1], [c.cache for c in cache])
                    variants.append((outputs, cache))
                for outputs, cache in variants[1:]:
                    for actual, expected in zip(outputs, variants[0][0]):
                        self.assertTrue(mx.array_equal(actual, expected).item())
                    for actual, expected in zip(cache, variants[0][1]):
                        self.assertEqual(actual.offset, 6)
                        for i in range(4):
                            if actual[i] is None:
                                self.assertIsNone(expected[i])
                            else:
                                self.assertTrue(
                                    mx.array_equal(actual[i], expected[i]).item()
                                )
        finally:
            self.model.configure_eval_groups(
                decode=original_decode, prefill=original_prefill
            )

    def test_kda_qkv_fusion_preserves_quantization_and_canonical_weights(self):
        """Load-time fusion must preserve mixed-bit fallback, state and save format."""
        import copy

        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        for policy in ("fp32", "bf16", "q4", "q4_all", "mixed"):
            with self.subTest(policy=policy):
                reference = copy.deepcopy(self.model.layers[0].self_attn)
                if policy == "bf16":
                    reference.set_dtype(mx.bfloat16)
                elif policy in ("q4", "q4_all", "mixed"):
                    names = ("q_proj", "k_proj", "v_proj")
                    if policy == "q4_all":
                        names += ("f_a_proj", "g_a_proj", "b_proj")
                    for name in names:
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
                if policy != "mixed":
                    self.assertEqual(
                        len(prepared._fused_input_names),
                        6 if policy in ("fp32", "bf16", "q4_all") else 3,
                    )
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
                if policy == "q4_all":
                    prepared.g_a_proj.scales = prepared.g_a_proj.scales * 1.01
                    reference.g_a_proj.scales = reference.g_a_proj.scales * 1.01
                elif policy in ("q4", "mixed"):
                    prepared.q_proj.scales = prepared.q_proj.scales * 1.01
                    reference.q_proj.scales = reference.q_proj.scales * 1.01
                elif policy == "bf16":
                    prepared.g_a_proj.weight = prepared.g_a_proj.weight + 0.01
                    reference.g_a_proj.weight = reference.g_a_proj.weight + 0.01
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
        """Canonical inputs share fused storage, with bounded alignment overhead."""
        import copy
        import gc

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import KDA

        args = copy.deepcopy(self.model.args)
        args.hidden_size = 1024
        args.linear_attn_config = dict(args.linear_attn_config, head_dim=128)
        attention = KDA(args)
        mx.eval(attention.parameters())
        gc.collect()
        before = mx.get_active_memory()
        self.assertTrue(attention.prepare_for_inference())
        mx.eval(attention.parameters())
        gc.collect()
        # Inputs occupy multiple MiB; a second resident copy exceeds this
        # allowance for alignment/materialization of unequal-width slices.
        self.assertLessEqual(mx.get_active_memory() - before, 16384)

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

    def _mtp_model(self):
        import copy

        import mlx.core as mx
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model

        args = copy.copy(self.model.args)
        args.num_nextn_predict_layers = 1
        mx.random.seed(71)
        model = Model(args, enable_mtp=True)
        model.load_weights(tree_flatten(self.model.parameters()), strict=False)
        return model

    def test_mtp_head_matches_plain_reference_equations(self):
        """MTP is plain MLA and concatenates embedding before hidden."""
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        model = self._mtp_model()
        # Reuse independently loaded HF MLA/MoE weights; discard the trunk's
        # hyper-connection tensors, which do not belong to the MTP decoder.
        source = self.model.layers[3]
        model.mtp.load_weights(
            [
                (k, v)
                for k, v in tree_flatten(source.parameters())
                if not k.startswith("hc_")
            ],
            strict=False,
        )
        rng = np.random.default_rng(18)
        for norm in (model.mtp.enorm, model.mtp.hnorm, model.mtp.shared_head.norm):
            norm.weight = mx.array(
                rng.uniform(0.5, 1.5, self.config.hidden_size).astype("float32")
            )
        tokens = mx.array([[3, 5, 7]])
        hidden = mx.array(
            rng.normal(size=(1, 3, self.config.hidden_size)).astype("float32")
        )
        logits, raw = model.forward_mtp(tokens, hidden, NativeCache(4))
        layer = self.reference.layers[3]

        def tensor(value):
            return torch.from_numpy(np.array(value).copy())

        def norm(value, weight):
            return torch.nn.functional.rms_norm(
                value,
                (self.config.hidden_size,),
                tensor(weight),
                self.config.rms_norm_eps,
            )

        with torch.no_grad():
            e = self.reference.embed_tokens(tensor(tokens).long())
            joined = torch.cat(
                [
                    norm(e, model.mtp.enorm.weight),
                    norm(tensor(hidden), model.mtp.hnorm.weight),
                ],
                -1,
            )
            x = joined @ tensor(model.mtp.eh_proj.weight).T
            attn, _, _ = layer.self_attn(
                layer.input_layernorm(x),
                attention_mask=torch.ones((1, 3), dtype=torch.bool),
            )
            x = x + attn
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            expected = (
                norm(x, model.mtp.shared_head.norm.weight)
                @ tensor(model.lm_head.weight).T
            )
        np.testing.assert_allclose(np.array(raw), x.numpy(), atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(
            np.array(logits), expected.numpy(), atol=2e-5, rtol=2e-4
        )

    def test_mtp_verify_preserves_both_cache_commit_paths(self):
        """Rejection restores conv/state and indexer tails across pool boundaries."""
        import copy

        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model

        args = copy.copy(self.model.args)
        args.num_nextn_predict_layers = 1
        args.hc_mult = 4
        args.moe_intermediate_size = 64
        mx.random.seed(72)
        model = Model(args, enable_mtp=True)
        model.set_dtype(mx.bfloat16)
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, module: (
                {"group_size": 32, "bits": 5 if "down_proj" in path else 6}
                if ".mlp." in path and hasattr(module, "to_quantized")
                else False
            ),
        )
        # Match the production checkpoint's BF16 residual / FP16 affine metadata.
        model.load_weights(
            [
                (k, v.astype(mx.float16))
                for k, v in tree_flatten(model.parameters())
                if k.endswith((".scales", ".biases"))
            ],
            strict=False,
        )
        model.configure_optimizations("affine_fused")
        for length in (3, 4, 9, 10, 13):
            with self.subTest(prefix_length=length):
                serial, block = model.make_cache(), model.make_cache()
                prompt = mx.array([self.tokens[:length]])
                mx.eval(model(prompt, serial), model(prompt, block))
                root, draft = mx.array([[17]]), mx.array([[19]])
                first, h1 = model.forward_with_hidden(root, serial)
                root_expected = model.make_cache()
                for old, new in zip(serial, root_expected):
                    new.cache = [
                        mx.array(v) if v is not None else None for v in old.cache
                    ]
                    new.offset = old.offset
                second, h2 = model.forward_with_hidden(draft, serial)
                logits, hidden, root_cache = model.verify_mtp(root, draft, block)
                np.testing.assert_array_equal(
                    np.array(logits.astype(mx.float32)),
                    np.array(mx.concatenate([first, second], 1).astype(mx.float32)),
                )
                np.testing.assert_array_equal(
                    np.array(hidden.astype(mx.float32)),
                    np.array(mx.concatenate([h1, h2], 1).astype(mx.float32)),
                )
                for actual, expected in ((block, serial), (root_cache, root_expected)):
                    for a, b in zip(actual, expected):
                        self.assertEqual(a.offset, b.offset)
                        for av, bv in zip(a.cache, b.cache):
                            if av is None or bv is None:
                                self.assertIs(av, bv)
                            else:
                                np.testing.assert_array_equal(
                                    np.array(av.astype(mx.float32)),
                                    np.array(bv.astype(mx.float32)),
                                )
                    np.testing.assert_array_equal(
                        np.array(model(mx.array([[23]]), actual).astype(mx.float32)),
                        np.array(model(mx.array([[23]]), expected).astype(mx.float32)),
                    )

    def test_kda_shared_projection_preserves_each_recurrent_snapshot(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import KDA, NativeCache

        args = copy.copy(self.model.args)
        args.hidden_size = 256
        args.linear_attn_config = dict(
            num_heads=2, head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0
        )
        mx.random.seed(73)
        attn = KDA(args)
        attn.set_dtype(mx.bfloat16)
        attn.A_log = attn.A_log.astype(mx.float32)
        attn.dt_bias = attn.dt_bias.astype(mx.float32)
        attn.fuse_decode = True
        attn.prepare_for_inference()
        for count in (2, 3, 9, 17):
            x = mx.random.normal((1, count, 256)).astype(mx.bfloat16)
            projected = attn.project_verify_rows(x)
            self.assertIsNotNone(projected)
            serial, shared = NativeCache(4), NativeCache(4)
            for j in range(count):
                row = x[:, j : j + 1]
                expected = attn(row, serial)
                actual = attn(
                    row, shared, projected=tuple(v[:, j : j + 1] for v in projected)
                )
                mx.eval(actual, expected, serial.cache, shared.cache)
                self.assertTrue(mx.array_equal(actual, expected).item())
                self.assertEqual(serial.offset, shared.offset)
                for a, b in zip(serial.cache, shared.cache):
                    self.assertTrue(mx.array_equal(a, b).item())

        attn.q_proj.weight = attn.q_proj.weight * mx.array(0.5, mx.bfloat16)
        self.assertIsNone(attn.project_verify_rows(x))
        attn.prepare_for_inference()
        projected = attn.project_verify_rows(x)
        self.assertIsNotNone(projected)
        serial, shared = NativeCache(4), NativeCache(4)
        for j in range(x.shape[1]):
            expected = attn(x[:, j : j + 1], serial)
            actual = attn(
                x[:, j : j + 1],
                shared,
                projected=tuple(v[:, j : j + 1] for v in projected),
            )
            self.assertTrue(mx.array_equal(actual, expected).item())

    def test_mtp_batched_ffn_preserves_two_row_commit_boundaries(self):
        """Batching must retain snapshots usable after rejection or full reuse."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._mtp_model()
        base = model.make_cache()
        mx.eval(model(mx.array([self.tokens]), base))
        chain = mx.array([[17, 19]])
        model.configure_mtp_verification("serial")
        expected = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
        mx.eval(expected[:2])
        for mode in ("head", "ffn"):
            with self.subTest(mode=mode):
                model.configure_mtp_verification(mode)
                actual = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
                mx.eval(actual[:2])
                for layer in model.layers:
                    self.assertNotIn("batch_fusion", layer.mlp)
                for got, want in zip(actual[:2], expected[:2]):
                    self.assertLess(mx.max(mx.abs(got - want)).item(), 0.04)
                for depth in (0, 1):
                    got = [clone_cache(c) for c in actual[2][depth]]
                    want = [clone_cache(c) for c in expected[2][depth]]
                    self.assertTrue(
                        all(c.offset == len(self.tokens) + depth + 1 for c in got)
                    )
                    token = mx.array([[23]])
                    got_logits, want_logits = model(token, got), model(token, want)
                    mx.eval(got_logits, want_logits)
                    self.assertLess(
                        mx.max(mx.abs(got_logits - want_logits)).item(), 0.04
                    )

    def test_f_parallel_verify_causal_split_and_rejected_prefix(self):

        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            MLACache,
            ModelArgs,
            _FixedMLAAttention,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        mx.random.seed(954)
        args = ModelArgs(
            model_type="glm5_next",
            text_config={
                "hidden_size": 512,
                "rms_norm_eps": 1e-5,
                "attention_bias": False,
                "num_attention_heads": 64,
                "q_lora_rank": 512,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 0,
                "v_head_dim": 128,
                "index_head_dim": 128,
                "index_n_heads": 4,
                "index_topk": 2048,
                "index_kpool": 16,
                "index_kpool_always_select_tail": True,
            },
        )

        class CompactingMLA(MLA):
            def attend_rows(self, *inputs, indices_valid=False, **kwargs):
                return super().attend_rows(*inputs, indices_valid=False, **kwargs)

        attn = MLA(args)
        attn.set_dtype(mx.bfloat16)
        compacting = CompactingMLA(args)
        compacting.update(attn.parameters())
        for offset in (31, 255, 256, 257, 2047, 2048, 2049):
            base = MLACache(4)
            # Populate real index/cache state using original AR/prefill.
            mx.eval(
                attn(mx.random.normal((1, offset, 512)).astype(mx.bfloat16), base),
                base.state,
            )
            x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
            for compiled in (False, True):
                reference, rs = compacting.verify_rows(x, clone_cache(base))
                graph = _FixedMLAAttention(attn) if compiled else None
                actual, cs = attn.verify_rows(
                    x,
                    clone_cache(base),
                    attention_graph=graph,
                )
                self.assertTrue(
                    mx.array_equal(reference, actual).item(),
                    (offset, compiled, "outputs"),
                )
                for left, right in zip(rs, cs):
                    self.assertEqual(left.offset, right.offset)
                    self.assertEqual(len(left.cache), len(right.cache))
                    for a, b in zip(left.cache, right.cache):
                        if a is None or b is None:
                            self.assertIs(a, b)
                        else:
                            self.assertTrue(mx.array_equal(a, b).item())
                for consumed in (1, 2):
                    follow = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
                    expected = attn(follow, clone_cache(rs[consumed - 1]))
                    got = attn(follow, clone_cache(cs[consumed - 1]))
                    self.assertTrue(
                        mx.array_equal(expected, got).item(),
                        (offset, consumed, "resume"),
                    )
        # Poison future keys AND capacity region. Input causal metadata is
        # distinct from selected lengths, including a deliberately future id.
        q = mx.random.normal((1, 3, 64 * 128)).astype(mx.bfloat16)
        kv = mx.random.normal((1, 513, 512)).astype(mx.bfloat16)
        ends = mx.array([511, 512, 513], mx.uint32)
        ids = mx.broadcast_to(mx.arange(513, dtype=mx.int32), (1, 3, 513))
        lengths = mx.array([513, 513, 513], mx.uint32)
        ref = attn.attend_rows(q, kv, None, ids, lengths, visible_ends=ends)
        prefix = mx.pad(
            kv[:, :256], [(0, 0), (0, 256), (0, 0)], constant_values=float("nan")
        )
        tail = mx.pad(
            kv[:, 256:], [(0, 0), (0, 255), (0, 0)], constant_values=float("nan")
        )
        got = attn.attend_rows(
            q,
            prefix,
            tail,
            ids,
            lengths,
            split=mx.array(256, mx.uint32),
            visible_ends=ends,
        )
        self.assertTrue(mx.array_equal(ref, got).item())

    def test_mla_verify_rows_match_legal_prefixes_and_index_pool_growth(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLA, NativeCache
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        args = copy.copy(self.model.args)
        mx.random.seed(29)
        attn = MLA(args)
        for prefix in (0, 3, 7, 8, 9, 255, 259):
            base = NativeCache(4)
            if prefix:
                mx.eval(
                    attn(mx.random.normal((1, prefix, args.hidden_size)), base),
                    base.cache,
                )
            for count in (2, 3, 5, 9):
                x = mx.random.normal((1, count, args.hidden_size))
                serial, shared = clone_cache(base), clone_cache(base)
                rows, states = [], []
                for j in range(count):
                    rows.append(attn(x[:, j : j + 1], serial))
                    states.append(clone_cache(serial))
                actual, snapshots = attn.verify_rows(x, shared)
                self.assertLess(
                    mx.max(mx.abs(actual - mx.concatenate(rows, 1))).item(), 2e-5
                )
                for got, want in zip(snapshots, states):
                    self.assertEqual(got.offset, want.offset)
                    for a, b in zip(got.cache, want.cache):
                        if a is None or b is None:
                            self.assertIs(a, b)
                        else:
                            self.assertTrue(mx.array_equal(a, b).item())
                    continuation = mx.random.normal((1, 1, args.hidden_size))
                    self.assertLess(
                        mx.max(
                            mx.abs(
                                attn(continuation, clone_cache(got))
                                - attn(continuation, clone_cache(want))
                            )
                        ).item(),
                        2e-5,
                    )

    def test_mla_verify_empty_sparse_rows_are_zero_and_causal(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLA, NativeCache

        args = copy.copy(self.model.args)
        args.index_kpool_always_select_tail = False
        attn = MLA(args)
        x = mx.random.normal((1, 5, args.hidden_size))
        serial = NativeCache(4)
        expected = mx.concatenate([attn(x[:, j : j + 1], serial) for j in range(5)], 1)
        actual, _ = attn.verify_rows(x, NativeCache(4))
        self.assertTrue(mx.all(mx.isfinite(actual)).item())
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 2e-5)

    def _optimized_mtp_model(self, *, attention_heads=2, latent_rank=256):
        """Covered GLM geometry, genuine BF16 mHC and mixed affine experts."""
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            Model,
            ModelArgs,
            MoE,
        )
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )

        mx.random.seed(83)
        config = Glm5NextTextConfig(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=attention_heads,
            num_key_value_heads=attention_heads,
            n_routed_experts=8,
            num_experts_per_tok=8,
            n_shared_experts=1,
            q_lora_rank=128,
            kv_lora_rank=latent_rank,
            qk_nope_head_dim=128,
            qk_rope_head_dim=0,
            v_head_dim=128,
            index_head_dim=32,
            index_n_heads=2,
            index_topk=32,
            index_kpool=4,
            linear_num_heads=2,
            linear_head_dim=128,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            pad_token_id=None,
        ).to_dict()
        config.update(mhc=True, scoring_func="sigmoid", num_nextn_predict_layers=1)
        config["linear_attn_config"] = dict(
            num_heads=2, head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0
        )
        model = Model(
            ModelArgs(model_type="glm5_next", text_config=config), enable_mtp=True
        )
        model.set_dtype(mx.bfloat16)
        for layer in model.layers:
            for name in ("attn", "ffn"):
                layer[f"hc_{name}_fn"] = (mx.random.normal((24, 1024)) * 0.02).astype(
                    mx.bfloat16
                )
                layer[f"hc_{name}_base"] = layer[f"hc_{name}_base"].astype(mx.float32)
                layer[f"hc_{name}_scale"] = layer[f"hc_{name}_scale"].astype(mx.float32)
        for layer in model.model.layers:
            if isinstance(layer.mlp, MoE):
                for name, bits in (("gate_proj", 4), ("up_proj", 4), ("down_proj", 5)):
                    projection = layer.mlp.switch_mlp[name]
                    projection.set_dtype(mx.float16)
                    layer.mlp.switch_mlp[name] = projection.to_quantized(
                        group_size=64, bits=bits
                    )
        model.configure_optimizations("affine_fused")
        model.configure_mtp_verification("ffn")
        model.prepare_for_inference()
        mx.eval(model.parameters())
        return model

    def _assert_optimized_cache_equal(self, actual, expected):
        import mlx.core as mx

        for got, want in zip(actual, expected):
            self.assertEqual(got.offset, want.offset)
            for slot, (a, b) in enumerate(zip(got.cache, want.cache)):
                if hasattr(got, "latent_segments") and slot == 0:
                    a, b = got[0], want[0]
                if hasattr(got, "latent_segments") and slot == 4:
                    continue
                if a is None or b is None:
                    self.assertIs(a, b)
                else:
                    self.assertTrue(
                        mx.array_equal(a, b).item(), (slot, a.shape, b.shape)
                    )

    def test_graph_observation_rebuilds_mutated_structure_and_parameters(self):
        import copy

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.models.glm5_next import _GraphObservation

        module = nn.Module()
        module.projection = nn.Linear(256, 256)
        module.norms = [nn.RMSNorm(256)]
        module.options = {"limit": 1.0}
        plan = _GraphObservation(module)
        for change in (
            lambda: setattr(module.norms[0], "eps", 0.1),
            lambda: module.update({"projection": {"weight": module.projection.weight * 1.25}}),
            lambda: setattr(module, "projection", nn.Linear(256, 256)),
            lambda: module.norms.append(nn.RMSNorm(256)),
            lambda: module.options.update(extra=2),
            lambda: module.options.pop("limit"),
        ):
            self.assertTrue(plan.current(module))
            change()
            self.assertFalse(plan.current(module))
            plan = _GraphObservation(module)
            for (_, actual), (_, expected) in zip(
                tree_flatten(plan.parameters), tree_flatten(module.parameters())
            ):
                self.assertIs(actual, expected)
        with patch.object(nn.Linear, "compile_observation_flag", True, create=True):
            self.assertFalse(plan.current(module))
            plan = _GraphObservation(module)
            self.assertTrue(plan.current(module))
        self.assertFalse(plan.current(module))
        self.assertFalse(plan.current(copy.deepcopy(module)))

    def test_fixed_groups_refresh_replaced_layer_container(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache))
        tokens = mx.array([[11, 17, 23]])
        mx.eval(model.verify_mtp_rows(tokens, [clone_cache(c) for c in cache])[:2])
        replacement = copy.deepcopy(model.layers[0])
        replacement.post_attention_layernorm.eps = 0.1
        model.model.layers[0] = replacement
        actual = model.verify_mtp_rows(tokens, [clone_cache(c) for c in cache])
        model.configure_fixed_groups(False)
        expected = model.verify_mtp_rows(tokens, [clone_cache(c) for c in cache])
        self.assertTrue(mx.array_equal(actual[0], expected[0]).item())
        for got, want in zip(actual[2], expected[2]):
            self._assert_optimized_cache_equal(got, want)

    def test_token_only_serving_switches_contract_at_consumption_boundary(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
        from sglang.srt.hardware_backend.mlx.sampling import (
            GREEDY_PARAMS,
            MlxSamplingParams,
        )

        model = self._optimized_mtp_model()
        runner = object.__new__(MlxModelRunner)
        runner.model = model
        runner._enable_sampling = True
        runner._sanitize_nan = False
        runner._req_sampling = {"a": GREEDY_PARAMS}
        self.assertTrue(runner._token_only_allowed(["a"]))
        self.assertFalse(runner._token_only_allowed(["a"], logits_hook=lambda x: x))
        self.assertFalse(runner._token_only_allowed(["a"], logprob_spec=object()))
        self.assertFalse(runner._token_only_allowed(["a"], edit_rows=mx.zeros((1, 128))))
        runner._sanitize_nan = True
        self.assertFalse(runner._token_only_allowed(["a"]))
        runner._sanitize_nan = False
        runner._req_sampling["a"] = MlxSamplingParams(1, 4, 1, 0, None)
        self.assertFalse(runner._token_only_allowed(["a"]))
        for steps in (0, 2, 4):
            actual, reference = MlxMTPServing(model, steps), MlxMTPServing(model, steps)
            caches = [model.make_cache(), model.make_cache()]
            prompt = mx.array([[1, 2, 3, 4]])
            hidden = actual.prefill("a", prompt, caches[0])
            reference.prefill("a", prompt, caches[1])
            token = int(mx.argmax(model.lm_head(hidden[:, -1:])).item())
            for iteration in range(8):
                inputs = mx.array([[token]], mx.int32)
                expected = reference.decode_batch(["a"], inputs, [caches[1]], [token])
                token_only = iteration % 3 != 2
                output = actual.decode_batch(
                    ["a"], inputs, [caches[0]], [token], token_only=token_only
                )
                choice = output.reshape(-1) if token_only else mx.argmax(output, -1)
                expected_choice = mx.argmax(expected, -1)
                self.assertTrue(mx.array_equal(choice, expected_choice).item())
                self._assert_optimized_cache_equal(caches[0], caches[1])
                token = int(choice.item()) if iteration != 3 else 19
            actual.remove("a")
            stats = actual.stats()
            self.assertEqual(sum(stats["committed_tokens_by_budget"].values()), 8)
            self.assertTrue(stats["elapsed_seconds_by_budget"])

    def test_token_only_target_retains_logits_and_cache_contract(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model(attention_heads=64, latent_rank=512)
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4]]), cache))
        tokens = mx.array([[11, 17, 23]])
        logits, hidden, states = model.verify_mtp_rows(tokens, [clone_cache(c) for c in cache])
        actual, actual_hidden, actual_states = model.verify_mtp_rows(
            tokens, [clone_cache(c) for c in cache], token_only=True
        )
        self.assertTrue(mx.array_equal(actual, mx.argmax(logits, -1)).item())
        self.assertTrue(mx.array_equal(hidden, actual_hidden).item())
        for got, want in zip(actual_states, states):
            self._assert_optimized_cache_equal(got, want)
        first = [clone_cache(c) for c in cache]
        second = [clone_cache(c) for c in cache]
        predicted = model.forward_last_tokens(tokens[:, :1], first)
        expected = mx.argmax(model.forward_last_logits(tokens[:, :1], second), -1)
        self.assertTrue(mx.array_equal(predicted, expected).item())
        self._assert_optimized_cache_equal(first, second)

    def test_optimized_fixed_graphs_refresh_scalar_equations(self):
        """Compiled owners must observe changed FFN limits, norm and dispatch metadata."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLP
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model(attention_heads=64, latent_rank=512)
        base = model.make_cache()
        mx.eval(model((mx.arange(26) % 128)[None], base))
        tokens = mx.array([[11, 17, 23]])
        dense = MLP(model.args)
        dense.set_dtype(mx.bfloat16)
        model.layers[0].mlp = dense
        for change in ("limits", "flags", "norm", "quantization", "attention"):
            model.configure_fixed_groups(True)
            mx.eval(model.verify_mtp_rows(tokens, [clone_cache(c) for c in base])[:2])
            if change == "limits":
                model.layers[0].mlp.limit = 0.001
                for layer in model.layers[1:]:
                    if hasattr(layer.mlp, "switch_mlp"):
                        layer.mlp.switch_mlp.activation.limit = 0.001
                        layer.mlp.shared_experts.limit = 0.001
                    else:
                        layer.mlp.limit = 0.001
            elif change == "flags":
                for layer in model.layers:
                    layer.mlp.compile_activation = not getattr(
                        layer.mlp, "compile_activation", False
                    )
                    if hasattr(layer.mlp, "switch_mlp"):
                        layer.mlp.switch_mlp.activation.compile_activation = False
            elif change == "norm":
                for layer in model.layers:
                    layer.input_layernorm.eps = 0.1
                    layer.post_attention_layernorm.eps = 0.1
            elif change == "quantization":
                # Same packed weights, valid new grouping with matching metadata arrays.
                for layer in model.layers[1:]:
                    if not hasattr(layer.mlp, "switch_mlp"):
                        continue
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        projection = layer.mlp.switch_mlp[name]
                        projection.group_size = 32
                        projection.scales = mx.repeat(projection.scales, 2, -1)
                        projection.biases = mx.repeat(projection.biases, 2, -1)
            else:
                for layer in model.layers:
                    if layer.is_linear:
                        layer.self_attn.lower_bound = -2.0
                    else:
                        layer.self_attn.q_a_layernorm.eps = 0.1
            actual = model.verify_mtp_rows(tokens, [clone_cache(c) for c in base])
            mx.eval(actual[:2])
            model.configure_fixed_groups(False)
            expected = model.verify_mtp_rows(tokens, [clone_cache(c) for c in base])
            mx.eval(expected[:2])
            for a, b in zip(actual[:2], expected[:2]):
                self.assertTrue(mx.array_equal(a, b).item(), change)
            for got, want in zip(actual[2], expected[2]):
                self._assert_optimized_cache_equal(got, want)

    def test_optimized_mla_fixed_width_includes_pool_tails(self):
        """topk plus unfinished pools may exceed 4096; all snapshots stay exact."""
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            MLACache,
            _FixedMLAAttention,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model(attention_heads=64, latent_rank=512)
        for topk in (2048, 4096):
            args = copy.copy(model.args)
            args.index_topk = topk
            args.index_kpool_always_select_tail = True
            attention = MLA(args)
            attention.set_dtype(mx.bfloat16)
            owner = _FixedMLAAttention(attention)
            for offset in (4093, 4096, 4099):
                cache = MLACache(4)
                prefix = mx.random.normal((1, offset, 256)).astype(mx.bfloat16)
                qr = attention.q_a_layernorm(attention.q_a_proj(prefix))
                attention.indexer(prefix, qr, cache, 0)
                cache.append_latent(
                    mx.random.normal((1, offset, 512)).astype(mx.bfloat16)
                )
                cache.offset = offset
                mx.eval(cache.cache)
                rows = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
                actual = attention.verify_rows(
                    rows, clone_cache(cache), attention_graph=owner
                )
                expected = attention.verify_rows(rows, clone_cache(cache))
                self.assertTrue(
                    mx.array_equal(actual[0], expected[0]).item(), (topk, offset)
                )
                self._assert_optimized_cache_equal(actual[1], expected[1])

    def test_fixed_graphs_reuse_alternating_verification_depths(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model(attention_heads=64, latent_rank=512)
        base = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4]]), base))
        reference = copy.deepcopy(model)
        reference.configure_fixed_groups(False)
        owners = []
        for layer in model.layers:
            owner = getattr(layer, "_fixed_mla_layer", None)
            if owner is not None:
                owners.extend((owner, owner.attention))
        first = None
        for repeat in range(3):
            for count in (3, 2):
                tokens = mx.array([list(range(11, 11 + count))])
                actual = model.verify_mtp_rows(tokens, [clone_cache(c) for c in base])
                mx.eval(actual[:2])
                if repeat == 0 and count == 3:
                    first = [list(owner.graphs.values())[-1] for owner in owners]
                elif count == 3:
                    for owner, graph in zip(owners, first):
                        self.assertTrue(
                            any(value is graph for value in owner.graphs.values())
                        )
                expected = reference.verify_mtp_rows(
                    tokens, [clone_cache(c) for c in base]
                )
                mx.eval(expected[:2])
                for got, want in zip(actual[:2], expected[:2]):
                    self.assertTrue(mx.array_equal(got, want).item())
                for got, want in zip(actual[2], expected[2]):
                    self._assert_optimized_cache_equal(got, want)
        self.assertTrue(owners)
        self.assertTrue(all(len(owner.graphs) <= 8 for owner in owners))

    def test_optimized_fixed_groups_invalidate_weights_and_flags(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        base = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4]]), base))
        for update in ("weight", "flag"):
            model.configure_fixed_groups(True)
            mx.eval(model(mx.array([[5]]), [clone_cache(c) for c in base]))
            before = [
                g
                for owner in model.model._fixed_groups.values()
                for g in owner.graphs.values()
            ]
            if update == "weight":
                model.layers[0].self_attn.q_proj.weight = (
                    model.layers[0].self_attn.q_proj.weight * 1.125
                )
            else:
                model.layers[0].compile_residual = not model.layers[0].compile_residual
            actual_cache = [clone_cache(c) for c in base]
            actual = model(mx.array([[5]]), actual_cache)
            after = [
                g
                for owner in model.model._fixed_groups.values()
                for g in owner.graphs.values()
            ]
            self.assertIsNot(before[0], after[0])
            model.configure_fixed_groups(False)
            expected_cache = [clone_cache(c) for c in base]
            expected = model(mx.array([[5]]), expected_cache)
            self.assertTrue(mx.array_equal(actual, expected).item())
            self._assert_optimized_cache_equal(actual_cache, expected_cache)

    def test_optimized_sorted_prefill_matches_expert_reference(self):
        import mlx.core as mx
        from mlx_lm.models.switch_layers import SwitchGLU

        model = self._optimized_mtp_model()
        block = next(
            layer.mlp.switch_mlp
            for layer in model.layers
            if hasattr(layer.mlp, "switch_mlp")
        )
        self.assertIsNotNone(block._fused_gate_up)
        for count in (8, 32, 65, 256):
            x = mx.random.normal((1, count, 256)).astype(mx.float16)
            indices = mx.random.randint(0, 8, (1, count, 8))
            expected = SwitchGLU.__call__(block, x, indices)
            actual = block(x, indices)
            self.assertTrue(mx.array_equal(actual, expected).item(), count)

    def test_optimized_deferred_prefix_accept_reject_and_release(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import (
            MlxMTPServing,
            clone_cache,
        )

        model = self._optimized_mtp_model()
        for depth in range(1, 5):
            engine = MlxMTPServing(model, 4)
            cache = model.make_cache()
            mx.eval(engine.prefill("owner", mx.array([[1, 3, 5, 7]]), cache))
            engine.decode("owner", mx.array([[11]]), cache)
            for _ in range(depth):
                token = engine.states["owner"].rows[0].input_token
                engine.decode("owner", mx.array([[token.item()]]), cache)
            reference = [clone_cache(c) for c in cache]
            reference[0].mlx_mtp = cache[0].mlx_mtp
            engine.remove("owner")
            resumed = MlxMTPServing(model, 4)
            for token in (23, 24, 25):
                inputs = mx.array([[token]])
                actual = engine.decode("owner", inputs, cache)
                expected = resumed.decode("fork", inputs, reference)
                self.assertTrue(mx.array_equal(actual, expected).item())
                self._assert_optimized_cache_equal(cache, reference)
            engine.remove("owner")
            resumed.remove("fork")
            self.assertFalse(engine.states)
            self.assertFalse(resumed.states)

    def test_optimized_mtp_request_trunks_preserve_ragged_snapshots(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        owners = [model.make_cache() for _ in range(3)]
        for j, cache in enumerate(owners):
            mx.eval(model(mx.array([list(range(1, 5 + j))]), cache))
        for counts in ((3, 3, 3), (2, 5, 9), (1, 2, 17)):
            chains = [mx.array([list(range(11, 11 + count))]) for count in counts]
            reference = [[clone_cache(c) for c in owner] for owner in owners]
            expected = [
                model.verify_mtp_rows(chain, cache)
                for chain, cache in zip(chains, reference)
            ]
            actual_cache = [[clone_cache(c) for c in owner] for owner in owners]
            actual = model.verify_mtp_requests(chains, actual_cache)
            for got, want in zip(actual, expected):
                for a, b in zip(got[:2], want[:2]):
                    self.assertTrue(mx.array_equal(a, b).item(), counts)
                for a, b in zip(got[2], want[2]):
                    self._assert_optimized_cache_equal(a, b)
            for a, b in zip(actual_cache, reference):
                self._assert_optimized_cache_equal(a, b)

    def test_fixed_mla_preserves_prepared_router_precision(self):
        """The graph must retain the FP32 router when canonical weights are BF16."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models import glm5_next as module
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        layer = model.layers[3]
        self.assertEqual(layer.mlp.gate.weight.dtype, mx.bfloat16)
        self.assertEqual(layer.mlp._router_weight.dtype, mx.float32)
        cache = model.make_cache()[3]
        x = mx.random.normal((1, 3, 4, 256)).astype(mx.bfloat16)
        with (
            patch.object(layer, "_fixed_mla_layer", None),
            patch.object(layer, "_fixed_mla_attention", None),
        ):
            expected, expected_states = layer.verify_ffn_rows(x, clone_cache(cache))

        operands = []
        original = module.mlx_glm_router

        def observe(value, weight, *args, **kwargs):
            operands.append(weight.dtype)
            return original(value, weight, *args, **kwargs)

        with patch.object(module, "mlx_glm_router", observe):
            actual, states = module._FixedMLALayer(layer)(x, clone_cache(cache))
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertTrue(operands)
        self.assertTrue(all(dtype == mx.float32 for dtype in operands))
        for got, want in zip(states, expected_states):
            self._assert_optimized_cache_equal([got], [want])

    def test_optimized_fixed_verification_returns_every_depth(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        base = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4]]), base))
        for count in (2, 5, 9, 17):
            chain = mx.array([list(range(11, 11 + count))])
            a, b = ([clone_cache(c) for c in base], [clone_cache(c) for c in base])
            model.configure_fixed_groups(False)
            expected = model.verify_mtp_rows(chain, a)
            model.configure_fixed_groups(True)
            actual = model.verify_mtp_rows(chain, b)
            for x, y in zip(actual[:2], expected[:2]):
                self.assertTrue(mx.array_equal(x, y).item())
            for x, y in zip(actual[2], expected[2]):
                self._assert_optimized_cache_equal(x, y)
            self._assert_optimized_cache_equal(a, b)

    def test_consumed_head_rows_preserve_singleton_cache_boundaries(self):
        """Deferred accepted rows must retain head arithmetic across tile ends."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        model = self._optimized_mtp_model()
        for count in (2, 5, 9, 17):
            hidden = mx.random.normal((1, count, model.args.hidden_size)).astype(
                mx.bfloat16
            )
            embedded = mx.random.normal(hidden.shape).astype(mx.bfloat16)
            serial, grouped = NativeCache(4), NativeCache(4)
            expected = mx.concatenate(
                [
                    model.mtp(hidden[:, j : j + 1], embedded[:, j : j + 1], serial)
                    for j in range(count)
                ],
                1,
            )
            actual = model.mtp.consume_rows(hidden, embedded, grouped)
            self.assertTrue(mx.array_equal(actual, expected).item())
            self.assertEqual(serial.offset, grouped.offset)
            for got, want in zip(grouped.cache, serial.cache):
                if got is None or want is None:
                    self.assertIs(got, want)
                else:
                    self.assertTrue(mx.array_equal(got, want).item())

    def test_optimized_mla_graph_uses_runtime_visibility(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            _FixedMLAAttention,
        )

        model = self._optimized_mtp_model()
        args = copy.copy(model.args)
        args.num_attention_heads = 64
        args.kv_lora_rank = 512
        attention = MLA(args)
        attention.set_dtype(mx.bfloat16)
        graph = _FixedMLAAttention(attention)
        q = mx.random.normal((1, 3, 64 * 128)).astype(mx.bfloat16)
        for length in (255, 256, 257, 511, 512, 513):
            kv = mx.random.normal((1, length, 512)).astype(mx.bfloat16)
            split = length - 17
            table = mx.broadcast_to(mx.arange(32), (1, 3, 32))
            lengths = [17, 26, 31]
            for row, visible in enumerate(lengths):
                table[:, row, visible:] = -9999
            expected = attention.attend_rows(
                q, kv[:, :split], kv[:, split:], table, lengths
            )
            actual = graph(
                q, kv[:, :split], kv[:, split:], table, mx.array(lengths, mx.uint32)
            )
            self.assertTrue(mx.array_equal(actual, expected).item())

    def test_optimized_mla_fixed_model_preserves_growth_and_snapshots(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model(attention_heads=64, latent_rank=512)
        for context in (26, 255, 256, 257, 511, 512, 513):
            model.configure_fixed_groups(False)
            base = model.make_cache()
            mx.eval(model((mx.arange(context) % 128)[None], base))
            for count in (1, 3, 5):
                a = [clone_cache(c) for c in base]
                b = [clone_cache(c) for c in base]
                tokens = (mx.arange(count) + 17)[None]
                model.configure_fixed_groups(False)
                expected = model.verify_mtp_rows(tokens, a)
                model.configure_fixed_groups(True)
                actual = model.verify_mtp_rows(tokens, b)
                for got, want in zip(actual[:2], expected[:2]):
                    self.assertTrue(mx.array_equal(got, want).item())
                for got, want in zip(actual[2], expected[2]):
                    self._assert_optimized_cache_equal(got, want)

    def test_optimized_decode_batch_shares_drafting_and_preserves_owners(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        model = self._optimized_mtp_model()
        for steps in (1, 2, 3, 4):
            for count in (2, 4):
                batched, serial = (
                    MlxMTPServing(model, steps),
                    MlxMTPServing(model, steps),
                )
                a = [model.make_cache() for _ in range(count)]
                b = [model.make_cache() for _ in range(count)]
                tokens = []
                for j in range(count):
                    prompt = (mx.arange(26 + j) % 128)[None]
                    hidden = batched.prefill(str(j), prompt, a[j])
                    serial.prefill(str(j), prompt, b[j])
                    tokens.append(int(mx.argmax(model.lm_head(hidden[:, -1:])).item()))
                for iteration in range(7):
                    inputs = mx.array(tokens)[:, None]
                    expected = mx.concatenate(
                        [
                            serial.decode(str(j), inputs[j : j + 1], b[j])
                            for j in range(count)
                        ],
                        0,
                    )
                    actual = batched.decode_batch(
                        [str(j) for j in range(count)], inputs, a, tokens
                    )
                    self.assertTrue(mx.array_equal(actual, expected).item())
                    for got, want in zip(a, b):
                        self._assert_optimized_cache_equal(got, want)
                    tokens = [int(mx.argmax(actual[j]).item()) for j in range(count)]

    def test_request_batch_preserves_ragged_cache_and_prefix_owners(self):
        """Batched projections cannot merge unequal request visibility or state."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        model = self._optimized_mtp_model()
        model.prepare_for_inference()
        owners = [model.make_cache() for _ in range(3)]
        for j, cache in enumerate(owners):
            mx.eval(model(mx.array([list(range(1, 5 + j))]), cache))
        for step in range(4):
            reference = [[clone_cache(c) for c in cs] for cs in owners]
            tokens = [mx.array([[13 + step + j]]) for j in range(3)]
            expected = mx.concatenate(
                [model(t, c) for t, c in zip(tokens, reference)], 0
            )
            actual = model.decode_requests(tokens, owners)
            self.assertTrue(mx.array_equal(actual, expected).item())
            for got, want in zip(owners, reference):
                for gc, wc in zip(got, want):
                    self.assertEqual(gc.offset, wc.offset)
                    for slot, (gv, wv) in enumerate(zip(gc.cache, wc.cache)):
                        if hasattr(gc, "latent_segments") and slot == 0:
                            gv, wv = gc[0], wc[0]
                        if hasattr(gc, "latent_segments") and slot == 4:
                            continue
                        if gv is None or wv is None:
                            self.assertIs(gv, wv)
                        else:
                            self.assertTrue(mx.array_equal(gv, wv).item())

    def test_mtp_shared_execute_boundary_preserves_request_isolation(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        model = self._mtp_model()
        model.configure_mtp_verification("ffn")
        serial, batch = MlxMTPServing(model, 2), MlxMTPServing(model, 2)
        a, b = (
            [model.make_cache(), model.make_cache()],
            [model.make_cache(), model.make_cache()],
        )
        for i, rid in enumerate(("a", "b")):
            prompt = mx.array([[1, 3 + i, 5, 7]])
            mx.eval(serial.prefill(rid, prompt, a[i]), batch.prefill(rid, prompt, b[i]))
        for step in range(7):
            tokens = [13 + step, 27 - step]
            inputs = mx.array(tokens)[:, None]
            expected = mx.concatenate(
                [
                    serial.decode(
                        rid,
                        inputs[i : i + 1],
                        a[i],
                    )
                    for i, rid in enumerate(("a", "b"))
                ],
                0,
            )
            actual = batch.decode_batch(["a", "b"], inputs, b, tokens)
            self.assertTrue(mx.array_equal(actual, expected).item())
            for got, want in zip(b, a):
                for gc, wc in zip(got, want):
                    self.assertEqual(gc.offset, wc.offset)
                    for gv, wv in zip(gc.cache, wc.cache):
                        if gv is None or wv is None:
                            self.assertIs(gv, wv)
                        else:
                            self.assertTrue(mx.array_equal(gv, wv).item())
        batch.remove("a")
        self.assertNotIn("a", batch.states)
        self.assertIn("b", batch.states)

    def test_affine_mla_verify_and_rejected_prefix_resume(self):
        """Real affine modules must batch without missing attributes or stale KV."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            ModelArgs,
            NativeCache,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        args = ModelArgs(
            model_type="glm5_next",
            text_config=dict(
                hidden_size=512,
                rms_norm_eps=1e-5,
                attention_bias=False,
                num_attention_heads=64,
                q_lora_rank=512,
                kv_lora_rank=512,
                qk_nope_head_dim=256,
                qk_rope_head_dim=0,
                v_head_dim=256,
                index_head_dim=128,
                index_n_heads=4,
                index_topk=512,
                index_kpool=16,
                index_kpool_always_select_tail=True,
            ),
        )
        attn = MLA(args)
        attn.set_dtype(mx.bfloat16)
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            projection = getattr(attn, name).to_quantized(bits=4, group_size=64)
            projection.scales = projection.scales.astype(mx.float16)
            projection.biases = projection.biases.astype(mx.float16)
            setattr(attn, name, projection)
        attn.cache_latent_projection = True
        base = NativeCache(4)
        mx.eval(
            attn(mx.random.normal((1, 512, 512)).astype(mx.bfloat16), base), base.cache
        )
        x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
        serial = clone_cache(base)
        outputs, refs = [], []
        for j in range(3):
            outputs.append(attn(x[:, j : j + 1], serial))
            refs.append(clone_cache(serial))
        actual, snapshots = attn.verify_rows(x, clone_cache(base))
        self.assertTrue(mx.array_equal(actual, mx.concatenate(outputs, 1)).item())
        for got, want in zip(snapshots, refs):
            self.assertEqual(got.offset, want.offset)
            for gv, wv in zip(got.cache, want.cache):
                if gv is None or wv is None:
                    self.assertIs(gv, wv)
                else:
                    self.assertTrue(mx.array_equal(gv, wv).item())
        follow = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
        for consumed in (1, 2):
            got = attn(follow, clone_cache(snapshots[consumed - 1]))
            want = attn(follow, clone_cache(refs[consumed - 1]))
            self.assertTrue(mx.array_equal(got, want).item())

    def test_zero_step_decode_keeps_lazy_graph_and_cache_publication(self):
        """Zero-step execution publishes state without synchronizing lazy logits."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        model = self._mtp_model()
        serving = MlxMTPServing(model, 0)
        cache = model.make_cache()
        mx.eval(serving.prefill("zero", mx.array([[1, 3, 5]]), cache))
        pending = serving.build_decode("zero", mx.array([[7]]), cache, 7)
        self.assertIsNone(pending.chain)
        # A real lazy Metal graph makes an accidental execute-time eval fail.
        kernel = mx.fast.metal_kernel(
            name="sglang_test_lazy_zero_step",
            input_names=["x"],
            output_names=["out"],
            source="deliberate compile error",
        )
        pending.logits = kernel(
            inputs=[pending.logits],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[pending.logits.shape],
            output_dtypes=[pending.logits.dtype],
        )[0]
        out = serving.execute_batch([pending])[0]
        self.assertIs(out, pending.logits)
        self.assertEqual(serving.states["zero"].offset, cache[0].offset)
        self.assertEqual(cache[0].mlx_mtp[3], cache[0].offset)
        with self.assertRaises(RuntimeError):
            mx.eval(out)

    def test_dense_latent_cache_tracks_dtype_and_weight_replacement(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLA

        attn = MLA(self.model.args)
        attn.kv_b_proj.weight = attn.kv_b_proj.weight.astype(mx.bfloat16)
        reference = attn._latent_matrices(mx.float32)
        attn.cache_latent_projection = True
        promoted = attn._latent_matrices(mx.float32)
        self.assertIs(promoted, attn._latent_matrices(mx.float32))
        self.assertTrue(all(v.dtype == mx.float32 for v in promoted))
        for actual, expected in zip(promoted, reference):
            self.assertTrue(mx.array_equal(actual, expected.astype(mx.float32)).item())
        narrow = attn._latent_matrices(mx.bfloat16)
        self.assertIsNot(narrow, promoted)
        self.assertTrue(all(v.dtype == mx.bfloat16 for v in narrow))
        attn.kv_b_proj.weight = attn.kv_b_proj.weight + mx.array(0.25, mx.bfloat16)
        updated = attn._latent_matrices(mx.bfloat16)
        self.assertIsNot(updated, narrow)
        self.assertFalse(mx.array_equal(updated[0], narrow[0]).item())
        attn.cache_latent_projection = False
        attn._latent_matrices(mx.float32)
        self.assertIsNone(attn._latent_projection)

    def test_mla_retained_projection_invalidates_quantization_updates(self):
        import copy

        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLA

        args = copy.copy(self.model.args)
        args.kv_lora_rank = 32
        attn = MLA(args)
        attn.kv_b_proj = attn.kv_b_proj.to_quantized(group_size=32, bits=4)
        attn.cache_latent_projection = True
        first = attn._latent_matrices(mx.float32)
        self.assertIs(first, attn._latent_matrices(mx.float32))
        attn.kv_b_proj.scales = attn.kv_b_proj.scales * 1.01
        second = attn._latent_matrices(mx.float32)
        self.assertIsNot(first, second)
        p = attn.kv_b_proj
        expected = mx.dequantize(
            p.weight,
            p.scales,
            p.biases,
            group_size=p.group_size,
            bits=p.bits,
            mode=p.mode,
        ).astype(mx.float32)
        wk, wv = mx.split(
            expected.reshape(args.num_attention_heads, -1, args.kv_lora_rank),
            [args.qk_nope_head_dim],
            axis=1,
        )
        self.assertTrue(mx.array_equal(second[0], wk).item())
        self.assertTrue(mx.array_equal(second[1], wv.swapaxes(-1, -2)).item())
        attn.cache_latent_projection = False
        attn._latent_matrices(mx.float32)
        self.assertIsNone(attn._latent_projection)

    def test_mtp_generation_matches_ar_on_accept_and_reject(self):
        """Rejection and bonus tokens preserve following-token alignment."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp import greedy_mtp_generate

        for accept_all in (False, True):
            with self.subTest(accept_all=accept_all):
                model = self._mtp_model()
                if accept_all:
                    model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
                prompt = mx.array([self.tokens])
                cache = model.make_cache()
                x = prompt
                expected = []
                for _ in range(17):
                    logits = model.forward_last_logits(x, cache)
                    x = mx.argmax(logits, -1)
                    expected.append(x.item())
                actual = greedy_mtp_generate(model, prompt, 17)
                self.assertEqual(actual.tokens, expected)
                if accept_all:
                    self.assertEqual(
                        actual.num_correct_drafts, actual.num_proposed_drafts
                    )
                else:
                    self.assertGreater(
                        actual.num_proposed_drafts, actual.num_correct_drafts
                    )
                for limit in (0, 1, 2, 4):
                    self.assertEqual(
                        greedy_mtp_generate(model, prompt, limit).tokens,
                        expected[:limit],
                    )
                stop = expected[0]
                self.assertEqual(
                    greedy_mtp_generate(model, prompt, 17, eos_tokens=[stop]).tokens,
                    [stop],
                )

    def test_serving_mtp_publishes_only_consumed_rows(self):
        """Speculative state must not escape to prefix cache or output logprobs."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        for steps in (0, 1, 2, 4, 8, 16, -1):
            with self.subTest(steps=steps):
                model = self._mtp_model()
                engine = MlxMTPServing(model, steps)
                baseline, actual = model.make_cache(), model.make_cache()
                prompt = mx.array([self.tokens])
                expected = model.model(prompt, baseline)
                hidden = engine.prefill("a", prompt, actual)
                self.assertTrue(mx.array_equal(hidden, expected).item())
                token = mx.argmax(model.lm_head(expected[:, -1:]), -1)
                count = 20 if steps == 16 else 12
                for i in range(count):
                    # An external grammar/bias/sampler may pick a different
                    # token than the drafter. Its actual choice must win.
                    if i == 3:
                        token = mx.array([[(token.item() + 7) % 64]])
                    engine.set_remaining("a", count - i)
                    expected, _ = model.forward_with_hidden(token, baseline)
                    logits = engine.decode("a", token, actual)
                    self.assertTrue(mx.array_equal(logits, expected[:, -1]).item())
                    for a, b in zip(actual, baseline):
                        self.assertEqual(a.offset, b.offset)
                        for av, bv in zip(a.cache, b.cache):
                            if av is None or bv is None:
                                self.assertIs(av, bv)
                            else:
                                self.assertTrue(mx.array_equal(av, bv).item())
                    token = mx.argmax(expected, -1)
                engine.remove("a")
                self.assertFalse(engine.states)

    def test_serving_mtp_prefix_snapshot_and_chunk_handoff(self):
        """Restore target and head at one committed boundary, without pending drafts."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            _restore_cache,
            _snapshot_cache,
        )
        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        model = self._mtp_model()
        engine = MlxMTPServing(model, 3)
        cache = model.make_cache()
        # Chunk handoff pairs the previous chunk's hidden with the first
        # token of the next chunk, rather than losing one head position.
        prompt = mx.array([self.tokens])
        engine.prefill("a", prompt[:, :4], cache)
        hidden = engine.prefill("a", prompt[:, 4:], cache)
        self.assertEqual(engine.states["a"].head.offset, len(self.tokens) - 1)
        token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
        logits = engine.decode("a", token, cache)
        boundary = cache[0].offset
        saved = [_snapshot_cache(c) for c in cache]
        restored = model.make_cache()
        for dst, snapshot in zip(restored, saved):
            _restore_cache(dst, snapshot)
        self.assertEqual(restored[0].mlx_mtp[3], boundary)
        engine.remove("a")
        token = mx.argmax(logits, -1)[:, None]
        # Reusing the request id with a restored prefix must not reuse a
        # previous speculative queue. Only the saved committed head survives.
        resumed = engine.decode("a", token, restored)
        expected, _ = model.forward_with_hidden(token, cache)
        self.assertTrue(mx.array_equal(resumed, expected[:, -1]).item())
        self.assertEqual(engine.states["a"].head.offset, boundary)
        engine.clear()
        self.assertFalse(engine.states)

    def test_serving_mtp_can_resume_after_ordinary_decode(self):
        """Adaptive zero-step mode defers head work without losing known-token pairs."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

        model = self._mtp_model()
        engine = MlxMTPServing(model, 0)
        cache, baseline = model.make_cache(), model.make_cache()
        prompt = mx.array([self.tokens])
        hidden = engine.prefill("a", prompt, cache)
        mx.eval(model.model(prompt, baseline))
        token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
        for i in range(10):
            if i == 4:
                engine.num_steps = 3
            logits = engine.decode("a", token, cache)
            expected, _ = model.forward_with_hidden(token, baseline)
            self.assertTrue(mx.array_equal(logits, expected[:, -1]).item())
            token = mx.argmax(expected, -1)
        self.assertFalse(engine.states["a"].pending_hidden)
        self.assertEqual(engine.states["a"].head.offset, cache[0].offset - 1)

    def test_adaptive_mtp_scores_consumed_tokens_and_reexplores(self):
        """Wasted drafts must reduce a budget's score; zero remains a valid choice."""
        from sglang.srt.hardware_backend.mlx.mtp_serving import AdaptiveMTPBudget

        budget = AdaptiveMTPBudget()
        for _ in budget.candidates:
            steps = budget.choose()
            # Two drafts commit three rows at cost 0.1s; other windows
            # consume one row at cost 0.1s despite their draft count.
            budget.observe(steps, 3 if steps == 2 else 1, 0.1)
        self.assertEqual(budget.choose(), 2)
        budget.observe(0, 1, 0.001)
        self.assertEqual(budget.choose(), 0)
        # Revisit neighbors without repeatedly paying for distant losing budgets.
        explored = set()
        for _ in range(16 * len(budget.candidates)):
            steps = budget.choose()
            explored.add(steps)
            budget.observe(steps, 1, 0.001 if steps == 0 else 0.1)
        self.assertEqual(explored, {0, 1})

    def test_mtp_snapshots_share_storage_and_isolate_slice_assignment(self):
        """Isolate slice assignment without copying multi-MiB recurrent states."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache
        from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

        cache = NativeCache(4)
        cache[0] = mx.ones((1, 64, 128, 128), mx.float32)
        cache[3] = mx.zeros((1, 64, 128), mx.float32)
        mx.eval(cache.cache)
        before = mx.get_active_memory()
        snapshot = clone_cache(cache)
        mx.eval(snapshot.cache)
        self.assertLessEqual(mx.get_active_memory() - before, 4096)
        self.assertIsNot(snapshot[0], cache[0])
        self.assertIsNot(snapshot[3], cache[3])
        cache[0][0, 0, 0, 0] = 3
        cache[3][:, 0] = 7
        self.assertEqual(snapshot[0][0, 0, 0, 0].item(), 1)
        self.assertEqual(mx.max(snapshot[3]).item(), 0)


if __name__ == "__main__":
    unittest.main()
