"""DFlash lookahead must publish exactly the sampled target prefix.

A deterministic token tape replaces only the external draft checkpoint. Real
tiny GLM KDA/MLA, mHC, MoE, logits and cache snapshots remain under test.
"""

import copy
import importlib.util
import os
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=15, suite="stage-a-unit-test-mlx")


class _TapeDrafter:
    target_layer_ids = (2, 0, 3)
    block_size = 3

    def __init__(self, tape, *, miss=False):
        self.tape = tape
        self.miss = miss

    def make_state(self):
        import mlx.core as mx

        return {
            "features": mx.zeros((1, 0, 96)),
            "positions": mx.array([], dtype=mx.int32),
            "chunks": [],
        }

    def consume(self, features, positions, state):
        import mlx.core as mx

        # No in-place updates: published prefix snapshots may still own state.
        return {
            "features": mx.concatenate([state["features"], features], 1),
            "positions": mx.concatenate([state["positions"], positions]),
            "chunks": state["chunks"] + [(features, positions)],
        }

    def draft(self, anchors, states, embedding, lm_head, count):
        import mlx.core as mx

        # The tape was generated before serving starts; drafting never executes
        # the target or advances any request's consumed-feature context. Outside
        # its oracle horizon, repeat deterministically rather than overflowing;
        # hit assertions below stay within the precomputed continuation.
        return mx.array(
            [
                [
                    (
                        self.tape[(state["positions"].size + j + 1) % len(self.tape)]
                        + self.miss
                    )
                    % 64
                    for j in range(count)
                ]
                for state in states
            ],
            dtype=anchors.dtype,
        )


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxDFlashServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        precision = patch.dict(os.environ, {"MLX_ENABLE_TF32": "0"})
        precision.start()
        cls.addClassCleanup(precision.stop)
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model, ModelArgs
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )

        # Same dimensions and equations as TestGlm5NextMlx.setUpClass, without
        # importing its entire test suite or loading a reference checkpoint.
        config = Glm5NextTextConfig(
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
        cfg = config.to_dict()
        cfg.pop("architectures", None)
        cfg.update(mhc=True, scoring_func="sigmoid", max_position_embeddings=128)
        cfg["linear_attn_config"] = dict(
            num_heads=2, head_dim=32, short_conv_kernel_size=4, gate_lower_bound=-5.0
        )
        mx.random.seed(42)
        cls.template = Model(ModelArgs(model_type="glm5_next", text_config=cfg))
        mx.eval(cls.template.parameters())
        cls.prompt = list(range(1, 14))

    def setUp(self):
        self.ar = copy.deepcopy(self.template)
        self.ar.configure_fixed_groups(False)
        self.target = copy.deepcopy(self.template)
        self.target.configure_mtp_verification("ffn")
        self.target.prepare_for_inference()

    def test_image_embeddings_preserve_dflash_features_across_prefill_chunks(self):
        """Image spans must feed target captures, not hashed placeholder embeddings."""
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
        from sglang.srt.hardware_backend.mlx.vision import VisionPrompt

        values = mx.random.normal((5, self.target.args.hidden_size))
        ids = self.prompt[:2] + [1000000] * 5 + self.prompt[7:]
        prompt = VisionPrompt(((2, 7, values),))
        cache = self.target.make_cache()
        drafter = _TapeDrafter(self.prompt + [1] * 16)
        engine = MlxDFlashServing(self.target, None, 2, drafter=drafter)
        oracle_cache = self.ar.make_cache()
        for start, end in ((0, 4), (4, 8), (8, len(ids))):
            inputs = self._inputs(ids[start:end])
            embeds = prompt.embed(self.target.model.embed_tokens, inputs, start)
            actual = engine.prefill("image", inputs, cache, inputs_embeds=embeds)
            oracle_embeds = prompt.embed(self.ar.model.embed_tokens, inputs, start)
            expected, _ = self.ar.model(
                inputs,
                oracle_cache,
                capture_layer_ids=engine.capture_layer_ids,
                inputs_embeds=oracle_embeds,
            )
            self._assert_array(actual, expected)
            self._assert_cache(cache, oracle_cache)
        self.assertEqual(engine.states["image"].offset, len(ids))
        self.assertTrue(engine.remove("image"))

    def _inputs(self, tokens):
        import mlx.core as mx

        return mx.array([tokens], dtype=mx.int32)

    def _assert_array(self, actual, expected):
        import mlx.core as mx
        import numpy as np

        self.assertEqual(actual.shape, expected.shape)
        # Established GLM fixture tolerances, including optimized reductions.
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            rtol=getattr(self, "rtol", 2e-4),
            atol=getattr(self, "atol", 2e-5),
        )

    def _assert_cache(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertEqual(got.offset, want.offset)
            self.assertEqual(len(got.cache), len(want.cache))
            for a, b in zip(got.cache, want.cache):
                if a is None or b is None:
                    self.assertIs(a, b)
                else:
                    self._assert_array(a, b)

    def _ar_forward(self, tokens, cache):
        """Explicit target layers; no DFlash/MTP verification or capture wrapper."""
        import mlx.core as mx

        x = self.ar.model.embed_tokens(self._inputs(tokens))
        x = mx.repeat(x[..., None, :], self.ar.args.hc_mult, -2)
        captures = {}
        for i, layer in enumerate(self.ar.layers):
            x = layer(x, cache[i])
            if i in _TapeDrafter.target_layer_ids:
                captures[i] = x.mean(-2)
        hidden = self.ar.model.norm(x.mean(-2))
        features = mx.concatenate(
            [captures[i] for i in _TapeDrafter.target_layer_ids], -1
        )
        return self.ar.lm_head(hidden)[:, -1], hidden, features

    def _tape(self, prompt):
        import mlx.core as mx

        cache = self.ar.make_cache()
        logits, _, _ = self._ar_forward(prompt, cache)
        tape = list(prompt)
        for _ in range(12):
            token = int(mx.argmax(logits, -1).item())
            tape.append(token)
            logits, _, _ = self._ar_forward([token], cache)
        return tape

    def _serving(self, *, miss=False, tape=None, drafts=2):
        from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing

        return MlxDFlashServing(
            self.target,
            None,
            num_drafts=drafts,
            drafter=_TapeDrafter(tape or self._tape(self.prompt), miss=miss),
        )

    def _assert_context(self, cache, expected_features):
        import mlx.core as mx

        offset, ids, state, pending, positions = cache[0].mlx_dflash
        self.assertEqual(offset, cache[0].offset)
        self.assertEqual(ids, _TapeDrafter.target_layer_ids)
        features = mx.concatenate([state["features"], *pending], 1)
        consumed = mx.concatenate([state["positions"], *positions])
        self.assertTrue(mx.array_equal(consumed, mx.arange(offset)).item())
        self._assert_array(features, expected_features)

    def _snapshots(self, cache):
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            _snapshot_cache,
        )

        return [_snapshot_cache(c) for c in cache]

    def _restore(self, snapshots):
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            _restore_cache,
        )

        cache = self.target.make_cache()
        for c, snapshot in zip(cache, snapshots):
            _restore_cache(c, snapshot)
        return cache

    def test_target_checkpoint_discards_unrequested_native_mtp(self):
        """A DFlash target must not retain the checkpoint's unused prediction head."""
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.models.glm5_next import Model

        target = Model(copy.copy(self.template.args))
        prediction_prefix = f"model.layers.{target.args.num_hidden_layers}."
        trunk = mx.ones((target.args.hidden_size,), mx.float32)
        weights = {
            "model.norm.weight": trunk,
            prediction_prefix + "enorm.weight": mx.ones_like(trunk),
            "model.mtp.enorm.weight": mx.ones_like(trunk),
            "mtp.enorm.weight": mx.ones_like(trunk),
        }
        sanitized = target.sanitize(weights)
        self.assertIsNone(target.mtp)
        self.assertEqual(len(target.model.layers), target.args.num_hidden_layers)
        self.assertEqual(set(sanitized), {"model.norm.weight"})
        self.assertTrue(mx.array_equal(sanitized["model.norm.weight"], trunk).item())
        self.assertEqual(weights, {})

    def test_hits_and_misses_preserve_raw_logits_tokens_and_consumed_caches(self):
        """Accept/reject must never expose the cache at the end of a draft chain."""
        self._check_hits_and_misses()

    def test_bf16_prepared_projections_preserve_serving_prefix(self):
        """BF16 prepared input projections retain logits and consumed features."""
        import mlx.core as mx

        # Cast canonical parameters before preparation creates fused views.
        self.ar = copy.deepcopy(self.template)
        self.ar.set_dtype(mx.bfloat16)
        self.ar.configure_fixed_groups(False)
        self.target = copy.deepcopy(self.ar)
        self.target.configure_fixed_groups(True)
        self.target.configure_mtp_verification("ffn")
        self.target.prepare_for_inference()
        self.assertEqual(self.target.model.embed_tokens.weight.dtype, mx.bfloat16)
        for layer in self.target.layers:
            if layer.is_linear:
                self.assertTrue(layer.self_attn._fused_qkv_is_current())
                self.assertEqual(len(layer.self_attn._fused_input_names), 6)
                self.assertEqual(layer.self_attn._fused_qkv.weight.dtype, mx.bfloat16)
        # Existing BF16 KDA projection tolerance. Head dim 32 exercises general
        # recurrence with fused input weights; head-dim-128 kernels have separate
        # coverage in TestGlm5NextMlx, and are not claimed by this serving test.
        self.rtol, self.atol = 0.03, 0.003
        self._check_hits_and_misses()

    def _check_hits_and_misses(self):
        import mlx.core as mx

        for miss in (False, True):
            for token_only in (False, True):
                with self.subTest(miss=miss, token_only=token_only):
                    serving = self._serving(miss=miss)
                    cache, reference = self.target.make_cache(), self.ar.make_cache()
                    hidden = serving.prefill("r", self._inputs(self.prompt), cache)
                    logits, expected, features = self._ar_forward(
                        self.prompt, reference
                    )
                    self._assert_array(hidden, expected)
                    token = int(mx.argmax(logits, -1).item())
                    for _ in range(7):
                        actual = serving.decode_batch(
                            ["r"],
                            self._inputs([token]),
                            [cache],
                            [token],
                            token_only=token_only,
                        )
                        logits, _, captured = self._ar_forward([token], reference)
                        expected_token = mx.argmax(logits, -1)
                        if token_only:
                            self.assertEqual(actual.shape, (1, 1))
                            self.assertEqual(
                                int(actual.item()), int(expected_token.item())
                            )
                        else:
                            self._assert_array(actual, logits)
                            self.assertTrue(
                                mx.array_equal(
                                    mx.argmax(actual, -1), expected_token
                                ).item()
                            )
                        self._assert_cache(cache, reference)
                        features = mx.concatenate([features, captured], 1)
                        self._assert_context(cache, features)
                        token = int(expected_token.item())
                    stats = serving.stats()
                    self.assertEqual(stats["num_accept_tokens"], 7)
                    self.assertEqual(
                        stats.get("num_correct_drafts", 0), 0 if miss else 4
                    )
                    self.assertEqual(stats.get("reject_ct", 0), 6 if miss else 0)
                    self.assertEqual(stats["verify_ct"], 7 if miss else 3)

    def test_remaining_clips_new_chains_and_terminal_release(self):
        """Budget clips proposals; a terminal sample never publishes extra rows."""
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing

        drafter = _TapeDrafter([0])
        drafter.block_size = 8
        default = MlxDFlashServing(self.target, None, drafter=drafter)
        explicit = MlxDFlashServing(self.target, None, 7, drafter=drafter)
        self.assertEqual(default.num_drafts, 3)
        self.assertEqual(explicit.num_drafts, 7)
        self.assertEqual(drafter.block_size, 8)

        for remaining, proposed in ((-2, 0), (0, 0), (1, 0), (2, 1), (3, 2)):
            with self.subTest(remaining=remaining):
                # A one-token repeating tape deliberately works at any offset.
                # Clipping is asserted through proposal counts/cache boundaries,
                # never through a fake-drafter IndexError.
                serving = self._serving(tape=[0])
                cache, reference = self.target.make_cache(), self.ar.make_cache()
                serving.prefill("r", self._inputs(self.prompt), cache)
                logits, _, features = self._ar_forward(self.prompt, reference)
                serving.set_remaining("r", remaining)
                token = int(mx.argmax(logits, -1).item())
                actual = serving.decode("r", self._inputs([token]), cache)
                logits, _, captured = self._ar_forward([token], reference)
                self._assert_array(actual, logits)
                self._assert_cache(cache, reference)
                self._assert_context(cache, mx.concatenate([features, captured], 1))
                self.assertEqual(serving.stats()["num_proposed_drafts"], proposed)
                self.assertEqual(len(serving.states["r"].rows), proposed)
                self.assertTrue(serving.remove("r"))

        # A reduced budget may encounter a previously verified queue. The
        # scheduler still consumes one row and releases the terminal request;
        # set_remaining does not need to eagerly trim those private snapshots.
        serving = self._serving()
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        serving.prefill("r", self._inputs(self.prompt), cache)
        logits, _, features = self._ar_forward(self.prompt, reference)
        for remaining in (3, 1):
            serving.set_remaining("r", remaining)
            token = int(mx.argmax(logits, -1).item())
            actual = serving.decode("r", self._inputs([token]), cache)
            logits, _, captured = self._ar_forward([token], reference)
            self._assert_array(actual, logits)
            self._assert_cache(cache, reference)
            features = mx.concatenate([features, captured], 1)
            self._assert_context(cache, features)
        self.assertEqual(serving.stats()["verify_ct"], 1)
        self.assertEqual(serving.stats()["num_accept_tokens"], 2)
        self.assertTrue(serving.remove("r"))
        self._assert_cache(self._restore(self._snapshots(cache)), reference)

    def test_sampler_edit_discards_verified_future(self):
        """Grammar/sampling edits must feed the chosen token through the target."""
        import mlx.core as mx

        serving = self._serving()
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        serving.prefill("r", self._inputs(self.prompt), cache)
        logits, _, features = self._ar_forward(self.prompt, reference)
        token = int(mx.argmax(logits, -1).item())
        actual = serving.decode("r", self._inputs([token]), cache)
        logits, _, captured = self._ar_forward([token], reference)
        self._assert_array(actual, logits)
        self._assert_cache(cache, reference)
        features = mx.concatenate([features, captured], 1)
        self._assert_context(cache, features)
        queued = serving.states["r"].rows[0]
        edited = (int(queued.input_token.item()) + 1) % 64
        for token in (edited, 19, 23):
            actual = serving.decode("r", self._inputs([token]), cache)
            logits, _, captured = self._ar_forward([token], reference)
            self._assert_array(actual, logits)
            self._assert_cache(cache, reference)
            features = mx.concatenate([features, captured], 1)
            self._assert_context(cache, features)
        self.assertGreaterEqual(serving.stats()["reject_ct"], 1)

    def test_chunked_prefill_and_prefix_snapshot_clone(self):
        """Restored chunk boundaries retain ordered captures and isolate branches."""
        import mlx.core as mx

        serving = self._serving()
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        features = []
        for start, end in ((0, 5), (5, 10)):
            chunk = self.prompt[start:end]
            hidden = serving.prefill("source", self._inputs(chunk), cache)
            _, expected, captured = self._ar_forward(chunk, reference)
            self._assert_array(hidden, expected)
            features.append(captured)
        snapshots = self._snapshots(cache)
        restored = self._restore(snapshots)
        self._assert_cache(restored, reference)
        saved_state = cache[0].mlx_dflash[2]
        restored_state = restored[0].mlx_dflash[2]
        self.assertIsNot(saved_state, restored_state)
        self.assertIsNot(saved_state["chunks"], restored_state["chunks"])
        self.assertIsNot(saved_state["features"], restored_state["features"])
        for rid, branch in (("source", cache), ("fork", restored)):
            hidden = serving.prefill(rid, self._inputs(self.prompt[10:]), branch)
            branch_ar = self.ar.make_cache()
            for chunk in (self.prompt[:5], self.prompt[5:10], self.prompt[10:]):
                logits, expected, captured = self._ar_forward(chunk, branch_ar)
            self._assert_array(hidden, expected)
            self._assert_cache(branch, branch_ar)
            self._assert_context(branch, mx.concatenate([*features, captured], 1))
            token = int(mx.argmax(logits, -1).item())
            self._assert_array(
                serving.decode(rid, self._inputs([token]), branch),
                self._ar_forward([token], branch_ar)[0],
            )
            self._assert_cache(branch, branch_ar)
        # Slice writes and nested container edits on either restore must leave
        # both the stored prefix and its original request unchanged.
        another = self._restore(snapshots)
        another[0].mlx_dflash[2]["features"][0, 0, 0] = 999
        another[0].mlx_dflash[2]["chunks"].clear()
        another[0].cache[0][0, 0, 0] = 999
        pristine = self._restore(snapshots)
        self._assert_cache(pristine, reference)
        self._assert_context(pristine, mx.concatenate(features, 1))
        self.assertEqual(saved_state["positions"].size, 10)
        self.assertEqual(len(saved_state["chunks"]), 2)

    def test_remove_and_clear_do_not_publish_future_rows(self):
        """Abort/release and request-ID reuse resume only the committed prefix."""
        import mlx.core as mx

        serving = self._serving()
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        serving.prefill("r", self._inputs(self.prompt), cache)
        logits, _, features = self._ar_forward(self.prompt, reference)
        token = int(mx.argmax(logits, -1).item())
        actual = serving.decode("r", self._inputs([token]), cache)
        logits, _, captured = self._ar_forward([token], reference)
        self._assert_array(actual, logits)
        features = mx.concatenate([features, captured], 1)
        self.assertEqual(len(serving.states["r"].rows), 2)
        self.assertEqual(serving.states["r"].rows[-1].cache[0].offset, 16)
        self._assert_cache(cache, reference)
        self._assert_context(cache, features)
        snapshots = self._snapshots(cache)
        # Future tensors are private even before release.
        serving.states["r"].rows[-1].cache[0].cache[0][0, 0, 0] = 999
        self._assert_cache(cache, reference)
        self.assertTrue(serving.remove("r"))
        self.assertFalse(serving.remove("r"))
        restored = self._restore(snapshots)
        token = int(mx.argmax(logits, -1).item())
        actual = serving.decode("r", self._inputs([token]), restored)
        logits, _, captured = self._ar_forward([token], reference)
        self._assert_array(actual, logits)
        self._assert_cache(restored, reference)
        self._assert_context(restored, mx.concatenate([features, captured], 1))
        self.assertEqual(serving.stats()["verify_ct"], 2)
        self.assertEqual(serving.stats().get("num_correct_drafts", 0), 0)
        serving.clear()
        self.assertEqual(serving.states, {})
        fresh = self.target.make_cache()
        hidden = serving.prefill("r", self._inputs([7, 9]), fresh)
        _, expected, captured = self._ar_forward([7, 9], self.ar.make_cache())
        self._assert_array(hidden, expected)
        self._assert_context(fresh, captured)

    def test_gpu_round_prefetch_is_private_until_anchor_confirmation(self):
        import mlx.core as mx

        serving = self._serving()
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        serving.prefill("r", self._inputs(self.prompt), cache)
        logits, _, _ = self._ar_forward(self.prompt, reference)
        token = int(mx.argmax(logits, -1).item())
        for _ in range(6):
            serving.set_remaining("r", 20)
            got = serving.decode_batch(
                ["r"],
                self._inputs([token]),
                [cache],
                [token],
                token_only=True,
                prefetch_allowed=True,
            )
            expected, _, _ = self._ar_forward([token], reference)
            mx.eval(got, expected)
            self.assertEqual(int(got.item()), int(mx.argmax(expected, -1).item()))
            self._assert_cache(cache, reference)
            state = serving.states["r"]
            self.assertEqual(
                state.offset, len(self.prompt) + serving.counters["num_accept_tokens"]
            )
            token = int(got.item())
        self.assertGreater(serving.stats().get("prefetch_ct", 0), 0)
        self.assertGreater(serving.stats().get("prefetch_use_ct", 0), 0)
        # A stop/mode transition must not publish the prefetched future.
        frozen = self._snapshots(cache)
        offset = cache[0].offset
        serving.remove("r")
        self.assertEqual(cache[0].offset, offset)
        restored = self._restore(frozen)
        self._assert_cache(restored, reference)
        self.assertIsNone(serving._state("new", restored).prefetch)

    def test_rejection_boundary_prefetch_preserves_consumed_prefix(self):
        import mlx.core as mx

        serving = self._serving(miss=True)
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        serving.prefill("r", self._inputs(self.prompt), cache)
        logits, _, _ = self._ar_forward(self.prompt, reference)
        token = int(mx.argmax(logits, -1).item())
        for step in range(8):
            serving.set_remaining("r", 20 - step)
            got = serving.decode_batch(
                ["r"],
                self._inputs([token]),
                [cache],
                [token],
                token_only=True,
                prefetch_allowed=True,
            )
            expected, _, _ = self._ar_forward([token], reference)
            mx.eval(got, expected)
            self.assertEqual(int(got.item()), int(mx.argmax(expected, -1).item()))
            self._assert_cache(cache, reference)
            token = int(got.item())
        self.assertGreater(serving.stats().get("reject_ct", 0), 0)
        self.assertGreater(serving.stats().get("prefetch_reject_boundary_ct", 0), 0)
        frozen = self._snapshots(cache)
        serving.remove("r")
        self._assert_cache(self._restore(frozen), reference)

    def test_copy_proposals_verify_edits_and_prefix_restore(self):
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.context_copy import ContextCopyProposer
        from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing

        class UnexpectedDraft(_TapeDrafter):
            def draft(self, *args, **kwargs):
                raise AssertionError(
                    "A complete copy proposal must avoid the draft model"
                )

        # A caller may edit any sampled token. The copied suffix is only a proposal.
        prompt = list(range(1, 9)) + [21, 22, 23] + list(range(1, 8))
        serving = MlxDFlashServing(self.target, None, 2, drafter=UnexpectedDraft([]))
        cache, reference = self.target.make_cache(), self.ar.make_cache()
        for chunk in (prompt[:11], prompt[11:]):
            serving.prefill("source", self._inputs(chunk), cache)
            self._ar_forward(chunk, reference)
        frozen = self._snapshots(cache)
        restored = self._restore(frozen)
        for rid, branch in (("source", cache), ("restored", restored)):
            state = serving._state(rid, branch)
            self.assertEqual(state.copy_proposer.tokens, prompt)
            self.assertEqual(state.copy_proposer.propose(8, 2), (21, 22))
        # Verification consumes only the anchor even when the copy is wrong.
        actual = serving.decode_batch(["source"], self._inputs([8]), [cache], [8])
        expected, _, _ = self._ar_forward([8], reference)
        mx.eval(actual, expected)
        self._assert_array(actual, expected)
        self._assert_cache(cache, reference)
        self.assertEqual(serving.stats()["num_copy_drafts"], 2)
        self.assertEqual(serving.states["source"].copy_proposer.tokens, prompt + [8])
        self.assertEqual(
            ContextCopyProposer(restored[0].mlx_context_copy).tokens, prompt
        )
        # Replace the sampler's next token. New context must exclude rejected 21/22.
        serving.num_drafts = 0
        actual = serving.decode_batch(["source"], self._inputs([42]), [cache], [42])
        expected, _, _ = self._ar_forward([42], reference)
        mx.eval(actual, expected)
        self._assert_array(actual, expected)
        self._assert_cache(cache, reference)
        self.assertEqual(
            serving.states["source"].copy_proposer.tokens, prompt + [8, 42]
        )
        self.assertTrue(serving.remove("source"))
        rebuilt = self._restore(frozen)
        self.assertEqual(serving._state("new", rebuilt).copy_proposer.tokens, prompt)

    def test_ragged_batch_is_independent_with_remaining_and_mode_changes(self):
        """Cohorts and cached token-only rows must not cross request boundaries."""
        import mlx.core as mx

        serving = self._serving()
        prompts = (self.prompt, [7, 9, 11, 3, 2])
        caches = [self.target.make_cache() for _ in prompts]
        references = [self.ar.make_cache() for _ in prompts]
        features, tokens = [], []
        for i, prompt in enumerate(prompts):
            serving.prefill(str(i), self._inputs(prompt), caches[i])
            logits, _, captured = self._ar_forward(prompt, references[i])
            features.append(captured)
            tokens.append(int(mx.argmax(logits, -1).item()))
        serving.set_remaining("1", 1)
        for step, token_only in enumerate((True, False, False, True)):
            # Reverse ordering to catch accidental use of the first request's
            # context, cache, remaining budget or cached output representation.
            order = [1, 0] if step % 2 else [0, 1]
            actual = serving.decode_batch(
                [str(i) for i in order],
                mx.array([[tokens[i]] for i in order], dtype=mx.int32),
                [caches[i] for i in order],
                [tokens[i] for i in order],
                token_only=token_only,
            )
            for row, i in enumerate(order):
                logits, _, captured = self._ar_forward([tokens[i]], references[i])
                expected_token = int(mx.argmax(logits, -1).item())
                if token_only:
                    self.assertEqual(int(actual[row].item()), expected_token)
                else:
                    self._assert_array(actual[row : row + 1], logits)
                tokens[i] = expected_token
                self._assert_cache(caches[i], references[i])
                features[i] = mx.concatenate([features[i], captured], 1)
                self._assert_context(caches[i], features[i])
            self.assertFalse(serving.states["1"].rows)
            if step == 0:
                self.assertEqual(len(serving.states["0"].rows), 2)
        self.assertIsNot(
            serving.states["0"].draft_state, serving.states["1"].draft_state
        )


if __name__ == "__main__":
    unittest.main()
