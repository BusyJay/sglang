# SPDX-License-Identifier: Apache-2.0
"""Experimental singleton greedy MTP for native GLM MLX models.

This model-level driver owns its caches and provides an isolated benchmark.
HTTP serving uses mtp_serving.MlxMTPServing, which publishes one consumed row
at a time through the ordinary sampler, scheduler and prefix-cache pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.models.glm5_next import (
    Model,
    ModelArgs,
    NativeCache,
)


def load_mtp_model(model_dir, profile="affine_fused", *, cache_latent=True):
    """Load a local original-layout checkpoint including its prediction layer.

    Keep layer N at model.layers.N so mlx-lm applies the checkpoint's exact
    per-module quantization overrides, including mixed 4/5/6-bit experts.
    Strict weight loading rejects incomplete or incompatible MTP heads.
    """
    from mlx_lm.utils import load_model

    def model_classes(config):
        return lambda args: Model(args, enable_mtp=True), ModelArgs

    mx.set_cache_limit(1024**3)
    model, _ = load_model(Path(model_dir), lazy=True, get_model_classes=model_classes)
    model.configure_optimizations(profile)
    model.configure_latent_projection_cache(cache_latent)
    model.prepare_for_inference()
    mx.eval(model.parameters())
    return model


@dataclass
class MTPResult:
    tokens: list[int]
    num_correct_drafts: int = 0
    num_proposed_drafts: int = 0
    verify_ct: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0


def greedy_mtp_generate(model, prompt, max_new_tokens, *, eos_tokens=()):
    """Generate with one draft and a target bonus; reject by restoring root state.

    Every verification row uses singleton decode arithmetic. MTP is primed
    with (hidden[i], token[i+1]) pairs, then advances only along committed
    tokens. Both convolution/recurrent state and latent/indexer state roll
    back together. The final output token remains unconsumed, as in AR.
    """
    if model.mtp is None:
        raise ValueError("MTP generation requires a loaded prediction head")
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("MTP generation requires one nonempty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    result = MTPResult(tokens=[])
    if max_new_tokens == 0:
        return result
    started = perf_counter()
    cache, mtp_cache = model.make_cache(), NativeCache(4)
    logits, hidden = model.forward_with_hidden(prompt, cache)
    root = mx.argmax(logits[:, -1:], axis=-1)
    # Fill the head's history with known following tokens, excluding the
    # final prompt position (whose following token is the first generated).
    if prompt.shape[1] > 1:
        _, out = model.forward_mtp(prompt[:, 1:], hidden[:, :-1], mtp_cache)
        mx.eval(out, mtp_cache.cache)
    hidden = hidden[:, -1:]
    result.tokens.append(root.item())
    result.prefill_s = perf_counter() - started
    decode_started = perf_counter()
    eos_tokens = set(eos_tokens)
    while len(result.tokens) < max_new_tokens and result.tokens[-1] not in eos_tokens:
        if max_new_tokens - len(result.tokens) == 1:
            logits, _ = model.forward_with_hidden(root, cache)
            result.tokens.append(mx.argmax(logits[:, -1:], -1).item())
            break
        draft_logits, _ = model.forward_mtp(root, hidden, mtp_cache)
        draft = mx.argmax(draft_logits, axis=-1)
        logits, rows, root_cache = model.verify_mtp(root, draft, cache)
        choices = mx.argmax(logits, axis=-1)
        mx.eval(choices, draft, [c.cache for c in root_cache])
        target_token = choices[0, 0].item()
        result.verify_ct += 1
        result.num_proposed_drafts += 1
        result.tokens.append(target_token)
        if target_token == draft.item():
            result.num_correct_drafts += 1
            # The head has consumed root; absorb the committed draft using
            # root's target hidden before starting the next prediction.
            _, out = model.forward_mtp(draft, rows[:, :1], mtp_cache)
            mx.eval(out, mtp_cache.cache)
            if target_token in eos_tokens:
                break
            root = choices[:, 1:]
            hidden = rows[:, 1:]
            result.tokens.append(root.item())
        else:
            cache = root_cache
            root = choices[:, :1]
            hidden = rows[:, :1]
    result.decode_s = perf_counter() - decode_started
    return result
