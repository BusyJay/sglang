# SPDX-License-Identifier: Apache-2.0
"""Startup evaluation of small routed FFNs without request or cache state."""

import mlx.core as mx


def warmup_mlx_switch_glu(block, *, picks, token_counts=(1, 2), dtypes=None):
    """Evaluate the serving FFN callable before exposing the model to requests.

    Use the loaded projection banks directly, with bounded synthetic activation
    and routing arrays. The all-equal route also exercises expert-pair sharing.
    No checkpoint copies, request caches or RNG state are created or modified.
    This warms the eager FFN path and its Metal pipelines; callers using an
    outer compiled graph must additionally evaluate that same graph at startup.
    """
    if dtypes is None:
        dtypes = (mx.float32, mx.bfloat16)
    if picks < 1 or not token_counts or any(c < 1 or c > 8 for c in token_counts):
        raise ValueError("Expected positive picks and token counts in 1..8")
    hidden_size = block.gate_proj.weight.shape[-1] * 2
    for dtype in dtypes:
        for count in token_counts:
            # match the model's token layout; singleton routes are valid even
            # when a compact test checkpoint contains fewer experts than picks.
            x = mx.zeros((1, count, hidden_size), dtype)
            indices = mx.zeros((1, count, picks), mx.uint32)
            mx.eval(block(x, indices))


def warmup_mlx_model(
    model, *, verification_counts=(), capture_layer_ids=(), token_only=False
):
    """Evaluate real decode/verification graph owners with disposable caches.

    These calls retain the model's normal compiled graph instances. Only private
    startup caches advance; no request is registered and no prefix is published.
    Later unseen cache layouts or request shapes can still require compilation.
    """
    if any(c < 2 or c > 8 for c in verification_counts):
        raise ValueError("Expected verification counts in 2..8")
    cache = model.make_cache()

    def finish(value):
        mx.eval(value, [state.state for state in cache])

    finish(model.forward_last_logits(mx.zeros((1, 2), mx.int32), cache))
    # Verification publishes its normal snapshots but they are private and
    # discarded here. Counts are explicit serving shapes, not template values.
    for count in verification_counts:
        finish(
            model.verify_mtp_rows(
                mx.zeros((1, count), mx.int32),
                cache,
                token_only=token_only,
                capture_layer_ids=capture_layer_ids,
            )
        )
    # Decode last so bounded graph-owner caches retain the AR variant. Limiting
    # verification warmup to configured depths avoids compiling every possible
    # shape and evicting the very graphs that startup is intended to retain.
    decode = model.forward_last_tokens if token_only else model.forward_last_logits
    for _ in range(2):
        finish(decode(mx.zeros((1, 1), mx.int32), cache))
