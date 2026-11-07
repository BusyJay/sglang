# SPDX-License-Identifier: Apache-2.0
"""Request-owned adaptive vocabulary for draft scoring; target scoring is full."""

from collections import deque

import mlx.core as mx

from sglang.kernels.ops.gemm.mlx_mxfp8_rows import mlx_mxfp8_indexed_rows
from sglang.kernels.ops.sampling.mlx_topk import mlx_topk, mlx_topk_candidates
from sglang.kernels.ops.sampling.mlx_vocabulary_set import (
    mlx_vocabulary_candidates,
    mlx_vocabulary_set,
)
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

_SENTINEL = 0xFFFFFFFF


class DraftReadoutState:
    """Proposal metadata only; no speculative tokens or KV enter the context."""

    def __init__(self):
        self.recent = deque(maxlen=512)
        self.ids = None
        self.pool = None
        self.calls = 0
        self.full_ct = 0
        self.indexed_ct = 0

    def observe(self, tokens):
        self.recent.extend(int(token) for token in tokens)


class DraftReadout:
    """Share candidate rows across a ready cohort without copying weight rows.

    Full refreshes retain high-scoring vocabulary rows at each draft position.
    Other calls score these rows plus recently consumed request tokens.
    Prefix restore starts a fresh refresh. These proposal hints are separate
    from canonical target/drafter KV and may be discarded at any boundary.
    """

    def __init__(
        self,
        head,
        states,
        vocab_size,
        *,
        refresh_size=16384,
        refresh_period=4,
        deduplicate=True,
    ):
        if refresh_size < 16 or refresh_period < 1:
            raise ValueError(
                "Expected at least 16 candidates and a positive refresh period"
            )
        self.head, self.states, self.vocab_size = head, states, vocab_size
        self.refresh_size, self.refresh_period = refresh_size, refresh_period
        self.deduplicate = deduplicate
        self.refresh = False
        self.limit = None

    @staticmethod
    def supports(head):
        return (
            isinstance(head, FP8Linear)
            and head.block_size == 32
            and head.activation_mode == "a16"
            and "bias" not in head
        )

    def __call__(self, hidden):
        self.limit = None
        self.refresh = any(
            s.ids is None or s.calls % self.refresh_period == 0 for s in self.states
        )
        for state in self.states:
            state.calls += 1
            if self.refresh:
                state.full_ct += 1
            else:
                state.indexed_ct += 1
        if self.refresh:
            return self.head(hidden)[..., : self.vocab_size]
        # Fixed recent-buffer shapes avoid new kernels as context grows.
        # Selection removes repeated IDs, so the steady path needs no sort.
        recent = sorted(
            {t for s in self.states for t in s.recent if 0 <= t < self.vocab_size}
        )
        required = mx.array(
            recent + [_SENTINEL] * (512 * len(self.states) - len(recent)), mx.uint32
        )
        if self.deduplicate:
            ids, limit = mlx_vocabulary_candidates(
                [s.pool for s in self.states], required, self.vocab_size
            )
            self.limit = limit
            return mlx_mxfp8_indexed_rows(
                hidden, self.head.weight, self.head.scales, ids, valid_count=limit
            ), ids
        ids = mx.concatenate([*(s.ids for s in self.states), required])
        return mlx_mxfp8_indexed_rows(
            hidden, self.head.weight, self.head.scales, ids
        ), ids

    def topk(self, logits, token_ids, k):
        """Select after the caller's multiplier/softcap, retaining stable IDs."""
        if not self.refresh:
            return mlx_topk_candidates(logits, token_ids, k, valid_count=self.limit)
        rank = min(self.refresh_size, self.vocab_size)
        top = mx.argpartition(-logits, kth=rank - 1, axis=-1)[..., :rank].astype(
            mx.uint32
        )
        for row, state in enumerate(self.states):
            recent = sorted({t for t in state.recent if 0 <= t < self.vocab_size})
            required = mx.array(recent + [_SENTINEL] * (512 - len(recent)), mx.uint32)
            size = top[row].size + 512
            capacity = 1 << (max(16, size) - 1).bit_length()
            if self.deduplicate:
                state.pool = mlx_vocabulary_set(
                    mx.concatenate([top[row].reshape(-1), required]),
                    self.vocab_size,
                    size,
                )
                state.ids = state.pool[1]
                continue
            # Global-ID selection removes duplicates. Keeping these rows
            # unsorted avoids two extra refresh sorts and a host ID read.
            state.ids = mx.concatenate(
                [
                    top[row].reshape(-1),
                    required,
                    mx.full((capacity - size,), _SENTINEL, mx.uint32),
                ]
            )
        return mlx_topk(logits, k)
