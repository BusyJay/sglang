# SPDX-License-Identifier: Apache-2.0
"""Transactional MTP lookahead for the ordinary MLX serving pipeline.

Only a consumed row is published to request caches. Future rows retain private
snapshots and raw logits, so the normal sampler and scheduler still process
one token at a time, including grammar edits, logprobs, stopping and release.
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

logger = logging.getLogger(__name__)


def clone_cache(cache):
    result = NativeCache(4)
    # A distinct array object protects slice assignment; view shares storage.
    # mx.array(value) instead copies every multi-MiB recurrent state.
    result.cache = [v.view(v.dtype) if v is not None else None for v in cache.cache]
    result.offset = cache.offset
    return result


class AdaptiveMTPBudget:
    """Explore bounded budgets, then favor measured committed-token throughput.

    Scores include rejected work and the serving time between consumed rows.
    Periodic exploration revisits neighbors as context and workload change.
    Truncated final cycles do not bias a budget's score.
    """

    candidates = (2, 1, 3, 4, 6, 8, 0)

    def __init__(self):
        self.scores = {}
        self.cycle_ct = 0

    def choose(self):
        for steps in self.candidates:
            if steps not in self.scores:
                return steps
        best = max(self.scores, key=self.scores.get)
        if self.cycle_ct % 16 == 0:
            ordered = sorted(self.candidates)
            index = ordered.index(best)
            neighbors = (
                ordered[max(0, index - 1) : index] + ordered[index + 1 : index + 2]
            )
            return neighbors[(self.cycle_ct // 16) % len(neighbors)]
        return best

    def observe(self, steps, num_tokens, elapsed):
        if steps not in self.candidates or num_tokens < 1 or elapsed <= 0:
            return
        score = num_tokens / elapsed
        previous = self.scores.get(steps, score)
        self.scores[steps] = 0.5 * previous + 0.5 * score
        self.cycle_ct += 1


@dataclass
class VerifiedRow:
    input_token: int
    logits: mx.array
    hidden: mx.array
    cache: list
    depth: int


@dataclass
class RequestState:
    head: NativeCache
    anchor: object = None
    hidden: mx.array | None = None
    offset: int = 0
    rows: deque = field(default_factory=deque)
    pending_hidden: list = field(default_factory=list)
    pending_tokens: list = field(default_factory=list)
    remaining: int | None = None
    cycle_start: float = 0.0
    cycle_steps: int = 0
    cycle_tokens: int = 0
    # Do not train on a budget clipped by the remaining output allowance.
    train_cycle: bool = True


class MlxMTPServing:
    def __init__(self, model, num_steps):
        if model.mtp is None:
            raise ValueError("MLX MTP serving requires loaded prediction weights")
        if not -1 <= num_steps <= 16:
            raise ValueError("mlx_mtp_num_steps must be -1 (auto), or 0-16")
        self.model = model
        self.num_steps = num_steps
        self.states = {}
        self.budget = AdaptiveMTPBudget()
        self.step_histogram = Counter()
        self.verify_ct = 0
        self.num_reuse_tokens = 0
        self.num_proposed_drafts = 0
        self.eligible_by_depth = Counter()
        self.correct_by_depth = Counter()
        self.proposed_by_depth = Counter()
        self.eligible_by_budget_depth = Counter()
        self.correct_by_budget_depth = Counter()

    def _state(self, rid, cache):
        state = self.states.get(rid)
        offset = cache[0].offset
        if state is not None and state.anchor is cache[0] and state.offset == offset:
            return state
        state = RequestState(head=NativeCache(4), anchor=cache[0], offset=offset)
        saved = getattr(cache[0], "mlx_mtp", None)
        if saved is not None and saved[3] == offset:
            arrays, head_offset, hidden, _, pending_h, pending_t = saved
            state.head.cache = [
                v.view(v.dtype) if v is not None else None for v in arrays
            ]
            state.head.offset = head_offset
            state.hidden = hidden
            state.pending_hidden = list(pending_h)
            state.pending_tokens = list(pending_t)
        self.states[rid] = state
        return state

    @staticmethod
    def _publish(state, cache):
        # The auxiliary-state pool clones this tuple along with the target's
        # recurrent/latent caches. Never include speculative rows here.
        cache[0].mlx_mtp = (
            tuple(v.view(v.dtype) if v is not None else None for v in state.head.cache),
            state.head.offset,
            state.hidden,
            state.offset,
            tuple(state.pending_hidden),
            tuple(state.pending_tokens),
        )

    def set_remaining(self, rid, remaining):
        state = self.states.get(rid)
        if state is not None:
            state.remaining = max(0, remaining)

    def _flush_head(self, state):
        if state.pending_hidden:
            out = self.model.mtp(
                mx.concatenate(state.pending_hidden, axis=1),
                self.model.model.embed_tokens(
                    mx.concatenate(state.pending_tokens, axis=1)
                ),
                state.head,
            )
            mx.async_eval(out, state.head.cache)
            state.pending_hidden.clear()
            state.pending_tokens.clear()

    def prefill(self, rid, inputs, cache):
        state = self._state(rid, cache)
        state.rows.clear()
        state.cycle_start = 0.0
        self._flush_head(state)
        hidden = self.model.model(inputs, cache)
        pairs_h, pairs_t = [], []
        if state.hidden is not None:
            pairs_h.append(state.hidden)
            pairs_t.append(inputs[:, :1])
        if inputs.shape[1] > 1:
            pairs_h.append(hidden[:, :-1])
            pairs_t.append(inputs[:, 1:])
        if pairs_h:
            out = self.model.mtp(
                mx.concatenate(pairs_h, axis=1),
                self.model.model.embed_tokens(mx.concatenate(pairs_t, axis=1)),
                state.head,
            )
            mx.async_eval(out, state.head.cache)
        state.hidden = hidden[:, -1:]
        state.offset = cache[0].offset
        self._publish(state, cache)
        return hidden

    def _close_cycle(self, state):
        if state.cycle_start and state.train_cycle:
            self.budget.observe(
                state.cycle_steps,
                state.cycle_tokens,
                perf_counter() - state.cycle_start,
            )
        state.cycle_start = 0.0

    def decode(self, rid, inputs, cache):
        state = self._state(rid, cache)
        token = inputs.item()
        if state.rows:
            row = state.rows[0]
            key = f"{state.cycle_steps}:{row.depth}"
            self.eligible_by_depth[row.depth] += 1
            self.eligible_by_budget_depth[key] += 1
            if row.input_token == token:
                self.correct_by_depth[row.depth] += 1
                self.correct_by_budget_depth[key] += 1
            else:
                state.rows.clear()
        if not state.rows:
            self._close_cycle(state)
            steps = self.budget.choose() if self.num_steps == -1 else self.num_steps
            unclipped = steps
            if state.remaining is not None:
                steps = min(steps, max(0, state.remaining - 1))
            state.cycle_start = perf_counter()
            state.cycle_steps = steps
            state.cycle_tokens = 0
            state.train_cycle = steps == unclipped
            if state.hidden is None or steps == 0:
                logits, hidden = self.model.forward_with_hidden(inputs, cache)
                if state.hidden is not None:
                    state.pending_hidden.append(state.hidden)
                    state.pending_tokens.append(inputs)
                state.hidden = hidden[:, -1:]
                state.offset = cache[0].offset
                state.cycle_tokens = 1
                self.step_histogram[0] += 1
                self._publish(state, cache)
                return logits[:, -1]
            self._flush_head(state)
            out = self.model.mtp(
                state.hidden, self.model.model.embed_tokens(inputs), state.head
            )
            scratch_head = clone_cache(state.head)
            chain = [inputs]
            for depth in range(steps):
                draft = self.model.draft_token(out)
                chain.append(draft)
                if depth + 1 < steps:
                    out = self.model.mtp(
                        out, self.model.model.embed_tokens(draft), scratch_head
                    )
            chain = mx.concatenate(chain, axis=1)
            scratch = [clone_cache(c) for c in cache]
            logits, hidden, snapshots = self.model.verify_mtp_rows(chain, scratch)
            mx.eval(chain, logits, hidden)
            tokens = chain[0].tolist()
            state.rows.extend(
                VerifiedRow(t, logits[:, j], hidden[:, j : j + 1], snapshots[j], j)
                for j, t in enumerate(tokens)
            )
            self.step_histogram[steps] += 1
            self.verify_ct += 1
            self.num_proposed_drafts += steps
            self.proposed_by_depth.update(range(1, steps + 1))
            root_advanced = True
        else:
            self.num_reuse_tokens += 1
            root_advanced = False
        row = state.rows.popleft()
        if not root_advanced:
            out = self.model.mtp(
                state.hidden, self.model.model.embed_tokens(inputs), state.head
            )
            mx.async_eval(out, state.head.cache)
        for dst, src in zip(cache, row.cache):
            dst.cache = [v.view(v.dtype) if v is not None else None for v in src.cache]
            dst.offset = src.offset
        state.hidden = row.hidden
        state.offset = cache[0].offset
        state.cycle_tokens += 1
        self._publish(state, cache)
        return row.logits

    def remove(self, rid):
        state = self.states.pop(rid, None)
        if state is not None:
            self._close_cycle(state)
        return state is not None

    def clear(self):
        self.states.clear()

    def stats(self):
        return dict(
            verify_ct=self.verify_ct,
            num_proposed_drafts=self.num_proposed_drafts,
            num_reuse_tokens=self.num_reuse_tokens,
            step_histogram=dict(self.step_histogram),
            adaptive_scores=dict(self.budget.scores),
            proposed_by_depth=dict(self.proposed_by_depth),
            eligible_by_depth=dict(self.eligible_by_depth),
            correct_by_depth=dict(self.correct_by_depth),
            eligible_by_budget_depth=dict(self.eligible_by_budget_depth),
            correct_by_budget_depth=dict(self.correct_by_budget_depth),
        )
