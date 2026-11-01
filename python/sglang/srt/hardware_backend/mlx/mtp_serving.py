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
    result = type(cache)(4)
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
    input_token: int | mx.array
    logits: mx.array
    hidden: mx.array
    cache: list
    depth: int
    token_only: bool = False


@dataclass
class PendingMTPDecode:
    state: "RequestState"
    inputs: mx.array
    cache: list
    chain: mx.array | None = None
    logits: mx.array | None = None
    hidden: mx.array | None = None
    snapshots: list | None = None
    draft_steps: int = 0
    token_only: bool = False


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
            out = self.model.mtp.consume_rows(
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

    def build_decode(
        self, rid, inputs, cache, token, *,
        defer_verification=False, token_only=False,
    ):
        """Build private speculation; caller supplies the already sampled input ID."""
        state = self._state(rid, cache)
        if state.rows and state.rows[0].token_only != token_only:
            state.rows.clear()
        if state.rows:
            row = state.rows[0]
            key = f"{state.cycle_steps}:{row.depth}"
            self.eligible_by_depth[row.depth] += 1
            self.eligible_by_budget_depth[key] += 1
            expected_token = (
                int(row.input_token.item())
                if isinstance(row.input_token, mx.array)
                else row.input_token
            )
            if expected_token == token:
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
                logits, hidden = (self.model.forward_with_hidden(inputs, cache, token_only=True)
                                  if token_only else self.model.forward_with_hidden(inputs, cache))
                if state.hidden is not None:
                    state.pending_hidden.append(state.hidden)
                    state.pending_tokens.append(inputs)
                self.step_histogram[0] += 1
                return PendingMTPDecode(
                    state,
                    inputs,
                    cache,
                    logits=logits[:, -1:] if token_only else logits[:, -1],
                    hidden=hidden[:, -1:],
                    token_only=token_only,
                )
            self._flush_head(state)
            if defer_verification and hasattr(self.model, "draft_tokens"):
                self.step_histogram[steps] += 1
                self.verify_ct += 1
                self.num_proposed_drafts += steps
                self.proposed_by_depth.update(range(1, steps + 1))
                return PendingMTPDecode(state, inputs, cache, draft_steps=steps, token_only=token_only)
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
            if defer_verification:
                logits, hidden, snapshots = None, None, []
            else:
                scratch = [clone_cache(c) for c in cache]
                logits, hidden, snapshots = (self.model.verify_mtp_rows(chain, scratch, token_only=True)
                                             if token_only else self.model.verify_mtp_rows(chain, scratch))
            self.step_histogram[steps] += 1
            self.verify_ct += 1
            self.num_proposed_drafts += steps
            self.proposed_by_depth.update(range(1, steps + 1))
            return PendingMTPDecode(
                state, inputs, cache, chain, logits, hidden, snapshots, token_only=token_only
            )
        self.num_reuse_tokens += 1
        # The head is only read before drafting the next cycle. Retain the
        # consumed pairs now; flush them together when drafting or extending.
        # Prefix snapshots carry these pairs, so release/reject never publish
        # a head advanced through an unconsumed target row.
        state.pending_hidden.append(state.hidden)
        state.pending_tokens.append(inputs)
        return PendingMTPDecode(state, inputs, cache, token_only=token_only)

    def publish_decode(self, pending):
        """Publish only the consumed cache boundary after batch execution."""
        state, cache = pending.state, pending.cache
        if pending.chain is not None:
            # Keep future token IDs lazy. Only the next consumption boundary
            # materializes the scalar needed to accept/reject that private row.
            tokens = [pending.chain[0, j] for j in range(pending.chain.shape[1])]
            state.rows.extend(
                VerifiedRow(
                    t,
                    pending.logits[:, j:j+1] if pending.token_only else pending.logits[:, j],
                    pending.hidden[:, j : j + 1],
                    pending.snapshots[j],
                    j,
                    pending.token_only,
                )
                for j, t in enumerate(tokens)
            )
        elif pending.logits is not None:
            state.hidden = pending.hidden
            state.offset = cache[0].offset
            state.cycle_tokens = 1
            self._publish(state, cache)
            return pending.logits
        row = state.rows.popleft()
        for dst, src in zip(cache, row.cache):
            dst.cache = [v.view(v.dtype) if v is not None else None for v in src.cache]
            dst.offset = src.offset
        state.hidden = row.hidden
        state.offset = cache[0].offset
        state.cycle_tokens += 1
        self._publish(state, cache)
        return row.logits

    def execute_batch(self, pending):
        # Submit complete private transactions together. Array handles and
        # snapshots retain ownership until sampling/consumption materializes
        # them; publication never reads unfinished cache contents on the host.
        drafting = [p for p in pending if p.draft_steps]
        if drafting:
            chains = [[p.inputs] for p in drafting]
            outputs = self.model.mtp.decode_requests(
                [p.state.hidden for p in drafting],
                [self.model.model.embed_tokens(p.inputs) for p in drafting],
                [p.state.head for p in drafting],
            )
            scratch_heads = [clone_cache(p.state.head) for p in drafting]
            active = list(range(len(drafting)))
            for depth in range(max(p.draft_steps for p in drafting)):
                tokens = self.model.draft_tokens([outputs[i] for i in active])
                for i, token in zip(active, tokens):
                    chains[i].append(token)
                continuing = [
                    (i, token)
                    for i, token in zip(active, tokens)
                    if depth + 1 < drafting[i].draft_steps
                ]
                if continuing:
                    values = self.model.mtp.decode_requests(
                        [outputs[i] for i, _ in continuing],
                        [
                            self.model.model.embed_tokens(token)
                            for _, token in continuing
                        ],
                        [scratch_heads[i] for i, _ in continuing],
                    )
                    for (i, _), value in zip(continuing, values):
                        outputs[i] = value
                active = [i for i, _ in continuing]
            for p, chain in zip(drafting, chains):
                p.chain = mx.concatenate(chain, 1)
        verifying = [p for p in pending if p.chain is not None]
        deferred = [p for p in verifying if p.logits is None]
        if deferred:
            # A batch may contain requests with different output capabilities.
            # Verify each capability group explicitly, preserving request order.
            for token_only in (False, True):
                group = [p for p in deferred if p.token_only == token_only]
                if not group:
                    continue
                scratch = [[clone_cache(c) for c in p.cache] for p in group]
                results = (
                    self.model.verify_mtp_requests([p.chain for p in group], scratch, token_only=True)
                    if token_only else self.model.verify_mtp_requests([p.chain for p in group], scratch)
                )
                for transaction, (logits, hidden, snapshots) in zip(group, results):
                    transaction.logits, transaction.hidden, transaction.snapshots = logits, hidden, snapshots
        if verifying:
            mx.async_eval(
                [p.chain for p in verifying],
                [p.logits for p in verifying if p.logits is not None],
                [p.hidden for p in verifying if p.hidden is not None],
                [[c.cache for row in p.snapshots for c in row] for p in verifying],
            )
        return [self.publish_decode(p) for p in pending]

    def decode(self, rid, inputs, cache):
        pending = self.build_decode(rid, inputs, cache, inputs.item())
        return self.execute_batch([pending])[0]

    def decode_batch(self, req_ids, inputs, caches, tokens, *, token_only=False):
        pending = [
            self.build_decode(
                rid,
                inputs[i : i + 1],
                caches[i],
                tokens[i],
                defer_verification=hasattr(self.model, "verify_mtp_requests"),
                token_only=token_only,
            )
            for i, rid in enumerate(req_ids)
        ]
        return mx.concatenate(self.execute_batch(pending), axis=0)

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
