# SPDX-License-Identifier: Apache-2.0
"""Transactional DFlash2 drafting through MLX's ordinary sampler and caches.

Only consumed target features enter the drafter context. Verified future rows
remain private, so sampling edits, stopping, abort and prefix reuse use the
same target distribution and cache boundaries as ordinary decode.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter

import mlx.core as mx
from sglang.srt.hardware_backend.mlx.context_copy import ContextCopyProposer
from sglang.srt.hardware_backend.mlx.dflash2 import load_dflash2
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache


@dataclass
class DFlashRow:
    input_token: mx.array
    logits: mx.array
    features: mx.array
    cache: list
    depth: int
    token_only: bool


@dataclass
class PendingDFlashRound:
    offset: int
    token_only: bool
    chain: mx.array
    result: tuple
    draft_state: object
    num_copy_drafts: int = 0


@dataclass
class DFlashRequest:
    draft_state: object
    anchor: object
    offset: int
    pending_features: list = field(default_factory=list)
    pending_positions: list = field(default_factory=list)
    rows: deque = field(default_factory=deque)
    remaining: int | None = None
    chain_tokens: mx.array | None = None
    input_tokens: tuple[int, ...] | None = None
    copy_proposer: ContextCopyProposer = field(default_factory=ContextCopyProposer)
    prefetch: PendingDFlashRound | None = None


class MlxDFlashServing:
    def __init__(self, model, draft_path, num_drafts=None, *, drafter=None):
        self.model = model
        self.drafter = (
            drafter if drafter is not None else load_dflash2(draft_path, model)
        )
        self.capture_layer_ids = tuple(self.drafter.target_layer_ids)
        # Full-model M5 A/B favors three proposals over five/seven; the draft
        # backbone still computes every trained slot before clipping output.
        self.num_drafts = (
            min(3, self.drafter.block_size - 1)
            if num_drafts is None
            else int(num_drafts)
        )
        if not 0 <= self.num_drafts < self.drafter.block_size:
            raise ValueError("DFlash draft count must fit the checkpoint's block size")
        self.states = {}
        self.counters = Counter()
        self.correct_by_depth = Counter()
        self.eligible_by_depth = Counter()
        self.timings = Counter()

    def _state(self, rid, cache):
        state = self.states.get(rid)
        offset = cache[0].offset
        if state is not None and state.anchor is cache[0] and state.offset == offset:
            return state
        saved = getattr(cache[0], "mlx_dflash", None)
        if (
            saved is not None
            and saved[0] == offset
            and saved[1] == self.capture_layer_ids
        ):
            state = DFlashRequest(
                saved[2],
                cache[0],
                offset,
                list(saved[3]),
                list(saved[4]),
                copy_proposer=ContextCopyProposer(
                    getattr(cache[0], "mlx_context_copy", None)
                ),
            )
        else:
            if offset:
                raise RuntimeError(
                    "A reused GLM prefix must retain its DFlash feature context"
                )
            state = DFlashRequest(self.drafter.make_state(), cache[0], offset)
        self.states[rid] = state
        return state

    def _publish(self, state, cache):
        cache[0].mlx_dflash = (
            state.offset,
            self.capture_layer_ids,
            state.draft_state,
            tuple(state.pending_features),
            tuple(state.pending_positions),
        )
        cache[0].mlx_context_copy = state.copy_proposer.snapshot

    def _consume_features(self, state):
        if state.pending_features:
            features = mx.concatenate(state.pending_features, 1)
            positions = mx.concatenate(state.pending_positions)
            state.draft_state = self.drafter.consume(
                features, positions, state.draft_state
            )
            state.pending_features.clear()
            state.pending_positions.clear()

    def prefill(self, rid, inputs, cache, *, inputs_embeds=None):
        state = self._state(rid, cache)
        state.rows.clear()
        state.prefetch = None
        offset = state.offset
        hidden, features = self.model.model(
            inputs,
            cache,
            capture_layer_ids=self.capture_layer_ids,
            inputs_embeds=inputs_embeds,
        )
        state.pending_features.append(features)
        state.pending_positions.append(
            mx.arange(offset, offset + inputs.shape[1], dtype=mx.int32)
        )
        # Bound prefill graphs and retained features to a chunk; no full-context
        # feature materialization survives into the request's prefix snapshot.
        self._consume_features(state)
        mx.async_eval(state.draft_state)
        state.offset = cache[0].offset
        state.copy_proposer.extend(inputs.reshape(-1).tolist())
        self._publish(state, cache)
        return hidden

    def set_remaining(self, rid, remaining):
        state = self.states.get(rid)
        if state is not None:
            state.remaining = max(0, remaining)

    def _install_rows(self, state, chain, result, token_only):
        logits, _hidden, snapshots, features = result
        state.chain_tokens = chain
        state.input_tokens = None
        state.rows.extend(
            DFlashRow(
                chain[0, j],
                logits[:, j : j + 1] if token_only else logits[:, j],
                features[:, j : j + 1],
                snapshots[j],
                j,
                token_only,
            )
            for j in range(chain.shape[1])
        )

    def _prefetch_round(self, state, cache, row, token_only):
        """Submit one private round while the CPU reads the committed output."""
        count = self.num_drafts
        if state.remaining is not None:
            count = min(count, max(0, state.remaining - 2))
        if count <= 0:
            return
        anchor = (
            row.logits.reshape(1, 1).astype(mx.int32)
            if token_only
            else mx.argmax(row.logits, -1).reshape(1, 1).astype(mx.int32)
        )
        # The last committed row is now official, but consuming its features
        # remains private until the actual sampler confirms the next anchor.
        prepared = self.drafter.consume(
            mx.concatenate(state.pending_features, 1),
            mx.concatenate(state.pending_positions),
            state.draft_state,
        )
        copied = ()
        if state.copy_proposer.can_match_next():
            copied = state.copy_proposer.propose(int(anchor.item()), count)
        drafts = (
            mx.array([copied], anchor.dtype)
            if copied
            else self.drafter.draft(
                anchor,
                [prepared],
                self.model.model.embed_tokens,
                self.model.lm_head,
                count,
            )
        )
        chain = mx.concatenate([anchor, drafts], 1)
        private = [clone_cache(c) for c in cache]
        result = self.model.verify_mtp_requests(
            [chain],
            [private],
            token_only=token_only,
            capture_layer_ids=self.capture_layer_ids,
        )[0]
        state.prefetch = PendingDFlashRound(
            state.offset, token_only, chain, result, prepared, len(copied)
        )
        mx.async_eval(
            chain,
            result[0],
            result[3],
            [c.cache for boundary in result[2] for c in boundary],
        )
        self.counters["prefetch_ct"] += 1

    def decode_batch(
        self,
        req_ids,
        inputs,
        caches,
        tokens,
        *,
        token_only=False,
        prefetch_allowed=False,
    ):
        started = perf_counter()
        requests, work = [], []
        for i, (rid, cache, token) in enumerate(zip(req_ids, caches, tokens)):
            state = self._state(rid, cache)
            if state.rows and state.rows[0].token_only != token_only:
                state.rows.clear()
            if state.rows:
                row = state.rows[0]
                self.eligible_by_depth[row.depth] += 1
                # Verification already depends on the complete chain. Read its
                # token vector once per chain, avoiding a fresh scalar slice
                # evaluation and device readback at every consumed draft row.
                if state.input_tokens is None:
                    state.input_tokens = tuple(state.chain_tokens.reshape(-1).tolist())
                if state.input_tokens[row.depth] != token:
                    state.rows.clear()
                    self.counters["reject_ct"] += 1
                else:
                    self.correct_by_depth[row.depth] += 1
                    self.counters["num_correct_drafts"] += 1
            requests.append((state, cache))
            if not state.rows:
                count = self.num_drafts
                if state.remaining is not None:
                    count = min(count, max(0, state.remaining - 1))
                prefetched = state.prefetch
                state.prefetch = None
                if (
                    prefetched is not None
                    and prefetched.offset == state.offset
                    and prefetched.token_only == token_only
                    and prefetched.chain.shape[1] == count + 1
                    and int(prefetched.chain[0, 0].item()) == token
                ):
                    state.draft_state = prefetched.draft_state
                    state.pending_features.clear()
                    state.pending_positions.clear()
                    self._install_rows(
                        state, prefetched.chain, prefetched.result, token_only
                    )
                    self.counters["prefetch_use_ct"] += 1
                    if prefetched.num_copy_drafts:
                        self.counters["copy_verify_ct"] += 1
                        self.counters["num_copy_drafts"] += prefetched.num_copy_drafts
                    self.counters["verify_ct"] += 1
                    self.counters["num_proposed_drafts"] += count
                else:
                    if prefetched is not None:
                        self.counters["prefetch_discard_ct"] += 1
                    self._consume_features(state)
                    work.append((i, count))
        self.timings["host_prepare_seconds"] += perf_counter() - started
        # Share all draft projections within cohorts; different target context
        # lengths are represented by each request's explicit context mask.
        chains = {}
        started = perf_counter()
        draft_work = []
        for i, count in work:
            copied = requests[i][0].copy_proposer.propose(tokens[i], count)
            if copied:
                chains[i] = mx.concatenate(
                    [inputs[i : i + 1], mx.array([copied], inputs.dtype)], 1
                )
                self.counters["copy_verify_ct"] += 1
                self.counters["num_copy_drafts"] += len(copied)
                self.counters["verify_ct"] += 1
                self.counters["num_proposed_drafts"] += count
            else:
                draft_work.append((i, count))
        for count in sorted({count for _, count in draft_work}):
            indices = [i for i, n in draft_work if n == count]
            anchors = mx.concatenate([inputs[i : i + 1] for i in indices], 0)
            if count:
                drafts = self.drafter.draft(
                    anchors,
                    [requests[i][0].draft_state for i in indices],
                    self.model.model.embed_tokens,
                    self.model.lm_head,
                    count,
                )
                values = mx.concatenate([anchors, drafts], 1)
            else:
                values = anchors
            for j, i in enumerate(indices):
                chains[i] = values[j : j + 1]
                self.counters["verify_ct"] += 1
                self.counters["num_proposed_drafts"] += count
        self.timings["host_draft_build_seconds"] += perf_counter() - started
        if work:
            started = perf_counter()
            indices = [i for i, _ in work]
            private = [[clone_cache(c) for c in caches[i]] for i in indices]
            results = self.model.verify_mtp_requests(
                [chains[i] for i in indices],
                private,
                token_only=token_only,
                capture_layer_ids=self.capture_layer_ids,
            )
            for i, result in zip(indices, results):
                self._install_rows(requests[i][0], chains[i], result, token_only)
            mx.async_eval(
                [r[0] for r in results],
                [r[3] for r in results],
                [[c.cache for row in r[2] for c in row] for r in results],
            )
            self.timings["host_verify_build_seconds"] += perf_counter() - started
        outputs = []
        for i, (state, cache) in enumerate(requests):
            row = state.rows.popleft()
            offset = state.offset
            for dst, src in zip(cache, row.cache):
                dst.cache = [
                    v.view(v.dtype) if v is not None else None for v in src.cache
                ]
                dst.offset = src.offset
            state.pending_features.append(row.features)
            state.pending_positions.append(mx.array([offset], mx.int32))
            state.offset = cache[0].offset
            state.copy_proposer.extend((tokens[i],))
            self._publish(state, cache)
            outputs.append(row.logits)
            self.counters["num_accept_tokens"] += 1
            if prefetch_allowed and not state.rows:
                started = perf_counter()
                self._prefetch_round(state, cache, row, token_only)
                self.timings["host_prefetch_build_seconds"] += perf_counter() - started
        return mx.concatenate(outputs, 0)

    def decode(self, rid, inputs, cache):
        return self.decode_batch([rid], inputs, [cache], [int(inputs.item())])[0:1]

    def remove(self, rid):
        return self.states.pop(rid, None) is not None

    def clear(self):
        self.states.clear()

    def stats(self):
        return dict(
            self.counters,
            correct_by_depth=dict(self.correct_by_depth),
            eligible_by_depth=dict(self.eligible_by_depth),
            timings=dict(self.timings),
            num_drafts=self.num_drafts,
        )
