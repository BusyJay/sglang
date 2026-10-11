# SPDX-License-Identifier: Apache-2.0
"""CPU shadow policies for fixed-budget, live expert-cache allocation."""

from __future__ import annotations

from sglang.srt.layers.moe.qwen38_expert_cache import LayerExpertCachePolicy


def allocate_slots(curves, budget):
    """Minimize observed misses over candidate capacities and a slot budget."""
    states = {0: (0, ())}
    for curve in curves:
        next_states = {}
        for used, (misses, choices) in states.items():
            for capacity, cost in curve.items():
                total = used + capacity
                if total > budget:
                    continue
                candidate = (misses + cost, choices + (capacity,))
                if total not in next_states or candidate < next_states[total]:
                    next_states[total] = candidate
        states = next_states
    if not states:
        raise ValueError("slot budget cannot cover minimum per-layer capacity")
    used = min(states, key=lambda total: (states[total][0], total, states[total][1]))
    return list(states[used][1]), states[used][0]


def relocate_slots(moves, move, save, restore):
    """Apply an injective old-slot -> new-slot map with one scratch slot.

    Paths are moved backward into free destinations. Cycles rotate through the
    scratch slot. ``move``, ``save`` and ``restore`` may enqueue ordered CUDA
    copies; no source is overwritten before its value has moved.
    """
    remaining = {src: dst for src, dst in moves.items() if src != dst}
    if len(set(remaining.values())) != len(remaining):
        raise ValueError("adaptive slot destinations must be unique")
    reverse = {dst: src for src, dst in remaining.items()}
    for free in list(set(reverse) - set(remaining)):
        while free in reverse:
            source = reverse.pop(free)
            move(source, free)
            del remaining[source]
            free = source
    while remaining:
        start = next(iter(remaining))
        save(start)
        destination = start
        while True:
            source = reverse[destination]
            del reverse[destination]
            del remaining[source]
            if source == start:
                restore(destination)
                break
            move(source, destination)
            destination = source


class AdaptiveLayerAllocator:
    """Track candidate LFU policies without delaying the GPU transfer worker.

    The caller feeds one-token decode active sets in layer order. A proposal is
    made only after every layer has a complete observation window. Shadow caches
    persist between windows so cost estimates reflect steady-state behavior.
    """

    def __init__(
        self,
        initial_slots,
        num_experts,
        *,
        min_slots=1,
        candidates=None,
        window=512,
        min_gain=0.05,
        cooldown_windows=2,
        payback_windows=8,
        policy="lfu",
        protected_fraction=0.7,
    ):
        self.num_layers = len(initial_slots)
        self.budget = sum(initial_slots)
        self.current = list(initial_slots)
        if candidates is None:
            baseline = min(initial_slots)
            step = 16 if num_experts >= 64 else 1
            lower = max(min_slots, ((baseline // 2 + step - 1) // step) * step)
            upper = min(num_experts, max(initial_slots) + 16)
            candidates = range(lower, upper + 1, step)
        self.candidates = tuple(sorted(set(candidates) | set(initial_slots)))
        if (
            not self.num_layers
            or window < 1
            or not 0 <= min_gain < 1
            or cooldown_windows < 0
            or payback_windows < 1
            or any(not min_slots <= c <= num_experts for c in self.candidates)
        ):
            raise ValueError("invalid adaptive expert-cache configuration")
        self.window = window
        self.min_gain = min_gain
        self.cooldown_windows = cooldown_windows
        self.payback_windows = payback_windows
        self.cooldown = 0
        self.evaluations = 0
        self.last_evaluation = None
        self.seen = [0] * self.num_layers
        self.misses = [{c: 0 for c in self.candidates} for _ in initial_slots]
        self.policies = [
            {
                c: LayerExpertCachePolicy(
                    num_experts, c, policy=policy, protected_fraction=protected_fraction
                )
                for c in self.candidates
            }
            for _ in initial_slots
        ]
        self.pending = None

    def observe(self, layer, active):
        if not 0 <= layer < self.num_layers:
            raise ValueError("invalid expert-cache layer")
        active = sorted(set(active))
        for capacity in self.candidates:
            _, missing = self.policies[layer][capacity].ensure(active)
            self.misses[layer][capacity] += len(missing)
        self.seen[layer] += 1
        if min(self.seen) < self.window:
            return None
        baseline = sum(
            curve[capacity] for curve, capacity in zip(self.misses, self.current)
        )
        proposed, predicted = allocate_slots(self.misses, self.budget)
        cold_refill = sum(max(0, old - new) for old, new in zip(self.current, proposed))
        self.last_evaluation = {
            "baseline_misses": baseline,
            "proposed_misses": predicted,
            "evicted_slots": cold_refill,
            "proposed_slots": proposed,
        }
        self.evaluations += 1
        self.seen = [0] * self.num_layers
        self.misses = [{c: 0 for c in self.candidates} for _ in self.current]
        if self.cooldown:
            self.cooldown -= 1
            return None
        # Shrinking layers evicts some residents. Require enough avoided misses
        # to pay for that refill, as well as a relative improvement.
        if (
            proposed != self.current
            and (baseline - predicted) * self.payback_windows >= cold_refill
            and baseline - predicted >= baseline * self.min_gain
        ):
            self.pending = proposed
            self.cooldown = self.cooldown_windows
            return proposed
        return None

    def applied(self, layout):
        self.current = list(layout)
        if self.pending == layout:
            self.pending = None
