# SPDX-License-Identifier: Apache-2.0
"""Pure policy: shared slots, per-layer immutable compute windows, deferred LFU."""

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field


class HybridCostModel:
    """Select a measured full branch for an exact route workload.

    Calibration compares elapsed time after the route decision until the GPU
    output is ready, with identical geometry and routes for both branches.
    ``cpu_us`` measures the complete mixed CPU/GPU branch, including mapped
    request writes, worker execution, GPU work, the completion wait, mapped
    result reads, and merge. ``gpu_us`` measures the all-GPU branch for the
    same workload, including registered expert fills and GPU compute. Branch
    elapsed times include overlap; individual phase durations are not added.
    In sample keys, ``hit_routes`` counts all GPU
    routes, including misses that will be admitted or filled on GPU;
    ``miss_routes`` and ``miss_experts`` count only the selected CPU subset.
    Kernel-only timings are not comparable calibration samples. ``cores`` is
    the physical CPU core count, not the number of logical threads.
    ``miss_tokens`` records sorted route counts per selected CPU expert, so
    different distributions with the same total cannot share measurements.
    ``rejected_tokens`` records sorted route counts for all LFU-rejected
    experts before CPU selection, distinguishing GPU cache routes from
    rejected routes that still require GPU transfers.
    Optional transport and runtime version metadata must match exactly when
    supplied by either the runtime or the calibration table.
    """

    def __init__(
        self,
        calibration,
        *,
        backend,
        hidden,
        width,
        top_k,
        cores,
        gpu,
        cpu,
        transport=None,
        runtime_version=None,
    ):
        self.margin = 0.1
        self.samples = {}
        self.top_k = top_k
        if calibration is None:
            return
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be a JSON object")
        if (
            type(calibration.get("schema_version")) is not int
            or calibration["schema_version"] != 1
        ):
            raise ValueError("calibration.schema_version must be 1")

        metadata = dict(
            backend=backend,
            hidden=hidden,
            width=width,
            top_k=top_k,
            cores=cores,
            gpu=gpu,
            cpu=cpu,
        )
        for name, expected in (
            ("transport", transport),
            ("runtime_version", runtime_version),
        ):
            if expected is not None or name in calibration:
                metadata[name] = expected
        for name, expected in metadata.items():
            actual = calibration.get(name)
            if name in ("hidden", "width", "top_k", "cores"):
                valid = type(expected) is int and expected > 0
            else:
                valid = type(expected) is str and bool(expected.strip())
            if not valid:
                raise ValueError(f"invalid runtime metadata: {name}")
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"calibration.{name} must exactly match runtime metadata "
                    f"{expected!r}, got {actual!r}"
                )

        margin = self._number(calibration.get("margin", 0.1), "margin")
        if not 0 <= margin < 1:
            raise ValueError("calibration.margin must be in [0, 1)")
        self.margin = margin
        samples = calibration.get("samples")
        if not isinstance(samples, list):
            raise ValueError("calibration.samples must be a list")
        for index, sample in enumerate(samples):
            path = f"samples[{index}]"
            if not isinstance(sample, dict):
                raise ValueError(f"calibration.{path} must be a JSON object")
            key = tuple(
                sample.get(name)
                for name in ("tokens", "hit_routes", "miss_routes", "miss_experts")
            )
            if any(type(value) is not int for value in key):
                raise ValueError(f"calibration.{path} route counts must be integers")
            tokens, hit_routes, miss_routes, miss_experts = key
            if not 1 <= tokens <= 3:
                raise ValueError(f"calibration.{path}.tokens must be in [1, 3]")
            if hit_routes < 0:
                raise ValueError(f"calibration.{path}.hit_routes must be >= 0")
            if miss_routes < 1:
                raise ValueError(f"calibration.{path}.miss_routes must be >= 1")
            if not 1 <= miss_experts <= miss_routes:
                raise ValueError(
                    f"calibration.{path}.miss_experts must be in [1, miss_routes]"
                )
            if hit_routes + miss_routes > tokens * top_k:
                raise ValueError(
                    f"calibration.{path} total routes exceed tokens * top_k"
                )
            miss_tokens = sample.get("miss_tokens")
            if not isinstance(miss_tokens, list):
                raise ValueError(
                    f"calibration.{path}.miss_tokens must be a sorted list"
                )
            if (
                len(miss_tokens) != miss_experts
                or any(
                    type(count) is not int or not 1 <= count <= tokens * top_k
                    for count in miss_tokens
                )
                or sum(miss_tokens) != miss_routes
                or miss_tokens != sorted(miss_tokens)
            ):
                raise ValueError(
                    f"calibration.{path}.miss_tokens must contain miss_experts "
                    "sorted integer counts in [1, tokens * top_k], "
                    "summing to miss_routes"
                )
            rejected_tokens = sample.get("rejected_tokens")
            if (
                not isinstance(rejected_tokens, list)
                or any(
                    type(count) is not int or not 1 <= count <= tokens * top_k
                    for count in rejected_tokens
                )
                or rejected_tokens != sorted(rejected_tokens)
                or sum(rejected_tokens) > hit_routes + miss_routes
                or Counter(miss_tokens) - Counter(rejected_tokens)
            ):
                raise ValueError(
                    f"calibration.{path}.rejected_tokens must be sorted positive "
                    "integer counts within total routes, containing miss_tokens "
                    "as a multiset subset"
                )
            key += (tuple(miss_tokens), tuple(rejected_tokens))
            cpu_us = self._number(sample.get("cpu_us"), f"{path}.cpu_us")
            gpu_us = self._number(sample.get("gpu_us"), f"{path}.gpu_us")
            if cpu_us <= 0 or gpu_us <= 0:
                raise ValueError(f"calibration.{path} timings must be positive")
            if key in self.samples:
                raise ValueError(f"calibration.{path} duplicates workload {key}")
            self.samples[key] = (cpu_us, gpu_us)

    @staticmethod
    def _number(value, name):
        if type(value) not in (int, float):
            raise ValueError(f"calibration.{name} must be a finite number")
        try:
            value = float(value)
        except OverflowError:
            raise ValueError(f"calibration.{name} must be a finite number") from None
        if not math.isfinite(value):
            raise ValueError(f"calibration.{name} must be a finite number")
        return value

    def choose(
        self,
        tokens,
        hit_routes,
        miss_routes,
        miss_experts,
        miss_tokens=None,
        rejected_tokens=None,
    ):
        """Compare exact costs; omitted rejected profile equals the CPU profile."""
        if (
            any(
                type(value) is not int
                for value in (tokens, hit_routes, miss_routes, miss_experts)
            )
            or not 1 <= tokens <= 3
            or hit_routes < 0
            or not 0 <= miss_experts <= miss_routes
            or hit_routes + miss_routes > tokens * self.top_k
        ):
            return "gpu"
        if miss_routes == 0:
            return "hit"
        if miss_tokens is None:
            return "gpu"
        try:
            profile = tuple(sorted(miss_tokens))
        except TypeError:
            return "gpu"
        if (
            len(profile) != miss_experts
            or any(
                type(count) is not int or not 1 <= count <= tokens * self.top_k
                for count in profile
            )
            or sum(profile) != miss_routes
        ):
            return "gpu"
        try:
            rejected_profile = (
                profile if rejected_tokens is None else tuple(sorted(rejected_tokens))
            )
        except TypeError:
            return "gpu"
        if (
            any(
                type(count) is not int or not 1 <= count <= tokens * self.top_k
                for count in rejected_profile
            )
            or sum(rejected_profile) > hit_routes + miss_routes
            or Counter(profile) - Counter(rejected_profile)
        ):
            return "gpu"
        sample = self.samples.get(
            (tokens, hit_routes, miss_routes, miss_experts, profile, rejected_profile)
        )
        if sample is not None:
            cpu_us, gpu_us = sample
            if cpu_us < gpu_us * (1 - self.margin):
                return "cpu"
        return "gpu"

    def select_cpu(self, tokens, rows, mapping, rejected):
        """Choose the fastest eligible measured prefix of LFU-rejected experts.

        Prefixes sort by (route count, expert id). Hits and admitted misses
        always stay on GPU. Missing calibration or invalid input selects no
        CPU experts; negative route ids are padding and do not count.
        """
        empty = frozenset()
        if type(tokens) is not int or not 1 <= tokens <= 3:
            return empty
        if not isinstance(mapping, dict) or any(
            type(expert) is not int or expert < 0 or type(slot) is not int or slot < 0
            for expert, slot in mapping.items()
        ):
            return empty
        try:
            rows = [tuple(row) for row in rows]
            rejected = tuple(rejected)
        except TypeError:
            return empty
        if len(rows) != tokens or any(
            len(row) > self.top_k or any(type(expert) is not int for expert in row)
            for row in rows
        ):
            return empty
        if any(type(expert) is not int or expert < 0 for expert in rejected):
            return empty
        counts = Counter(expert for row in rows for expert in row if expert >= 0)
        rejected = set(rejected)
        if not rejected <= counts.keys() or rejected & mapping.keys():
            return empty
        candidates = sorted(rejected, key=lambda expert: (counts[expert], expert))
        rejected_profile = tuple(counts[expert] for expert in candidates)
        total_routes = sum(counts.values())
        cpu_routes = 0
        profile = []
        best_us = math.inf
        best = empty
        for k, expert in enumerate(candidates, 1):
            cpu_routes += counts[expert]
            profile.append(counts[expert])
            key = (
                tokens,
                total_routes - cpu_routes,
                cpu_routes,
                k,
                tuple(profile),
                rejected_profile,
            )
            if self.choose(*key) == "cpu":
                cpu_us = self.samples[key][0]
                if cpu_us < best_us:
                    best_us = cpu_us
                    best = frozenset(candidates[:k])
        return best


@dataclass
class LayerState:
    epoch: int = 0
    phase: str = "locked"
    mapping: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResizePlan:
    generation: int
    old_keys: tuple
    budget: int
    moves: tuple
    keys: tuple
    evicted: tuple
    reserved: tuple


class GlobalLfu:
    def __init__(self, layers, experts, layout, decay=0.95, admission=1.9):
        if layers <= 0 or experts <= 0 or not 0 < decay <= 1 or admission < 0:
            raise ValueError("invalid LFU parameters")
        if len(layout) != layers or any(n < 0 for n in layout):
            raise ValueError("invalid global cache layout")
        self.layers = [LayerState() for _ in range(layers)]
        self.experts = experts
        self.frequency = [[0.0] * experts for _ in range(layers)]
        self.owner = [layer for layer, count in enumerate(layout) for _ in range(count)]
        self.keys = [None] * len(self.owner)
        self._index_owners()
        self.decay, self.admission = decay, admission
        self.generation = 0
        self.reserved = set()

    def _index_owners(self):
        self.owned_slots = [set() for _ in self.layers]
        self.free_slots = set()
        for slot, owner in enumerate(self.owner):
            if owner is None:
                self.free_slots.add(slot)
            elif owner >= 0:
                self.owned_slots[owner].add(slot)

    def set_owner(self, slot, owner):
        """Update indexes after the caller retires any previous resident key."""
        previous = self.owner[slot]
        if previous == owner:
            return
        if previous is None:
            self.free_slots.remove(slot)
        elif previous >= 0:
            self.owned_slots[previous].remove(slot)
        self.owner[slot] = owner
        if owner is None:
            self.free_slots.add(slot)
        elif owner >= 0:
            self.owned_slots[owner].add(slot)

    def plan_resize(self, budget, *, reserved=None):
        """Global LFU retention; preserve surviving prefix addresses in place."""
        if (
            type(budget) is not int
            or not 0 <= budget <= len(self.layers) * self.experts
        ):
            raise ValueError("invalid global cache budget")
        if any(state.phase != "locked" for state in self.layers):
            raise RuntimeError(
                "resize requires all compute and maintenance windows drained"
            )
        residents = [
            (slot, key) for slot, key in enumerate(self.keys) if key is not None
        ]
        residents.sort(
            key=lambda item: (-self.frequency[item[1][0]][item[1][1]], item[0])
        )
        reserved = (
            {slot for slot in self.reserved if slot < budget}
            if reserved is None
            else set(reserved)
        )
        if any(not 0 <= slot < budget for slot in reserved):
            raise ValueError("reserved slot outside resized cache")
        selected = residents[: max(0, budget - len(reserved))]
        kept = {key for _, key in selected}
        keys = [None] * budget
        moves, tail = [], []
        for slot, key in selected:
            if slot < budget and slot not in reserved:
                keys[slot] = key
            else:
                tail.append((slot, key))
        free = [
            slot
            for slot, key in enumerate(keys)
            if key is None and slot not in reserved
        ]
        for (source, key), destination in zip(tail, free):
            keys[destination] = key
            moves.append((source, destination))
        evicted = tuple((slot, key) for slot, key in residents if key not in kept)
        return ResizePlan(
            self.generation,
            tuple(self.keys),
            budget,
            tuple(moves),
            tuple(keys),
            evicted,
            tuple(sorted(reserved)),
        )

    def commit_resize(self, plan):
        # Mapping is published only after every move completes, before unmap.
        if (
            plan.generation != self.generation
            or plan.old_keys != tuple(self.keys)
            or any(state.phase != "locked" for state in self.layers)
        ):
            raise RuntimeError("stale global cache resize")
        self.keys = list(plan.keys)
        self.reserved = set(plan.reserved)
        self.owner = [
            key[0] if key is not None else (-1 if slot in self.reserved else None)
            for slot, key in enumerate(self.keys)
        ]
        self._index_owners()
        for state in self.layers:
            state.mapping.clear()
            state.epoch += 1
        for slot, key in enumerate(self.keys):
            if key is not None:
                self.layers[key[0]].mapping[key[1]] = slot
        self.generation += 1

    def seed(self, layer, mapping):
        state = self.layers[layer]
        if state.phase != "locked":
            raise RuntimeError("cannot seed dirty layer")
        owned = sorted(self.owned_slots[layer])
        for (expert, _), slot in zip(sorted(mapping.items()), owned):
            state.mapping[expert] = slot
            self.keys[slot] = (layer, expert)

    def begin(self, layer):
        state = self.layers[layer]
        if state.phase != "locked":
            raise RuntimeError("previous layer maintenance not ready")
        state.epoch += 1
        state.phase = "computing"
        return state.epoch, dict(state.mapping)

    def release(self, layer, epoch):
        state = self.layers[layer]
        if state.epoch != epoch or state.phase != "computing":
            raise RuntimeError("stale layer release")
        state.phase = "managing"

    def observe(self, layer, ids):
        # Every distinct expert per window counts once, including CPU misses.
        self.frequency[layer] = [f * self.decay for f in self.frequency[layer]]
        for expert in set(ids):
            if not 0 <= expert < self.experts:
                raise ValueError("invalid expert")
            self.frequency[layer][expert] += 1

    def plan(self, layer, epoch, ids, *, force=False, safe_layers=None):
        state = self.layers[layer]
        if state.phase != "managing" or state.epoch != epoch:
            raise RuntimeError("cache mutation outside released layer")
        # Cross-layer reuse requires the caller to retire those GPU readers.
        # Legacy asynchronous maintenance only certifies its own layer.
        eligible = {layer} if safe_layers is None else set(safe_layers) | {layer}
        if any(type(l) is not int or not 0 <= l < len(self.layers) for l in eligible):
            raise ValueError("invalid reader-safe layer")
        if any(l != layer and self.layers[l].phase != "locked" for l in eligible):
            raise RuntimeError("global victim layer still computing or managing")
        requested = set(ids)
        self.observe(layer, requested)
        candidates = sorted(
            requested - state.mapping.keys(),
            key=lambda e: (-self.frequency[layer][e], e),
        )
        evicted, fills = [], []
        # Reserve each available slot once; candidate admission never creates
        # another free slot before using its chosen victim immediately.
        free = iter(
            sorted(
                self.free_slots
                | {
                    slot
                    for l in eligible
                    for slot in self.owned_slots[l]
                    if self.keys[slot] is None
                }
            )
        )
        victims = None
        for expert in candidates:
            slot = next(free, None)
            if slot is None:
                # Empty slots need no probation. Replacing a resident does:
                # cold misses compute on CPU until they become frequent.
                if not force and self.frequency[layer][expert] < self.admission:
                    continue
                # Heat and eligibility stay fixed throughout this window.
                # New fills are unpublished, so only removals change victims.
                if victims is None:
                    victims = [
                        (self.frequency[l][e], l, e, slot)
                        for l in eligible
                        for e, slot in self.layers[l].mapping.items()
                        if l != layer or e not in requested
                    ]
                    heapq.heapify(victims)
                if not victims:
                    continue
                _, victim_layer, victim, slot = victims[0]
                if (
                    not force
                    and self.frequency[layer][expert]
                    <= self.frequency[victim_layer][victim]
                ):
                    continue
                heapq.heappop(victims)
                self.layers[victim_layer].mapping.pop(victim)
                self.keys[slot] = None
                evicted.append((victim, slot))
            self.set_owner(slot, layer)
            # New entries stay invisible until the copy-completion commit.
            fills.append((expert, slot))
            self.keys[slot] = (layer, expert)
        # Reclaim genuinely cold slots only from this released layer.
        for expert in list(state.mapping):
            if expert not in requested and self.frequency[layer][expert] < 0.5:
                slot = state.mapping.pop(expert)
                self.keys[slot] = None
                self.set_owner(slot, None)
                evicted.append((expert, slot))
        # Unused ownership returns to the global free pool only in this layer's window.
        for slot in tuple(self.owned_slots[layer]):
            if self.keys[slot] is None:
                self.set_owner(slot, None)
        return evicted, fills

    def commit(self, layer, epoch, fills):
        state = self.layers[layer]
        if state.phase != "managing" or state.epoch != epoch:
            raise RuntimeError("stale maintenance completion")
        for expert, slot in fills:
            if self.owner[slot] != layer or self.keys[slot] != (layer, expert):
                raise RuntimeError("invalid cache publication")
            state.mapping[expert] = slot
        state.phase = "locked"
