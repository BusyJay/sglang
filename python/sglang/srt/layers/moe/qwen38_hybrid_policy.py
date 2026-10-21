# SPDX-License-Identifier: Apache-2.0
"""Pure policy: shared slots, per-layer immutable compute windows, deferred LFU."""

from dataclasses import dataclass, field


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

    def plan(self, layer, epoch, ids, *, force=False):
        state = self.layers[layer]
        if state.phase != "managing" or state.epoch != epoch:
            raise RuntimeError("cache mutation outside released layer")
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
                | {slot for slot in self.owned_slots[layer] if self.keys[slot] is None}
            )
        )
        for expert in candidates:
            slot = next(free, None)
            if slot is None:
                # Empty slots need no probation. Replacing a resident does:
                # cold misses compute on CPU until they become frequent.
                if not force and self.frequency[layer][expert] < self.admission:
                    continue
                victims = [e for e in state.mapping if e not in requested]
                if not victims:
                    continue
                victim = min(victims, key=lambda e: (self.frequency[layer][e], e))
                if (
                    not force
                    and self.frequency[layer][expert] <= self.frequency[layer][victim]
                ):
                    continue
                slot = state.mapping.pop(victim)
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
