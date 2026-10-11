# SPDX-License-Identifier: Apache-2.0
"""Device-independent Qwen3.8 expert-cache policy and ID remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class CacheSlot:
    expert_id: int | None = None
    generation: int = 0
    frequency: float = 0.0
    last_used: int = 0
    protected: int = 0


class LayerExpertCachePolicy:
    """Decay-LFU with LRU tie-breaking and explicit active-slot protection."""

    def __init__(
        self,
        num_experts: int,
        num_slots: int,
        *,
        decay: float = 0.95,
        policy: str = "lfu",
        protected_fraction: float = 0.7,
    ):
        if num_experts <= 0 or not 0 < num_slots <= num_experts:
            raise ValueError(
                "num_experts and num_slots must satisfy 0 < slots <= experts"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if policy not in ("lfu", "slru") or not 0 <= protected_fraction < 1:
            raise ValueError("invalid cache policy or protected fraction")
        self.num_experts = num_experts
        self.decay = decay
        self.policy = policy
        self.protected_slots = int(num_slots * protected_fraction)
        self.slots = [CacheSlot() for _ in range(num_slots)]
        self.expert_to_slot: dict[int, int] = {}
        self.clock = 0

    def _check_expert(self, expert_id: int) -> None:
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"expert id {expert_id} is outside [0, {self.num_experts})"
            )

    def _touch(self, slot_id: int) -> None:
        self.clock += 1
        slot = self.slots[slot_id]
        slot.last_used = self.clock
        if self.policy == "lfu":
            slot.frequency += 1.0
        elif slot.frequency != 1.0:
            # frequency stores the SLRU segment: 0 probation, 1 protected.
            slot.frequency = 1.0
            protected = [s for s in self.slots if s.frequency == 1.0]
            if len(protected) > self.protected_slots:
                min(protected, key=lambda s: s.last_used).frequency = 0.0

    def _victim(self, active_experts: set[int]) -> int:
        for slot_id, slot in enumerate(self.slots):
            if slot.expert_id is None:
                return slot_id
        candidates = [
            (slot.frequency, slot.last_used, slot_id)
            for slot_id, slot in enumerate(self.slots)
            if slot.protected == 0 and slot.expert_id not in active_experts
        ]
        if not candidates:
            raise RuntimeError("no evictable Qwen3.8 expert-cache slot")
        return min(candidates)[2]

    def ensure(self, expert_ids: Iterable[int]) -> tuple[dict[int, int], list[int]]:
        ordered = list(dict.fromkeys(int(value) for value in expert_ids))
        if len(ordered) > len(self.slots):
            raise ValueError(
                "one routing step touches more experts than the cache has slots"
            )
        for expert_id in ordered:
            self._check_expert(expert_id)
        active = set(ordered)
        missing_count = sum(expert not in self.expert_to_slot for expert in ordered)
        available = sum(
            slot.protected == 0 and slot.expert_id not in active for slot in self.slots
        )
        if missing_count > available:
            raise RuntimeError("no evictable Qwen3.8 expert-cache slot")
        # Decay all residents, so old hot experts eventually become evictable.
        if self.policy == "lfu":
            for slot in self.slots:
                slot.frequency *= self.decay
        misses: list[int] = []
        for expert_id in ordered:
            slot_id = self.expert_to_slot.get(expert_id)
            if slot_id is None:
                slot_id = self._victim(active)
                slot = self.slots[slot_id]
                if slot.expert_id is not None:
                    del self.expert_to_slot[slot.expert_id]
                slot.expert_id = expert_id
                slot.generation += 1
                slot.frequency = 0.0
                slot.protected = 0
                self.expert_to_slot[expert_id] = slot_id
                misses.append(expert_id)
            if self.policy == "slru" and expert_id in misses:
                # A first access enters probation; only a subsequent hit promotes.
                self.clock += 1
                self.slots[slot_id].last_used = self.clock
            else:
                self._touch(slot_id)
        return {
            expert_id: self.expert_to_slot[expert_id] for expert_id in ordered
        }, misses

    def protect(self, expert_ids: Iterable[int]) -> None:
        for expert_id in expert_ids:
            slot_id = self.expert_to_slot[int(expert_id)]
            self.slots[slot_id].protected += 1

    def release(self, expert_ids: Iterable[int]) -> None:
        for expert_id in expert_ids:
            slot = self.slots[self.expert_to_slot[int(expert_id)]]
            if slot.protected <= 0:
                raise RuntimeError(f"expert {expert_id} cache protection underflow")
            slot.protected -= 1

    def remap(self, topk_ids: Sequence[Sequence[int]]) -> list[list[int]]:
        result: list[list[int]] = []
        for row in topk_ids:
            result.append([self.expert_to_slot[int(expert_id)] for expert_id in row])
        return result


def remap_topk_ids(topk_ids, expert_to_slot):
    """Remap list-like or torch Tensor IDs without importing torch at module load."""
    if hasattr(topk_ids, "device") and hasattr(topk_ids, "shape"):
        import torch

        mapping = torch.full(
            (max(expert_to_slot) + 1,),
            -1,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        for expert_id, slot_id in expert_to_slot.items():
            mapping[int(expert_id)] = int(slot_id)
        result = mapping[topk_ids.to(torch.int64)]
        if bool((result < 0).any()):
            raise KeyError("top-k contains an expert that is not resident")
        return result.to(dtype=topk_ids.dtype)
    return [[expert_to_slot[int(value)] for value in row] for row in topk_ids]
