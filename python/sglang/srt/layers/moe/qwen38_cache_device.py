# SPDX-License-Identifier: Apache-2.0
"""CUDA decay-LFU decisions with per-slot ready/last-use H2D protection.

The host reads a bounded transfer plan to dispatch copies from ordinary mmap.
Policy state and ID mapping stay on CUDA. This eager path still synchronizes
one plan per invocation: ordinary pageable mmap cannot feed cudaMemcpyAsync
from a device-generated address without host scheduling or a different memory
transport. Graph capture is explicitly rejected rather than replaying stale plans.
"""

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import cache_plan, cache_route
from sglang.srt.layers.moe.qwen38_cache_async import Qwen38AsyncExpertRuntime
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts


class Qwen38DeviceExpertRuntime(Qwen38AsyncExpertRuntime):
    def __init__(self, pack, device, **kwargs):
        super().__init__(pack, device, **kwargs)
        if self.manifest.num_experts > 512:
            raise ValueError("device policy supports at most 512 experts")
        self.device_states = {}
        self.slot_ready = {}
        self.slot_last_use = {}
        self.arena_ready = None
        self.arena_last_use = None
        self.metadata_stream = torch.cuda.Stream(device=self.device)
        self.metadata_done = None
        self.host_plan = torch.empty(
            4 + 3 * self.slots_per_layer, dtype=torch.int32, pin_memory=True
        )
        self.pending = None
        self.poisoned = False

    def _state(self, layer_id):
        if layer_id not in self.device_states:
            slots, experts = self.slots_per_layer, self.manifest.num_experts
            self.device_states[layer_id] = {
                "lookup": torch.full(
                    (experts,), -1, dtype=torch.int32, device=self.device
                ),
                "keys": torch.full((slots,), -1, dtype=torch.int32, device=self.device),
                "frequency": torch.zeros(
                    slots, dtype=torch.float32, device=self.device
                ),
                "ages": torch.zeros(slots, dtype=torch.int64, device=self.device),
                "generation": torch.zeros(slots, dtype=torch.int64, device=self.device),
                "clock": torch.zeros(1, dtype=torch.int64, device=self.device),
                "counts": torch.empty(
                    experts + 1, dtype=torch.int32, device=self.device
                ),
                "plan": torch.empty(
                    4 + 3 * slots, dtype=torch.int32, device=self.device
                ),
            }
            self.slot_ready[layer_id] = [None] * slots
            self.slot_last_use[layer_id] = [None] * slots
        return self.device_states[layer_id]

    def reserve(self):
        super().reserve()
        for layer_id in self.manifest.tensors:
            self._state(layer_id)
        self.metadata_stream.wait_stream(torch.cuda.current_stream(self.device))

    def _allocate(self, capacity):
        tensors = super()._allocate(capacity)
        if hasattr(self, "transfer"):
            self.transfer.wait_stream(torch.cuda.current_stream(self.device))
        return tensors

    def _copy(self, view, destination):
        # Only explicit dependency barriers in prepare()/reserve() order GPU
        # writes. Do not join the whole compute stream on every tensor copy.
        index = self.ring_index
        self.ring_index = (index + 1) % len(self.ring)
        event = self.ring_events[index]
        if event is not None:
            event.synchronize()
        try:
            source = torch.frombuffer(bytearray(view), dtype=torch.uint8)
            staging = self.ring[index][: source.numel()]
            staging.copy_(source)
            with torch.cuda.stream(self.transfer):
                destination.reshape(-1).view(torch.uint8).copy_(
                    staging, non_blocking=True
                )
                event = torch.cuda.Event()
                event.record(self.transfer)
                self.ring_events[index] = event
                self.ready = event
            self.transferred_bytes += source.numel()
        finally:
            view.release()

    def prepare(self, layer_id, topk_ids):
        if self.poisoned:
            raise RuntimeError(
                "device expert cache failed a transfer/launch; recreate runtime"
            )
        if layer_id not in self.manifest.tensors:
            raise ValueError("invalid expert layer")
        if (
            topk_ids.device != self.device
            or topk_ids.dtype != torch.int32
            or topk_ids.ndim != 2
        ):
            raise ValueError(
                "route IDs must be a CUDA integer matrix on the cache device"
            )
        if topk_ids.shape[1] != self.manifest.top_k:
            raise ValueError("routing top-k differs from the pack")
        state = self._state(layer_id)
        compute = torch.cuda.current_stream(self.device)
        # Lazy allocations are initialized on compute; transfer/metadata join
        # before use. During server startup reserve() has already done this.
        if layer_id not in self.caches:
            self.caches[layer_id] = self._allocate(self.slots_per_layer)
        self.metadata_stream.wait_stream(compute)
        with torch.cuda.stream(self.metadata_stream):
            ids = topk_ids.to(torch.int32).contiguous()
            remapped = torch.empty_like(ids)
            cache_route(ids, state["lookup"], remapped, state["counts"])
            cache_plan(
                state["counts"],
                state["lookup"],
                state["keys"],
                state["frequency"],
                state["ages"],
                state["generation"],
                state["clock"],
                state["plan"],
            )
            cache_route(ids, state["lookup"], remapped, state["counts"])
            self.host_plan.copy_(state["plan"], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.metadata_stream)
        topk_ids.record_stream(self.metadata_stream)
        done.synchronize()  # bounded plan only; no CPU LFU/LRU or token-ID copy
        self.metadata_done = done
        values = self.host_plan.tolist()
        status, miss_count, active_count, arena = values[:4]
        if status:
            raise ValueError(f"device cache route/plan error {status}")
        self._record_invocation(bool(arena), topk_ids)
        if arena:
            if not self.prefill_arena:
                raise ValueError("routing exceeds cache capacity; enable prefill_arena")
            if self.arena is None:
                self.arena = self._allocate(self.manifest.num_experts)
                self.transfer.wait_stream(compute)
            if self.arena_layer != layer_id:
                if self.arena_last_use is not None:
                    self.transfer.wait_event(self.arena_last_use)
                self.arena_layer = None
                self.pending = (layer_id, [], True)
                for expert in range(self.manifest.num_experts):
                    self._fill(layer_id, expert, self.arena, expert)
                self.arena_ready = self.ready
                self.arena_layer = layer_id
                self.arena_loads += 1
            self.pending = (layer_id, [], True)
            if self.arena_ready is not None:
                compute.wait_event(self.arena_ready)
            # Serialize shared arena reads across compute streams so one event
            # describes the last reader; decode slots remain independent.
            if self.arena_last_use is not None:
                compute.wait_event(self.arena_last_use)
            return PreparedExperts(
                {**self.arena, **self._scales(layer_id)},
                topk_ids,
                self.manifest.num_experts,
                True,
            )
        slots = values[
            4 + 2 * self.slots_per_layer : 4 + 2 * self.slots_per_layer + active_count
        ]
        self.pending = (layer_id, slots, False)
        for index in range(miss_count):
            expert, slot = values[4 + 2 * index : 6 + 2 * index]
            last = self.slot_last_use[layer_id][slot]
            if last is not None:
                self.transfer.wait_event(last)
            self._fill(layer_id, expert, self.caches[layer_id], slot)
            self.slot_ready[layer_id][slot] = self.ready
        for slot in slots:
            ready = self.slot_ready[layer_id][slot]
            last = self.slot_last_use[layer_id][slot]
            if ready is not None:
                compute.wait_event(ready)
            if last is not None:
                compute.wait_event(last)
        self.misses += miss_count
        self.hits += active_count - miss_count
        remapped.record_stream(compute)
        return PreparedExperts(
            {**self.caches[layer_id], **self._scales(layer_id)},
            remapped,
            self.slots_per_layer,
            False,
        )

    def run(self, layer_id, topk_ids, execute):
        with self.lock, torch.no_grad(), torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "device-cache misses require eager host H2D scheduling; CUDA Graph capture unsupported"
                )
            self.pending = None
            try:
                return execute(self.prepare(layer_id, topk_ids))
            except Exception:
                # Policy commits before H2D dispatch; never treat a partial fill
                # as a hit after a failed operation. Restart rather than recover.
                self.poisoned = True
                raise
            finally:
                compute = torch.cuda.current_stream(self.device)
                if self.pending is not None:
                    layer, slots, arena = self.pending
                    # On exceptional partial copies join transfer for cleanup.
                    if self.poisoned and self.ready is not None:
                        compute.wait_event(self.ready)
                    last = torch.cuda.Event()
                    last.record(compute)
                    if arena:
                        self.arena_last_use = last
                    else:
                        for slot in slots:
                            self.slot_last_use[layer][slot] = last

    def synchronize(self):
        with self.lock:
            self.metadata_stream.synchronize()
            self.transfer.synchronize()
            for events in self.slot_last_use.values():
                for event in events:
                    if event is not None:
                        event.synchronize()
            if self.arena_last_use is not None:
                self.arena_last_use.synchronize()

    def memory_budget(self):
        result = super().memory_budget()
        result["pinned_staging_bytes"] += self.host_plan.numel() * 4
        result["device_policy_bytes"] = sum(
            t.numel() * t.element_size()
            for state in self.device_states.values()
            for t in state.values()
        )
        return result
