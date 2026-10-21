# SPDX-License-Identifier: Apache-2.0
"""Synchronous reference runtime for the SM120 expert pack.

Only small staging buffers are pinned. Cache/arena addresses stay fixed; every
CUDA invocation finishes before buffers can be overwritten. This deliberately
synchronizes routing and compute, and is a correctness baseline for a future
stream/event implementation, not the asynchronous production fast path.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.layers.moe.qwen38_cache_capacity import normalize_layer_slots
from sglang.srt.layers.moe.qwen38_expert_cache import LayerExpertCachePolicy
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    LAYER_TENSORS,
    REQUIRED_TENSORS,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedExperts:
    tensors: dict[str, torch.Tensor]
    topk_ids: torch.Tensor
    num_experts: int
    arena: bool
    cache_tensors: dict[str, torch.Tensor] | None = None
    cache_experts: dict[int, int] | None = None
    arena_experts: list[int] | None = None
    wait_for_weights: Callable[[], None] | None = None
    early_output: torch.Tensor | None = None
    cache_lookup: torch.Tensor | None = None
    cache_slot_limit: int | None = None


class Qwen38SynchronousExpertRuntime:
    def __init__(
        self,
        pack,
        device,
        *,
        slots_per_layer=56,
        prefill_arena=True,
        stats_log_interval=0,
        staging_buffer_bytes=0,
        layer_cache_slots=None,
        route_trace_path=None,
        route_trace_limit=10000,
    ):
        self.pack = pack
        if getattr(pack.manifest, "scale_encoding", {}):
            pack.validate_scale_slabs()
        self.manifest = pack.manifest
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.layer_cache_slots = normalize_layer_slots(
            slots_per_layer,
            layer_cache_slots,
            self.manifest.num_layers,
            self.manifest.num_experts,
        )
        self.slots_per_layer = slots_per_layer
        self.max_layer_slots = max(self.layer_cache_slots)
        if type(route_trace_limit) is not int or route_trace_limit < 1:
            raise ValueError("route_trace_limit must be a positive integer")
        self.route_trace_path = route_trace_path
        if route_trace_path is not None and not isinstance(route_trace_path, str):
            raise ValueError("route_trace_path must be a string")
        self.route_trace_limit = route_trace_limit
        self.route_trace_events = 0
        self.prefill_arena = prefill_arena
        self.stats_log_interval = int(stats_log_interval)
        if self.stats_log_interval < 0:
            raise ValueError("stats_log_interval must be non-negative")
        self.policies = {}
        self.caches = {}
        self.arena = None
        self.arena_layer = None
        self.layer_scales = {}
        self.lock = threading.Lock()
        self.hits = self.misses = self.arena_loads = self.transferred_bytes = 0
        self.h2d_copy_calls = self.staging_batches = 0
        self.run_invocations = 0
        self.cache_invocations = 0
        self.arena_invocations = 0
        self.route_token_count = 0
        self._last_logged_runs = 0
        self._atexit_registered = False
        largest = max(
            record.nbytes // self.manifest.num_experts
            for records in self.manifest.tensors.values()
            for name, record in records.items()
            if name in REQUIRED_TENSORS
        )
        if staging_buffer_bytes < 0:
            raise ValueError("staging_buffer_bytes must be non-negative")
        staging_bytes = (
            max(largest, staging_buffer_bytes) if staging_buffer_bytes else largest
        )
        self.staging = torch.empty(
            staging_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        # A shared arena requires the same transformed shape/dtype in all layers.
        first = self.manifest.tensors[0]
        for records in self.manifest.tensors.values():
            for name in REQUIRED_TENSORS:
                if (records[name].shape, records[name].dtype) != (
                    first[name].shape,
                    first[name].dtype,
                ):
                    raise ValueError(
                        "expert pack layers have different runtime layouts"
                    )
        self._register_atexit_logger()

    def _register_atexit_logger(self):
        if self._atexit_registered:
            return

        def _log_final_stats(runtime=self):
            runtime.log_stats("final")

        atexit.register(_log_final_stats)
        self._atexit_registered = True

    def slots_for_layer(self, layer_id):
        return self.layer_cache_slots[layer_id]

    def _record_invocation(self, arena: bool, topk_ids: torch.Tensor, layer_id=None):
        self.run_invocations += 1
        self.route_token_count += int(topk_ids.numel())
        if arena:
            self.arena_invocations += 1
        else:
            self.cache_invocations += 1
        if self.route_trace_path and self.route_trace_events < self.route_trace_limit:
            # Explicit profiling mode: synchronizes route IDs to CPU and writes
            # ordered JSONL. Disabled by default; never use for latency results.
            with open(self.route_trace_path, "a", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "layer": layer_id,
                            "routes": topk_ids.detach().cpu().tolist(),
                        }
                    )
                    + "\n"
                )
            self.route_trace_events += 1
        self._maybe_log_stats()

    def stats_snapshot(self):
        cache_events = self.hits + self.misses
        return {
            "slots_per_layer": self.slots_per_layer,
            "layer_cache_slots": self.layer_cache_slots,
            "cache_slots_total": sum(self.layer_cache_slots),
            "route_trace_events": self.route_trace_events,
            "prefill_arena": self.prefill_arena,
            "run_invocations": self.run_invocations,
            "cache_invocations": self.cache_invocations,
            "arena_invocations": self.arena_invocations,
            "route_token_count": self.route_token_count,
            "hits": self.hits,
            "misses": self.misses,
            "arena_loads": self.arena_loads,
            "transferred_bytes": self.transferred_bytes,
            "h2d_copy_calls": self.h2d_copy_calls,
            "staging_batches": self.staging_batches,
            "cache_event_count": cache_events,
            "cache_hit_rate": (
                float(self.hits) / float(cache_events) if cache_events else None
            ),
            "avg_route_tokens_per_run": (
                float(self.route_token_count) / float(self.run_invocations)
                if self.run_invocations
                else None
            ),
        }

    def log_stats(self, label="snapshot"):
        stats = self.stats_snapshot()
        if stats["run_invocations"] == 0:
            return
        import os

        from sglang.srt.layers.moe.qwen38_cache_stats_format import compact_stats

        style = os.getenv("SGLANG_QWEN38_CACHE_STATS_FORMAT", "compact")
        if style not in ("compact", "full"):
            raise ValueError("cache stats format must be compact or full")
        payload = compact_stats(stats) if style == "compact" else stats
        logger.info(
            "Qwen38ExpertCacheStats[%s] %s",
            label,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _maybe_log_stats(self):
        if not self.stats_log_interval:
            return
        if self.run_invocations - self._last_logged_runs < self.stats_log_interval:
            return
        self._last_logged_runs = self.run_invocations
        self.log_stats("periodic")

    def reserve(self):
        """Allocate before SGLang measures free memory for the KV pool."""
        from sglang.srt.layers.moe.qwen38_startup_memory import (
            get_startup_memory_probe,
            memory_mark,
        )

        probe = get_startup_memory_probe(self.device)
        with self.lock:
            for layer_id in self.manifest.tensors:
                memory_mark(probe, f"base.layer.{layer_id:02d}.before")
                slots = self.slots_for_layer(layer_id)
                if layer_id not in self.caches:
                    self.caches[layer_id] = self._allocate(slots)
                    self.policies[layer_id] = LayerExpertCachePolicy(
                        self.manifest.num_experts, slots
                    )
                self._scales(layer_id)
                memory_mark(probe, f"base.layer.{layer_id:02d}.after")
            memory_mark(probe, "base.arena.before")
            if self.prefill_arena and self.arena is None:
                self.arena = self._allocate(self.manifest.num_experts)
            memory_mark(probe, "base.arena.after")

    def _allocate(self, capacity):
        from sglang.srt.layers.moe.qwen38_startup_memory import (
            get_startup_memory_probe,
            memory_mark,
        )

        probe = get_startup_memory_probe(self.device)
        tensors = {}
        for name in REQUIRED_TENSORS:
            memory_mark(probe, f"base.allocate.{capacity}.{name}.before")
            tensors[name] = torch.zeros(
                (capacity, *self.manifest.tensors[0][name].shape[1:]),
                dtype=getattr(torch, self.manifest.tensors[0][name].dtype),
                device=self.device,
            )
            memory_mark(
                probe,
                f"base.allocate.{capacity}.{name}.after",
                metadata={
                    "tensor_bytes": tensors[name].numel() * tensors[name].element_size()
                },
            )
        return tensors

    def _copy(self, view, destination):
        # Copy read-only mmap bytes into bounded writable staging storage.
        # The temporary bytearray is at most one expert tensor, never a layer.
        try:
            source = torch.frombuffer(bytearray(view), dtype=torch.uint8)
            staging = self.staging[: source.numel()]
            staging.copy_(source)
            destination.reshape(-1).view(torch.uint8).copy_(staging, non_blocking=False)
            self.transferred_bytes += source.numel()
            self.h2d_copy_calls += 1
            self.staging_batches += 1
        finally:
            view.release()

    def _fill(self, layer_id, expert_id, tensors, slot_id):
        for name in REQUIRED_TENSORS:
            self._copy(
                self.pack.expert_tensor_bytes(layer_id, expert_id, name),
                tensors[name][slot_id],
            )

    def _fill_arena(self, layer_id):
        # Reference path; asynchronous runtimes override with contiguous copies.
        for expert in range(self.manifest.num_experts):
            self._fill(layer_id, expert, self.arena, expert)

    def _scales(self, layer_id):
        if layer_id not in self.layer_scales:
            scales = {}
            for name in LAYER_TENSORS:
                tensor = torch.empty(1, dtype=torch.float32, device=self.device)
                self._copy(self.pack.tensor_bytes(layer_id, name), tensor)
                if not bool(torch.isfinite(tensor).all()) or not bool(
                    (tensor > 0).all()
                ):
                    raise ValueError(
                        f"layer {layer_id} {name} must be finite and positive"
                    )
                # FlashInfer's NVFP4 CUTLASS binding requires the activation
                # global scales to be true scalars (or per-expert vectors); a
                # [1]-tensor fails its dimension checks, so keep them 0-d like
                # the resident ModelOpt layer attributes.
                scales[name] = tensor.reshape(())
            self.layer_scales[layer_id] = scales
        return self.layer_scales[layer_id]

    def prepare(self, layer_id, topk_ids):
        """Called inside run's serialized scope; IDs -1 represent padded routes."""
        if topk_ids.device != self.device:
            raise ValueError("route IDs and expert cache must be on the same device")
        if topk_ids.dtype not in (torch.int32, torch.int64) or topk_ids.ndim != 2:
            raise ValueError("route IDs must be a two-dimensional integer tensor")
        if topk_ids.shape[1] != self.manifest.top_k:
            raise ValueError("routing top-k differs from the expert pack")
        if layer_id not in self.manifest.tensors:
            raise ValueError("invalid expert-pack layer")
        ids = self._route_ids(layer_id, topk_ids)
        if any(value < -1 or value >= self.manifest.num_experts for value in ids):
            raise ValueError("route ID is outside the expert pack")
        active = list(dict.fromkeys(value for value in ids if value >= 0))
        slots = self.slots_for_layer(layer_id)
        use_arena = len(active) > slots
        self._record_invocation(use_arena, topk_ids, layer_id)
        if use_arena:
            if not self.prefill_arena:
                raise ValueError("routing exceeds cache capacity; enable prefill_arena")
            if self.arena is None:
                self.arena = self._allocate(self.manifest.num_experts)
            if self.arena_layer != layer_id:
                self.arena_layer = None
                self._fill_arena(layer_id)
                self.arena_layer = layer_id
                self.arena_loads += 1
            tensors = self.arena
            remapped = topk_ids
            capacity = self.manifest.num_experts
        else:
            if layer_id not in self.policies:
                self.policies[layer_id] = LayerExpertCachePolicy(
                    self.manifest.num_experts, slots
                )
                self.caches[layer_id] = self._allocate(slots)
            policy = self.policies[layer_id]
            mapping, misses = policy.ensure(active)
            tensors = self.caches[layer_id]
            try:
                for expert in misses:
                    self._fill(layer_id, expert, tensors, mapping[expert])
            except Exception:
                # A partially copied expert must not become a hit on retry.
                for expert in misses:
                    slot = policy.expert_to_slot.pop(expert)
                    policy.slots[slot].expert_id = None
                raise
            self.hits += len(active) - len(misses)
            self.misses += len(misses)
            remapped = self._remap(layer_id, topk_ids, mapping)
            capacity = slots
        return PreparedExperts(
            {**tensors, **self._scales(layer_id)}, remapped, capacity, use_arena
        )

    def _route_ids(self, layer_id, topk_ids):
        return topk_ids.detach().to("cpu").reshape(-1).tolist()

    def _remap(self, layer_id, topk_ids, mapping):
        lookup = torch.full(
            (self.manifest.num_experts + 1,),
            -1,
            dtype=topk_ids.dtype,
            device=self.device,
        )
        for expert, slot in mapping.items():
            lookup[expert] = slot
        return lookup[topk_ids.to(torch.int64)]

    def run(self, layer_id, topk_ids, execute):
        with self.lock, torch.no_grad():
            try:
                return execute(self.prepare(layer_id, topk_ids))
            finally:
                # Even if a kernel raises, no slot or shared arena is reused
                # while a previous launch still reads it.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

    def memory_budget(self):
        per_layer = [
            sum(r.nbytes for n, r in records.items() if n in REQUIRED_TENSORS)
            for records in self.manifest.tensors.values()
        ]
        return {
            "decode_cache_bytes": sum(
                size // self.manifest.num_experts * self.slots_for_layer(layer)
                for layer, size in zip(self.manifest.tensors, per_layer)
            ),
            "prefill_arena_bytes": max(per_layer) if self.prefill_arena else 0,
            "pinned_staging_bytes": (
                self.staging.numel() if self.device.type == "cuda" else 0
            ),
        }
