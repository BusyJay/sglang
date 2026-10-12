# SPDX-License-Identifier: Apache-2.0
"""Experimental event-protected transfers with a bounded pinned staging ring.

JIT routing transfers a 513-int summary to the host, not the token IDs. Cache
policy still runs on the CPU and synchronizes this summary once per layer.
This is not a fully device-controlled cache policy or a CUDA graph path.
"""

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import cache_route
from sglang.srt.layers.moe.qwen38_cache_runtime import Qwen38SynchronousExpertRuntime
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS


class Qwen38AsyncExpertRuntime(Qwen38SynchronousExpertRuntime):
    def __init__(
        self, pack, device, *, staging_buffers=4, staging_buffer_bytes=64 << 20, **kwargs
    ):
        super().__init__(
            pack, device, staging_buffer_bytes=staging_buffer_bytes, **kwargs
        )
        if self.device.type != "cuda":
            raise ValueError("async expert runtime requires CUDA")
        if not 2 <= staging_buffers <= 64:
            raise ValueError("staging_buffers must be between 2 and 64")
        self.transfer = torch.cuda.Stream(device=self.device)
        self.ring = [self.staging] + [
            torch.empty_like(self.staging, device="cpu", pin_memory=True)
            for _ in range(staging_buffers - 1)
        ]
        self.ring_events = [None] * staging_buffers
        self.ring_index = 0
        self.last_compute = None
        self.ready = None
        self.lookups = {}
        self.route_counts = torch.empty(
            self.manifest.num_experts + 1, dtype=torch.int32, device=self.device
        )
        self.host_counts = torch.empty_like(
            self.route_counts, device="cpu", pin_memory=True
        )
        self.host_lookup = torch.empty(
            self.manifest.num_experts, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.lookup_event = None
        self.outputs = {}

    def reserve(self):
        # Allocations/zero-initialization happen on the caller's stream.
        # _copy joins that stream before writing destinations on transfer.
        super().reserve()
        self.transfer.synchronize()

    def _copy(self, view, destination):
        self._copy_batch([(view, destination)])

    def _order_transfer(self):
        self.transfer.wait_stream(torch.cuda.current_stream(self.device))

    def _copy_batch(self, entries):
        """Consume mmap views into reusable pinned pages, with one event per page.

        Full tensors are split into bounded contiguous H2Ds. Small tensors share
        a page/event, but retain their individual GPU destinations. No writable
        alias of the read-only pack or intermediate bytearray is constructed.
        """
        pending = []
        used = 0
        index = self.ring_index
        host_view = None
        batches_before = self.staging_batches
        try:
            self._order_transfer()
            for view, destination in entries:
                target = destination.reshape(-1).view(torch.uint8)
                if len(view) != target.numel():
                    raise ValueError("pack view and destination byte sizes differ")
                offset = 0
                while offset < len(view):
                    if host_view is None:
                        event = self.ring_events[index]
                        if event is not None:
                            event.synchronize()  # Never overwrite in-flight pinned bytes.
                        host_view = memoryview(self.ring[index].numpy()).cast("B")
                    count = min(len(view) - offset, len(host_view) - used)
                    part = view[offset : offset + count]
                    try:
                        host_view[used : used + count] = part
                    finally:
                        part.release()
                    pending.append((target[offset : offset + count], used, count))
                    offset += count
                    used += count
                    if used == len(host_view):
                        host_view.release()
                        host_view = None
                        self._dispatch_staging_batch(index, pending)
                        pending = []
                        used = 0
                        index = self.ring_index
            if pending:
                self._dispatch_staging_batch(index, pending)
        finally:
            if host_view is not None:
                host_view.release()
            try:
                if self.staging_batches != batches_before:
                    # Ring events are re-recorded on reuse. Publish an immutable
                    # completion event for persistent slot-ready dependencies.
                    self.ready = torch.cuda.Event()
                    self.ready.record(self.transfer)
            finally:
                for view, _ in entries:
                    view.release()

    def _dispatch_staging_batch(self, index, pending):
        with torch.cuda.stream(self.transfer):
            try:
                for target, offset, count in pending:
                    target.copy_(
                        self.ring[index][offset : offset + count], non_blocking=True
                    )
                    self.transferred_bytes += count
                    self.h2d_copy_calls += 1
            finally:
                # Record even on partial dispatch failure to protect staging reuse.
                event = self.ring_events[index]
                if event is None:
                    event = torch.cuda.Event(blocking=True)
                    self.ring_events[index] = event
                event.record(self.transfer)
                self.staging_batches += 1
                self.ring_index = (index + 1) % len(self.ring)

    def _fill(self, layer_id, expert_id, tensors, slot_id, names=REQUIRED_TENSORS):
        self._copy_batch(
            [
                (
                    self.pack.expert_tensor_bytes(layer_id, expert_id, name),
                    tensors[name][slot_id],
                )
                for name in names
            ]
        )

    def _fill_arena(self, layer_id):
        self._copy_batch(
            [
                (self.pack.tensor_bytes(layer_id, name), self.arena[name])
                for name in REQUIRED_TENSORS
            ]
        )

    def _scales(self, layer_id):
        # Scalar scale validation reads CUDA values; join transfer before reading.
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import LAYER_TENSORS

        if layer_id not in self.layer_scales:
            scales = {}
            for name in LAYER_TENSORS:
                tensor = torch.empty(1, dtype=torch.float32, device=self.device)
                self._copy(self.pack.tensor_bytes(layer_id, name), tensor)
                torch.cuda.current_stream(self.device).wait_event(self.ready)
                if not bool(torch.isfinite(tensor).all()) or not bool(
                    (tensor > 0).all()
                ):
                    raise ValueError(f"invalid layer input scale: {layer_id} {name}")
                # Keep activation global scales 0-d; the FlashInfer NVFP4 CUTLASS
                # binding rejects [1]-shaped tensors in getQuantParams.
                scales[name] = tensor.reshape(())
            self.layer_scales[layer_id] = scales
        return self.layer_scales[layer_id]

    def _route_ids(self, layer_id, topk_ids):
        ids = topk_ids.to(torch.int32).contiguous()
        if layer_id not in self.lookups:
            self.lookups[layer_id] = torch.full(
                (self.manifest.num_experts,), -1, dtype=torch.int32, device=self.device
            )
        output = torch.empty_like(ids)
        cache_route(ids, self.lookups[layer_id], output, self.route_counts)
        self.host_counts.copy_(self.route_counts, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        event.synchronize()
        counts = self.host_counts.tolist()
        if counts[-1]:
            raise ValueError("route ID is outside the expert pack")
        self.outputs[layer_id] = output
        return [expert for expert, count in enumerate(counts[:-1]) if count]

    def _remap(self, layer_id, topk_ids, mapping):
        if self.lookup_event is not None:
            self.lookup_event.synchronize()
        self.host_lookup.fill_(-1)
        for expert, slot in mapping.items():
            self.host_lookup[expert] = slot
        lookup = self.lookups[layer_id]
        lookup.copy_(self.host_lookup, non_blocking=True)
        self.lookup_event = torch.cuda.Event()
        self.lookup_event.record(torch.cuda.current_stream(self.device))
        output = self.outputs[layer_id]
        cache_route(
            topk_ids.to(torch.int32).contiguous(), lookup, output, self.route_counts
        )
        return output

    def run(self, layer_id, topk_ids, execute):
        with self.lock, torch.no_grad(), torch.cuda.device(self.device):
            compute = torch.cuda.current_stream(self.device)
            # Coarse last-use protection covers cache, arena and exceptional launches.
            # Transfer overwrites wait for the previous invocation's computation.
            if self.last_compute is not None:
                compute.wait_event(self.last_compute)
                self.transfer.wait_event(self.last_compute)
            self.transfer.wait_stream(compute)
            try:
                prepared = self.prepare(layer_id, topk_ids)
                if self.ready is not None:
                    compute.wait_event(self.ready)
                return execute(prepared)
            finally:
                # Also join transfers on failure, so all allocator lifetimes are safe.
                if self.ready is not None:
                    compute.wait_event(self.ready)
                self.last_compute = torch.cuda.Event()
                self.last_compute.record(compute)

    def synchronize(self):
        with self.lock:
            self.transfer.synchronize()
            if self.last_compute is not None:
                self.last_compute.synchronize()

    def memory_budget(self):
        result = super().memory_budget()
        result["pinned_staging_bytes"] = (
            sum(t.numel() for t in self.ring)
            + (self.host_counts.numel() + self.host_lookup.numel()) * 4
        )
        return result
