# SPDX-License-Identifier: Apache-2.0
"""Device policy and asynchronous CPU mailbox for the optional hybrid cache."""

import mmap
import time

import torch
from sglang.kernels.ops.moe.qwen38_expert_cache import _module as fill_module
from sglang.kernels.ops.moe.qwen38_hybrid_plan import (
    COST_COLUMNS,
    MAX_ROUTES,
    _module,
    hybrid_merge,
    hybrid_plan_indexed,
    hybrid_plan_stage_indexed,
)
from sglang.srt.layers.moe.qwen38_heap_index import build_heap_index


class _MappedBatch:
    """One reusable request and result in a CUDA-registered host mapping."""

    def __init__(self, mapping, hidden, top_k, offset, address):
        x_elements, routes = 3 * hidden, 3 * top_k
        ids_offset = 12 + x_elements * 2
        weights_offset = ids_offset + routes * 4
        result_offset = weights_offset + routes * 4
        completed_offset = result_offset + x_elements * 4
        self.head = torch.frombuffer(mapping, dtype=torch.int32, count=1, offset=offset)
        self.completed = torch.frombuffer(
            mapping, dtype=torch.int32, count=1, offset=offset + completed_offset
        )
        self.layer = torch.frombuffer(
            mapping, dtype=torch.int32, count=1, offset=offset + 4
        )
        self.tokens = torch.frombuffer(
            mapping, dtype=torch.int32, count=1, offset=offset + 8
        )
        self.x = torch.frombuffer(
            mapping, dtype=torch.bfloat16, count=x_elements, offset=offset + 12
        ).reshape(3, hidden)
        self.ids = torch.frombuffer(
            mapping, dtype=torch.int32, count=routes, offset=offset + ids_offset
        ).reshape(3, top_k)
        self.weights = torch.frombuffer(
            mapping,
            dtype=torch.float32,
            count=routes,
            offset=offset + weights_offset,
        ).reshape(3, top_k)
        self.result = torch.frombuffer(
            mapping,
            dtype=torch.float32,
            count=x_elements,
            offset=offset + result_offset,
        ).reshape(3, hidden)
        self.device_addresses = (
            address + 12,
            address + ids_offset,
            address + weights_offset,
            address + result_offset,
            address,
            address + completed_offset,
        )


class _MappedBatchPage:
    """One mmap and one CUDA registration for the ordered compute stream."""

    def __init__(self, hidden, top_k, device):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module as mailbox_module

        x_elements, routes = 3 * hidden, 3 * top_k
        ids_offset = 12 + x_elements * 2
        weights_offset = ids_offset + routes * 4
        result_offset = weights_offset + routes * 4
        slot_bytes = result_offset + x_elements * 4 + 4
        nbytes = slot_bytes
        nbytes = ((nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        self.mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_SHARED)
        self.bytes = torch.frombuffer(self.mapping, dtype=torch.uint8)
        try:
            with torch.cuda.device(device):
                self.address = mailbox_module().mailbox_register(self.bytes)
        except BaseException:
            del self.bytes
            self.mapping.close()
            raise
        self.batch = _MappedBatch(self.mapping, hidden, top_k, 0, self.address)

    def close(self, device):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module as mailbox_module

        with torch.cuda.device(device):
            mailbox_module().mailbox_unregister(self.bytes)
        del self.batch
        del self.bytes
        self.mapping.close()


class DeviceHybridPlanner:
    def __init__(self, cache):
        self.cache = cache
        self.valid = False
        self.dirty = False
        self.layouts = {}
        if not 1 <= cache.runtime.manifest.top_k <= MAX_ROUTES // 3:
            raise ValueError("device hybrid requires top_k in [1, 10]")
        if not 1 <= cache.runtime.manifest.num_experts <= 512:
            raise ValueError("device hybrid requires at most 512 experts")
        self._validate_capacity()
        from cuda.bindings import driver

        self.driver = driver
        self.flag = torch.zeros(1, dtype=torch.int32, device=cache.runtime.device)
        staged_pool = getattr(cache.runtime, "staged_pool", None)
        budgets = getattr(staged_pool, "budgets", None)
        self.max_index_slots = (
            max(budgets)
            if budgets
            else getattr(staged_pool, "maximum_budget", len(cache.policy.owner))
        )
        layers = cache.runtime.manifest.num_layers
        experts = cache.runtime.manifest.num_experts
        self.heap_index = torch.empty(
            2 * layers * experts + 3 * layers + 4 + self.max_index_slots,
            dtype=torch.int32,
            device=cache.runtime.device,
        )
        self.sequence = 0
        rows = []
        for key, (cpu_us, gpu_us) in cache.cost_model.samples.items():
            if cpu_us >= gpu_us * (1 - cache.cost_model.margin):
                continue
            tokens, gpu_routes, cpu_routes, cpu_experts, profile, rejected = key
            row = [tokens, gpu_routes, cpu_routes, cpu_experts, len(rejected), cpu_us]
            row += list(profile) + [0] * (MAX_ROUTES - len(profile))
            row += list(rejected) + [0] * (MAX_ROUTES - len(rejected))
            rows.append(row)
        self.costs = torch.tensor(
            rows, dtype=torch.float64, device=cache.runtime.device
        )
        self.costs = self.costs.reshape(-1, COST_COLUMNS)
        self.device_stats = torch.zeros(
            7, dtype=torch.int64, device=cache.runtime.device
        )
        self.observed = [0] * 7
        self.has_cpu_samples = bool(rows)
        self.compute_stream = None
        self.fill_ready = torch.cuda.Event()
        self.fill_done = torch.cuda.Event()
        # The ordered compute/fill join retires every reader before reuse,
        # including when graph warmup switches compute streams.
        top_k = cache.runtime.manifest.top_k
        self.route_buffers = {}
        for tokens in (1, 2, 3):
            self.route_buffers[tokens] = (
                torch.empty(
                    (tokens, top_k), dtype=torch.int32, device=cache.runtime.device
                ),
                torch.empty(
                    (tokens, top_k), dtype=torch.int32, device=cache.runtime.device
                ),
                torch.empty(
                    4 + 2 * tokens * top_k,
                    dtype=torch.int32,
                    device=cache.runtime.device,
                ),
            )
        # Compile at load time, before any serving invocation.
        _module()

    @property
    def device_bytes(self):
        tensors = [self.costs, self.device_stats, self.flag, self.heap_index]
        tensors += [t for buffers in self.route_buffers.values() for t in buffers]
        tensors += list(self.layouts.values())
        for name in ("lookup", "frequency", "owners", "keys"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                tensors.append(tensor)
        return sum(tensor.nbytes for tensor in tensors)

    def _validate_capacity(self, budget=None):
        c = self.cache
        # Unknown or unprofitable CPU subsets must still support all-GPU.
        required = min(c.policy.experts, 3 * c.runtime.manifest.top_k)
        if budget is None:
            budget = len(c.policy.owner)
        available = budget - len(c._reserved_slots(budget))
        if available < required:
            raise ValueError(
                f"hybrid GPU pool requires at least {required} usable slots "
                f"for decode/verify, but has {available}; increase cache budget"
            )

    def upload(self):
        c = self.cache
        # Only prefill/resize or first handoff reaches this host boundary.
        c.release_previous()
        c._join_queue(c.jobs)
        for layer in list(c.active):
            c._wait_layer(layer)
        c.current = None
        c._release_prefill_slots()
        self._validate_capacity()
        p, r = c.policy, c.runtime
        lookup = [[-1] * p.experts for _ in p.layers]
        for layer, state in enumerate(p.layers):
            for expert, slot in state.mapping.items():
                lookup[layer][expert] = slot
        self.lookup = torch.tensor(lookup, dtype=torch.int32, device=r.device)
        self.frequency = torch.tensor(p.frequency, dtype=torch.float64, device=r.device)
        # Device sentinel -1 is free, -2 is permanently reserved.
        owners = [
            -2 if s in p.reserved else (-1 if owner is None else owner)
            for s, owner in enumerate(p.owner)
        ]
        keys = [-1 if key is None else key[1] for key in p.keys]
        self.owners = torch.tensor(owners, dtype=torch.int32, device=r.device)
        self.keys = torch.tensor(keys, dtype=torch.int32, device=r.device)
        new_index = build_heap_index(
            p.frequency,
            owners,
            keys,
            p.experts,
            self.max_index_slots,
        )
        self.heap_index.copy_(new_index)
        self.layouts = {}
        # Upload address tables at an exclusive boundary, never on a decode
        # frame after a GPU decision. Layout creation could otherwise H2D-wait.
        for layer in range(r.manifest.num_layers):
            self.layout(layer)
        self.valid = True
        self.dirty = False

    def sync_host(self):
        if not self.dirty:
            return
        c, p = self.cache, self.cache.policy
        # Explicit prefill/resize/stats/close boundary. This is never decode gating.
        for event in c.runtime.layer_computed.values():
            event.synchronize()
        lookup = self.lookup.cpu().tolist()
        p.frequency = self.frequency.cpu().tolist()
        owners = self.owners.cpu().tolist()
        keys = self.keys.cpu().tolist()
        p.owner = [
            None if owner == -1 else (-1 if owner == -2 else owner) for owner in owners
        ]
        p.keys = [
            None if expert < 0 else (owners[s], expert) for s, expert in enumerate(keys)
        ]
        for layer, state in enumerate(p.layers):
            state.mapping = {e: s for e, s in enumerate(lookup[layer]) if s >= 0}
        p._index_owners()
        c.lookups.clear()
        names = (
            "evictions",
            "selected_hit_calls",
            "misses",
            "lfu_admitted_experts",
            "lfu_rejected_experts",
            "selected_cpu_experts",
            "gpu_miss_experts",
        )
        values = self.device_stats.cpu().tolist()
        for name, value, previous in zip(names, values, self.observed):
            c.stats[name] += value - previous
        c.stats["insertions"] += (
            values[3] - self.observed[3] + values[6] - self.observed[6]
        )
        self.observed = values
        self.dirty = False

    def layout(self, layer):
        key = layer
        if key not in self.layouts:
            c, r = self.cache, self.cache.runtime
            self.layouts[key] = torch.tensor(
                [
                    [
                        tensor.data_ptr(),
                        r.manifest.tensors[layer][name].nbytes
                        // r.manifest.num_experts,
                        r.miss_mailbox.pack_address
                        + r.manifest.tensors[layer][name].offset,
                    ]
                    for name, tensor in c.pool.items()
                ],
                dtype=torch.int64,
                device=r.device,
            )
        return self.layouts[key]

    def cpu_task(self, layer, batch, tokens, sequence):
        c = self.cache
        try:
            if c.failure is not None:
                self.publish(sequence)
                return
            ids = batch.ids[:tokens]
            c.cpu.run(
                layer,
                batch.x[:tokens],
                ids,
                batch.weights[:tokens],
                batch.result[:tokens],
            )
            c.stats["cpu_calls"] += 1
            self.publish(sequence)
        except BaseException as error:
            c.failure = error
            c.runtime.poisoned = True
            c.runtime.miss_mailbox._abort(error)
            # Release the selected result consumer after poisoning.
            self.publish(sequence)

    def poll(self):
        """Return one published CPU request, or None after the bounded poll."""
        page = self.cache.mapped_batch_pages[0]
        if not hasattr(self, "seen"):
            self.seen = torch.zeros(1, dtype=torch.int32)
        batch = page.batch
        if _module().poll(batch.head, self.seen) < 0:
            return None
        sequence = batch.head.item()
        self.seen.fill_(sequence)
        return batch.layer.item(), batch, batch.tokens.item(), sequence

    def publish(self, sequence):
        if self.cache.gpu_backend == "b12x":
            _module().complete(
                self.cache.mapped_batch_pages[0].batch.completed, sequence
            )
            return
        driver = self.driver
        (status,) = driver.cuStreamWriteValue32(
            driver.CUstream(self.cache.result_stream.cuda_stream),
            driver.CUdeviceptr(self.flag.data_ptr()),
            sequence,
            0,
        )
        if status != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"hybrid CPU completion publication failed: {status}")

    def run(self, layer, x, ids, weights, shared):
        c, r = self.cache, self.cache.runtime
        c._seed()
        c._check()
        if not self.valid:
            self.upload()
        compute = torch.cuda.current_stream(r.device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CPU hybrid decode must execute outside CUDA graph capture"
            )
        if self.compute_stream is None:
            self.compute_stream = compute.cuda_stream
        elif self.compute_stream != compute.cuda_stream:
            # Autotuning and breakable-graph warmup use different streams.
            # Serialize the shared pool and mapped request page explicitly.
            compute.wait_event(self.last_compute_done)
            self.compute_stream = compute.cuda_stream
        # Every decode planner, fill and preceding cache reader runs on this
        # same stream. Its ordering already permits cross-layer slot reuse.
        cache_ids, cpu_ids, plan = self.route_buffers[x.shape[0]]
        batch = None
        if self.has_cpu_samples:
            if not c.mapped_batch_pages:
                page = _MappedBatchPage(x.shape[1], ids.shape[1], r.device)
                c.mapped_batch_pages.append(page)
            batch = c.mapped_batch_pages[0].batch
            # The preceding FC2 output (or Triton merge) on this stream
            # completes before the next selected plan reuses the shared page.
            self.sequence += 1
            if self.sequence >= 2**31:
                raise RuntimeError("hybrid completion sequence exhausted")
            sequence = self.sequence
            hybrid_plan_stage_indexed(
                ids,
                self.lookup,
                self.frequency,
                self.owners,
                self.keys,
                self.costs,
                cache_ids,
                cpu_ids,
                plan,
                self.device_stats,
                layer,
                c.policy.decay,
                c.policy.admission,
                x,
                weights,
                *batch.device_addresses[:3],
                batch.device_addresses[4],
                self.sequence,
                self.heap_index,
            )
        else:
            hybrid_plan_indexed(
                ids,
                self.lookup,
                self.frequency,
                self.owners,
                self.keys,
                self.costs,
                cache_ids,
                cpu_ids,
                plan,
                self.device_stats,
                layer,
                c.policy.decay,
                c.policy.admission,
                self.heap_index,
            )
        self.dirty = True
        mailbox = r.miss_mailbox

        # One fill reads its device count; an empty plan copies nothing.
        def submit_fill():
            fill_module().mailbox_fill_grid(
                plan,
                self.layout(layer),
                ids.numel(),
                mailbox.failed_address,
                mailbox.fill_cta,
            )

        if shared is None:
            submit_fill()
        else:
            from sglang.srt.layers.moe.qwen38_cache_overlap import fill_with_overlap

            fill_with_overlap(
                compute,
                c.fill_stream,
                self.fill_ready,
                self.fill_done,
                submit_fill,
                shared,
            )
            c.stats["shared_fill_overlap_calls"] += 1
        try:
            gpu = c._gpu(x, cache_ids, weights, layer)
            # Keep both contributions in FP32 until the sole BF16 conversion.
            if batch is not None:
                output = torch.empty_like(x)
                hybrid_merge(
                    cpu_ids,
                    gpu,
                    self.flag,
                    output,
                    batch.device_addresses[3],
                    sequence,
                    mailbox.failed_address,
                    completed_address=(
                        batch.device_addresses[5] if c.gpu_backend == "b12x" else 0
                    ),
                )
        finally:
            done = torch.cuda.Event()
            done.record(compute)
            self.last_compute_done = done
            r.layer_computed[layer] = r.arena_last_use = done
        with mailbox.lock:
            mailbox.native_active = True
            mailbox.pending.append((done, time.monotonic() + c.timeout, layer))
        c.stats["device_planner_calls"] += 1
        r._record_invocation(False, ids, layer)
        r.finish_layer(layer)
        return output if batch is not None else gpu.to(x.dtype)
