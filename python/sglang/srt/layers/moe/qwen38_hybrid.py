# SPDX-License-Identifier: Apache-2.0
"""CPU A16 + GPU compute with next-layer deferred cache maintenance."""

import collections
import logging
import queue
import threading
from concurrent.futures import Future

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu, choose_all_gpu


class HybridCache:
    def __init__(self, runtime, gpu_backend):
        from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4

        if runtime.cache_pool is None or not runtime.adaptive_cache:
            raise ValueError("CPU hybrid requires an adaptive shared expert pool")
        if runtime.cache_policy != "lfu":
            raise ValueError("global hybrid cache requires cache_policy=lfu")
        if not runtime.prefill_arena:
            raise ValueError("global hybrid requires a prefill arena")
        if gpu_backend not in ("b12x", "triton"):
            raise ValueError("hybrid GPU backend must be b12x or triton")
        self.gpu_backend = gpu_backend
        self.runtime = runtime
        m = runtime.manifest
        self.policy = GlobalLfu(m.num_layers, m.num_experts, runtime.layer_cache_slots)
        self.pool = runtime.cache_pool
        self.cpu = CpuNvfp4(
            envs.SGLANG_QWEN38_CPU_LIBRARY.get(),
            runtime.pack,
            tuple(int(c) for c in envs.SGLANG_QWEN38_CPU_CORES.get()),
            gpu_backend=gpu_backend,
        )
        self.threshold = envs.SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD.get()
        if self.threshold < 1:
            raise ValueError("hybrid miss threshold must be positive")
        self.timeout = envs.SGLANG_QWEN38_CPU_READY_TIMEOUT.get()
        if not 0 < self.timeout < 3600:
            raise ValueError("hybrid ready timeout must be in (0, 3600)")
        self.closed = False
        self.readback = torch.cuda.Stream(device=runtime.device)
        self.fill_stream = torch.cuda.Stream(device=runtime.device)
        self.staging = {
            name: torch.empty(
                tensor[0].numel() * tensor.element_size(),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            for name, tensor in self.pool.items()
            if tensor.shape[0]
        }
        self.result_stream = torch.cuda.Stream(device=runtime.device)
        self.jobs = queue.Queue(maxsize=m.num_layers + 2)
        self.pending = None
        self.active = {}
        self.current = None
        self.failure = None
        self.seeded = False
        self.arena_restore = None
        self.stats = collections.Counter()
        self.b12x = {}
        self.host_buffers = {}
        self.lookups = {}

        self.worker = threading.Thread(
            target=self._work, name="qwen38-hybrid", daemon=True
        )
        self.worker.start()
        self.thread = threading.Thread(
            target=self._cpu_work, name="qwen38-cpu-a16", daemon=True
        )
        self.cpu_jobs = queue.Queue(maxsize=2)
        self.thread.start()
        logging.getLogger(__name__).info(
            "Qwen38 CPU hybrid enabled: backend=%s cores=%s library=%s global_slots=%d",
            gpu_backend,
            list(self.cpu.cores),
            self.cpu.library._name,
            len(self.policy.keys),
        )

    def _check(self):
        if self.closed:
            raise RuntimeError("CPU hybrid is closed")
        if self.failure is not None:
            raise RuntimeError("CPU hybrid background failure") from self.failure

    def _join_queue(self, jobs):
        import time

        deadline = time.monotonic() + self.timeout
        with jobs.all_tasks_done:
            while jobs.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("hybrid queue did not drain")
                jobs.all_tasks_done.wait(remaining)

    def _seed(self):
        if self.seeded:
            return
        # Import metadata only after returning legacy borrowed bytes under its lock.
        r = self.runtime
        with r.lock:
            r._raise_error()
            r._drain_arena_readers()
            r._return_arena()
            r._raise_error()
            r.arena_layer = None
            r.arena_residents = set()
            r.arena_last_use = None
            offset = 0
            for layer, slots in enumerate(self.runtime.layer_cache_slots):
                keys = self.runtime.device_states[layer]["keys"].cpu().tolist()
                frequencies = (
                    self.runtime.device_states[layer]["frequency"].cpu().tolist()
                )
                for local, expert in enumerate(keys):
                    if expert >= 0:
                        self.policy.layers[layer].mapping[expert] = offset + local
                        self.policy.keys[offset + local] = (layer, expert)
                        self.policy.frequency[layer][expert] = max(
                            1.0, frequencies[local]
                        )
                offset += slots
            if self.b12x:
                # Upstream inactive routes still read expert 0 before masking results.
                # Keep slot 0 immutable and zero so concurrent fills never race them.
                key = self.policy.keys[0]
                if key is not None:
                    self.policy.layers[key[0]].mapping.pop(key[1])
                self.policy.keys[0] = None
                self.policy.owner[0] = -1
                self.policy.reserved.add(0)
                for tensor in self.pool.values():
                    tensor[0].zero_()
                torch.cuda.current_stream(r.device).synchronize()
            self.seeded = True
            self.runtime.global_cache_slots = len(self.policy.keys)
            self.runtime.caches = {
                layer: self.pool for layer in range(self.runtime.manifest.num_layers)
            }
            self.stats["initial_global_handoff_waits"] += 1

    def release_previous(self):
        self._check()
        if self.pending is not None:
            layer, epoch, ids, done = self.pending
            self.pending = None
            self.policy.release(layer, epoch)
            future = Future()
            self.active[layer] = future
            self.jobs.put_nowait((layer, epoch, ids, done, future))

    def _wait_layer(self, layer):
        future = self.active.pop(layer, None)
        if future is not None and not future.done():
            self.stats["not_ready_waits"] += 1
            future.result(timeout=self.timeout)
        elif future is not None:
            future.result()
        self._check()

    def _wait_arena(self):
        if self.arena_restore is not None:
            self.arena_restore.result(timeout=self.timeout)
            self.arena_restore = None
        self._check()

    def layer_start(self, layer):
        # Called by the eager graph boundary before the next layer's attention.
        self.release_previous()
        self._wait_arena()
        if self.seeded:
            self._wait_layer(layer)
            epoch, mapping = self.policy.begin(layer)
            self.current = (layer, epoch, mapping)

    def _host_copy(self, tensor, stream):
        host = torch.empty_like(tensor, device="cpu", pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        tensor.record_stream(stream)
        return host

    def _cpu_work(self):
        with torch.cuda.device(self.runtime.device):
            while True:
                job = self.cpu_jobs.get()
                try:
                    if job is None:
                        self.cpu.release_worker()
                        return
                    (
                        layer,
                        epoch,
                        x,
                        ids,
                        weights,
                        ready,
                        result,
                        mapping,
                        resident_done,
                        device_input,
                        future,
                    ) = job
                    ready.synchronize()
                    rows = ids.tolist()
                    requested = {e for row in rows for e in row if e >= 0}
                    misses = requested - mapping.keys()
                    owned = self.policy.owner.count(layer)
                    self.stats["hits"] += len(requested) - len(misses)
                    self.stats["misses"] += len(misses)
                    all_gpu = choose_all_gpu(
                        len(misses), owned, len(requested), self.threshold
                    )
                    if owned == 0:
                        # Explicit slow path: wait pending cache work, then ask
                        # the sole manager to acquire free/released ownership.
                        self._join_queue(self.jobs)
                        self.stats["zero_slot_waits"] += 1
                        all_gpu = True
                    if all_gpu and misses:
                        resident_event, enqueued = resident_done
                        if not enqueued.wait(self.timeout):
                            raise TimeoutError("resident GPU launch not published")
                        filled = Future()
                        self.jobs.put_nowait(
                            (
                                "immediate",
                                layer,
                                epoch,
                                requested,
                                resident_event,
                                filled,
                            )
                        )
                        fill_result = filled.result(timeout=self.timeout)
                        if fill_result is not None:
                            self.stats["all_gpu_calls"] += 1
                            future.set_result(("gpu", misses, fill_result))
                            continue
                    ids = ids.clone()
                    for t, row in enumerate(rows):
                        for k, expert in enumerate(row):
                            if expert not in misses:
                                ids[t, k] = -1
                    if misses:
                        with torch.cuda.stream(self.readback):
                            self.readback.wait_event(ready)
                            device_x, device_weights = device_input
                            x.copy_(device_x, non_blocking=True)
                            weights.copy_(device_weights, non_blocking=True)
                            device_x.record_stream(self.readback)
                            device_weights.record_stream(self.readback)
                            copied_input = torch.cuda.Event()
                            copied_input.record(self.readback)
                        copied_input.synchronize()
                        self.cpu.run(layer, x, ids, weights, result)
                    else:
                        result.zero_()
                    future.set_result(("cpu" if misses else "hit", misses, None))
                except BaseException as error:
                    self.failure = error
                    if job is not None:
                        job[-1].set_exception(error)
                finally:
                    self.cpu_jobs.task_done()

    def _copy(self, layer, fills, done):
        if self.runtime._registered_prefill_enabled():
            from sglang.kernels.ops.moe.qwen38_expert_cache import _module

            # The immutable pack is already registered for the runtime's entire
            # lifetime. Queue DMA directly and publish only after the final
            # event; no staging buffer reuse forces a host wait per expert.
            source = self.runtime.miss_mailbox.pack_view
            with torch.cuda.stream(self.fill_stream):
                self.fill_stream.wait_event(done)
                for expert, slot in fills:
                    for name, tensor in self.pool.items():
                        record = self.runtime.manifest.tensors[layer][name]
                        count = record.nbytes // self.runtime.manifest.num_experts
                        begin = record.offset + expert * count
                        _module().registered_copy(
                            tensor[slot].reshape(-1).view(torch.uint8),
                            source[begin : begin + count],
                        )
                        self.stats["fill_bytes"] += count
                    self.stats["insertions"] += 1
                copied = torch.cuda.Event()
                copied.record(self.fill_stream)
            return copied
        # One reusable expert worth of pinned staging, regardless of batch size.
        # Only the manager waits before reuse; serving overlaps the DMA.
        copied = done
        for expert, slot in fills:
            with torch.cuda.stream(self.fill_stream):
                self.fill_stream.wait_event(done)
                for name, tensor in self.pool.items():
                    record = self.runtime.manifest.tensors[layer][name]
                    count = record.nbytes // self.runtime.manifest.num_experts
                    source = torch.frombuffer(
                        self.cpu.mapping,
                        dtype=torch.uint8,
                        count=count,
                        offset=record.offset + expert * count,
                    )
                    staging = self.staging[name]
                    staging.copy_(source)
                    tensor[slot].reshape(-1).view(torch.uint8).copy_(
                        staging, non_blocking=True
                    )
                    self.stats["fill_bytes"] += staging.numel()
                copied = torch.cuda.Event()
                copied.record(self.fill_stream)
            copied.synchronize()
            self.stats["insertions"] += 1
        return copied

    def _work(self):
        with torch.cuda.device(self.runtime.device):
            while True:
                job = self.jobs.get()
                try:
                    if job is None:
                        return
                    if job[0] == "restore_arena":
                        _, records, done, future = job
                        done.synchronize()
                        for layer, fills in records.items():
                            copied = self._copy(layer, fills, done)
                            copied.synchronize()
                        future.set_result(None)
                        continue
                    if job[0] == "immediate":
                        _, layer, epoch, requested, done, future = job
                        done.synchronize()
                        if self.policy.owner.count(layer) == 0:
                            # Exceptional slow path can reclaim globally only
                            # after the previous owners' readers and fills finish.
                            for owner, state in enumerate(self.policy.layers):
                                if (
                                    owner != layer
                                    and state.phase == "locked"
                                    and state.epoch
                                ):
                                    event = self.runtime.layer_computed.get(owner)
                                    if event is not None:
                                        event.synchronize()
                            evicted = self.policy.reclaim_zero_slot(
                                layer, epoch, requested
                            )
                            self.stats["zero_slot_global_evictions"] += len(evicted)
                        fills = self.policy.immediate(layer, epoch, requested)
                        if fills is None:
                            raise RuntimeError(
                                "all-GPU plan exceeded current-layer ownership"
                            )
                        else:
                            copied = self._copy(layer, fills, done)
                            copied.synchronize()
                            mapping = self.policy.layers[layer].mapping
                            for expert, slot in fills:
                                mapping[expert] = slot
                            future.set_result(dict(mapping))
                        continue
                    layer, epoch, ids, done, future = job
                    done.synchronize()  # never a main-thread fill wait
                    requested = [e for row in ids.tolist() for e in row if e >= 0]
                    evicted, fills = self.policy.plan(layer, epoch, requested)
                    self.stats["evictions"] += len(evicted)
                    copied = self._copy(layer, fills, done)
                    copied.synchronize()
                    self.policy.commit(layer, epoch, fills)
                    future.set_result(None)
                except BaseException as error:
                    self.failure = error
                    if job is not None:
                        job[-1].set_exception(error)
                finally:
                    self.jobs.task_done()

    def _gpu(self, x, slot_ids, weights, layer):
        if self.gpu_backend == "b12x":
            native = self.b12x.get(layer)
            if native is None:
                raise RuntimeError(
                    f"B12X was selected but layer {layer} is not initialized"
                )
            return native.run(
                x,
                slot_ids,
                weights,
                {**self.pool, **self.runtime.layer_scales[layer]},
                output_dtype=torch.float32,
            )
        # Explicit triton backend retains the FP32 partition contract.
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe

        return cached_w4a16_moe(
            x,
            slot_ids,
            weights,
            {**self.pool, **self.runtime.layer_scales[layer]},
            split_k=0,
            verify_mode="baseline",
            output_dtype=torch.float32,
        )

    def _remap(self, ids, mapping):
        # Lookup publication is only at layer entry, after previous dirty clears.
        key = tuple(sorted(mapping.items()))
        entry = self.lookups.get(key)
        if entry is None:
            lookup = torch.full(
                (self.runtime.manifest.num_experts,),
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for expert, slot in mapping.items():
                lookup[expert] = slot
            device_lookup = lookup.to(self.runtime.device, non_blocking=True)
            # Bound retained pointer tables; no weights or activations retained.
            if len(self.lookups) >= self.runtime.manifest.num_layers * 2:
                self.lookups.clear()
            self.lookups[key] = (lookup, device_lookup)
        else:
            _, device_lookup = entry
        return torch.where(ids >= 0, device_lookup[ids.clamp_min(0).long()], -1).int()

    def run(self, layer, x, ids, weights, shared=None):
        self._seed()
        self._wait_arena()
        self._wait_layer(layer)
        if self.current is not None:
            current_layer, epoch, mapping = self.current
            if current_layer != layer:
                raise RuntimeError("hybrid layer-entry/expert order differs")
            self.current = None
        else:
            epoch, mapping = self.policy.begin(layer)
        compute = torch.cuda.current_stream(self.runtime.device)
        # Reuse buffers by layer and shape; its previous management was joined.
        shape = (layer, tuple(x.shape), tuple(ids.shape))
        buffers = self.host_buffers.get(shape)
        if buffers is None:
            buffers = (
                torch.empty_like(x, device="cpu", pin_memory=True),
                torch.empty_like(ids, device="cpu", pin_memory=True),
                torch.empty_like(
                    weights, device="cpu", dtype=torch.float32, pin_memory=True
                ),
                torch.empty_like(x, device="cpu", dtype=torch.float32, pin_memory=True),
            )
            self.host_buffers[shape] = buffers
        host_x, host_ids, host_weights, host_result = buffers
        with torch.cuda.stream(self.readback):
            self.readback.wait_stream(compute)
            host_ids.copy_(ids, non_blocking=True)
            ids.record_stream(self.readback)
            ready = torch.cuda.Event()
            ready.record(self.readback)
        resident_done = torch.cuda.Event()
        enqueued = threading.Event()
        future = Future()
        self.cpu_jobs.put_nowait(
            (
                layer,
                epoch,
                host_x,
                host_ids,
                host_weights,
                ready,
                host_result,
                mapping,
                (resident_done, enqueued),
                (x, weights),
                future,
            )
        )
        try:
            if shared is not None:
                shared()
            resident_slots = self._remap(ids, mapping)
            gpu = self._gpu(x, resident_slots, weights, layer)
        finally:
            resident_done.record(compute)
            enqueued.set()
        mode, misses, filled_mapping = future.result(timeout=self.timeout)
        self.stats["result_join_waits"] += 1
        if mode == "gpu":
            # All-fill branch explicitly pays fill readiness. Resident work was
            # already enqueued and cannot be overwritten by current-layer fill.
            miss_mapping = {e: filled_mapping[e] for e in misses}
            gpu = gpu + self._gpu(x, self._remap(ids, miss_mapping), weights, layer)
        elif mode == "cpu":
            with torch.cuda.stream(self.result_stream):
                result = host_result.to(self.runtime.device, non_blocking=True)
                delivered = torch.cuda.Event()
                delivered.record(self.result_stream)
            compute.wait_event(delivered)
            result.record_stream(compute)
            gpu = gpu + result
        done = torch.cuda.Event()
        done.record(compute)
        self.runtime.layer_computed[layer] = done
        self.pending = (layer, epoch, host_ids, done)
        self.runtime._record_invocation(False, ids, layer)
        self.runtime.finish_layer(layer)
        return gpu.to(x.dtype)

    def drain(self):
        if self.closed:
            return
        if self.failure is None:
            self.release_previous()
        self._join_queue(self.jobs)
        self._join_queue(self.cpu_jobs)
        self._wait_arena()
        for layer in list(self.active):
            self._wait_layer(layer)
        self._check()

    def prefill(self, layer, ids, execute, shared=None):
        """Prefill uses the same global mapping and a leased GPU arena.

        CPU A16 is reserved for tiny decode/verify. Large prefill fills its
        misses on GPU; cache retention and subsequent admission stay global.
        """
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts

        self._seed()
        self._wait_arena()
        self._wait_layer(layer)
        if self.current is not None:
            current_layer, epoch, mapping = self.current
            if current_layer != layer:
                raise RuntimeError("prefill layer-entry/expert order differs")
            self.current = None
        else:
            epoch, mapping = self.policy.begin(layer)
        r = self.runtime
        if r.arena is None:
            raise ValueError("global cache prefill requires an arena")
        compute = torch.cuda.current_stream(r.device)
        host_ids = ids.cpu()
        requested = {int(e) for row in host_ids.tolist() for e in row if e >= 0}
        if bool(((host_ids < -1) | (host_ids >= r.manifest.num_experts)).any()):
            raise ValueError("prefill route outside expert range")
        # An arena borrowed from the physical tail may overlap ANY layer.
        # Acquire an exclusive lease; preserve its keys and restore source bytes
        # behind the prefill result before another window can read the tail.
        records = {}
        if r.borrowed_arena:
            self._join_queue(self.jobs)
            for event in r.layer_computed.values():
                event.synchronize()
            limit = len(self.policy.keys) - r.manifest.num_experts
            for slot in range(limit, len(self.policy.keys)):
                key = self.policy.keys[slot]
                if key is not None:
                    records.setdefault(key[0], []).append((key[1], slot))
            mapping = {e: slot for e, slot in mapping.items() if slot < limit}
        else:
            limit = len(self.policy.keys)
            if r.arena_last_use is not None:
                r.arena_last_use.synchronize()
        misses = sorted(requested - mapping.keys())
        hits = requested & mapping.keys()
        if misses:
            r.transfer.wait_stream(compute)
            r._fill_arena_selected(layer, misses)
            compute.wait_stream(r.transfer)
        if shared is not None:
            shared()
        if hits:
            lookup = torch.full(
                (r.manifest.num_experts,),
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for expert, slot in mapping.items():
                lookup[expert] = slot
            prepared = PreparedExperts(
                {**r.arena, **r.layer_scales[layer]},
                ids,
                r.manifest.num_experts,
                True,
                cache_tensors={**self.pool, **r.layer_scales[layer]},
                cache_experts={e: mapping[e] for e in hits},
                arena_experts=misses,
                cache_lookup=lookup.to(r.device, non_blocking=True),
                cache_slot_limit=limit,
            )
        else:
            prepared = PreparedExperts(
                {**r.arena, **r.layer_scales[layer]}, ids, r.manifest.num_experts, True
            )
        result = execute(prepared).hidden_states
        done = torch.cuda.Event()
        done.record(compute)
        r.layer_computed[layer] = done
        r.arena_last_use = done
        if records:
            self.arena_restore = Future()
            self.jobs.put_nowait(("restore_arena", records, done, self.arena_restore))
        self.pending = (layer, epoch, host_ids, done)
        self.stats["prefill_global_hits"] += len(hits)
        self.stats["prefill_gpu_misses"] += len(misses)
        r._record_invocation(bool(misses), ids, layer)
        r.finish_layer(layer)
        return result

    def resize(self, budget):
        """Exclusive result-boundary operation; payload before mapping before unmap."""
        r = self.runtime
        try:
            self._seed()
            self.drain()
            self.current = None
            for event in r.layer_computed.values():
                event.synchronize()
            plan = self.policy.plan_resize(budget)
            # VMM growth was committed by StagedKVLoan before this method;
            # shrink must retain the old backing until all moves complete.
            source = r.staged_pool.pool_views(max(len(self.policy.keys), budget))
            stream = torch.cuda.current_stream(r.device)
            with torch.cuda.stream(stream):
                for name, tensor in source.items():
                    for old, new in plan.moves:
                        tensor[new].copy_(tensor[old])
                done = torch.cuda.Event()
                done.record(stream)
            done.synchronize()
            self.policy.commit_resize(plan)
            r.cache_pool = r.staged_pool.pool_views(budget)
            self.pool = r.cache_pool
            self.lookups.clear()
            r.global_cache_slots = budget
            # Per-layer tensors are compatibility aliases, never slot owners.
            r.caches = {layer: self.pool for layer in range(r.manifest.num_layers)}
            r._refresh_staged_arena()
            r.arena_layer = None
            r.arena_residents = set()
            r.arena_last_use = None
            r.transfer.wait_stream(stream)
            self.fill_stream.wait_stream(stream)
            self.stats["resize_events"] += 1
            self.stats["resize_moves"] += len(plan.moves)
            self.stats["resize_evictions"] += len(plan.evicted)
            r.adaptive_migrated_experts += len(plan.moves)
            r.adaptive_evicted_experts += len(plan.evicted)
        except BaseException as error:
            self.failure = error
            r.poisoned = True
            raise

    def reset(self):
        # Explicit drain retains global keys, heat and ownership across requests.
        self.drain()

    def close(self):
        if self.closed:
            return
        error = None
        try:
            self.drain()
        except BaseException as exc:
            error = exc
        finally:
            self.jobs.put(None)
            self.cpu_jobs.put(None)
            self.worker.join(timeout=self.timeout)
            self.thread.join(timeout=self.timeout)
            if self.worker.is_alive() or self.thread.is_alive():
                raise RuntimeError(
                    "hybrid workers did not exit; source mapping retained"
                ) from error
            self.cpu.close()
            self.closed = True
        if error is not None:
            raise error
