# SPDX-License-Identifier: Apache-2.0
"""CPU A16 + GPU compute with next-layer deferred cache maintenance."""

import collections
import logging
import queue
import threading
import time
from concurrent.futures import Future

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu


class _StagedCopy:
    """Pollable fallback sharing one pinned expert buffer across all layers."""

    def __init__(self, cache, layer, fills, done):
        self.cache, self.layer, self.fills, self.done = cache, layer, fills, done
        self.index = 0
        self.copied = None

    def query(self):
        if self.copied is not None and not self.copied.query():
            return False
        if self.index == len(self.fills):
            return self.done.query() if self.copied is None else True
        cache = self.cache
        busy = getattr(cache, "staging_busy", None)
        if busy is not None and not busy.query():
            return False
        expert, slot = self.fills[self.index]
        with torch.cuda.stream(cache.fill_stream):
            cache.fill_stream.wait_event(self.done)
            for name, tensor in cache.pool.items():
                record = cache.runtime.manifest.tensors[self.layer][name]
                count = record.nbytes // cache.runtime.manifest.num_experts
                source = torch.frombuffer(
                    cache.cpu.mapping,
                    dtype=torch.uint8,
                    count=count,
                    offset=record.offset + expert * count,
                )
                staging = cache.staging[name]
                staging.copy_(source)
                tensor[slot].reshape(-1).view(torch.uint8).copy_(
                    staging, non_blocking=True
                )
                cache.stats["fill_bytes"] += count
            self.copied = torch.cuda.Event()
            self.copied.record(cache.fill_stream)
        cache.staging_busy = self.copied
        cache.stats["insertions"] += 1
        self.index += 1
        return False

    def synchronize(self):
        # Explicit drain/debug callers only; the manager exclusively polls.
        while not self.query():
            time.sleep(0.0001)


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
        self.jobs = queue.Queue()
        self.pending = None
        self.active = {}
        self.current = None
        self.failure = None
        self.seeded = False
        self.readable = {}
        self.stats = collections.Counter()
        self.b12x = {}
        self.host_buffers = {}
        self.lookups = {}
        self.fill_layouts = {}

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
            # The borrowed arena remains disjoint from persistent cache entries.
            # This small capacity reservation avoids per-prefill eviction/drain
            # cycles and lets admissions read the immutable host pack instead.
            for slot in self._reserved_slots(len(self.policy.keys)):
                key = self.policy.keys[slot]
                if key is not None:
                    self.policy.layers[key[0]].mapping.pop(key[1])
                self.policy.keys[slot] = None
                self.policy.set_owner(slot, -1)
                self.policy.reserved.add(slot)
            if self.b12x:
                # Inactive B12X routes read slot zero before masking their output.
                for tensor in self.pool.values():
                    tensor[0].zero_()
                torch.cuda.current_stream(r.device).synchronize()
            self.seeded = True
            self.stats["usable_cache_slots"] = len(self.policy.keys) - len(
                self.policy.reserved
            )
            self.runtime.global_cache_slots = len(self.policy.keys)
            self.runtime.caches = {
                layer: self.pool for layer in range(self.runtime.manifest.num_layers)
            }
            self.stats["initial_global_handoff_waits"] += 1

    def _reserved_slots(self, budget):
        reserved = {0} if self.b12x and budget else set()
        if getattr(self.runtime, "borrowed_arena", False):
            if budget <= self.runtime.manifest.num_experts:
                raise ValueError("hybrid pool must exceed the borrowed arena")
            reserved.update(range(budget - self.runtime.manifest.num_experts, budget))
        return reserved

    def release_previous(self):
        self._check()
        if self.pending is None:
            return
        layer, epoch, requested, done = self.pending
        self.pending = None
        if epoch is None:
            previous = self.active.get(layer)
            if previous is not None and not previous.done():
                # This window only read survivors of the in-flight plan. It must
                # not start another mutation, but its CPU misses still count.
                self.jobs.put_nowait(("observe", layer, requested))
                return
            if previous is not None:
                previous.result()
                self.active.pop(layer)
            epoch, _ = self.policy.begin(layer)
        self.policy.release(layer, epoch)
        # Clear before enqueue: a manager may invalidate/overwrite any victim.
        # Once planning finishes it publishes just the untouched survivors here.
        self.readable[layer] = {}
        future = Future()
        self.active[layer] = future
        self.jobs.put_nowait((layer, epoch, requested, done, future))

    def _wait_layer(self, layer):
        # Exclusive boundaries (resize/close) drain; ordinary forwards never do.
        future = self.active.pop(layer, None)
        if future is not None:
            future.result(timeout=self.timeout)
        self._check()

    def _begin_layer(self, layer):
        self._check()
        future = self.active.get(layer)
        if future is not None:
            if not future.done():
                self.stats["maintenance_bypasses"] += 1
                return None, self.readable.get(layer, {})
            future.result()
            self.active.pop(layer)
        return self.policy.begin(layer)

    def layer_start(self, layer):
        # No copy-completion wait at the eager graph boundary. Pending fills
        # remain misses, and CPU execution covers them until publication.
        self.release_previous()
        if self.seeded:
            epoch, mapping = self._begin_layer(layer)
            self.current = (layer, epoch, mapping)

    def _cpu_work(self):
        with torch.cuda.device(self.runtime.device):
            while True:
                job = self.cpu_jobs.get()
                try:
                    if job is None:
                        self.cpu.release_worker()
                        return
                    layer, x, ids, weights, ready, result, mapping, future = job
                    ready.synchronize()
                    rows = ids.tolist()
                    requested = tuple(
                        sorted({e for row in rows for e in row if e >= 0})
                    )
                    misses = set(requested) - mapping.keys()
                    self.stats["hits"] += len(requested) - len(misses)
                    self.stats["misses"] += len(misses)
                    if misses:
                        ids = ids.clone()
                        masked_ids = ids.numpy()
                        for t, row in enumerate(rows):
                            for k, expert in enumerate(row):
                                if expert not in misses:
                                    masked_ids[t, k] = -1
                        self.cpu.run(layer, x, ids, weights, result)
                        self.stats["cpu_calls"] += 1
                        self.stats["cpu_miss_experts"] += len(misses)
                    future.set_result((bool(misses), requested))
                except BaseException as error:
                    self.failure = error
                    if job is not None:
                        job[-1].set_exception(error)
                finally:
                    self.cpu_jobs.task_done()

    def _copy(self, layer, fills, done):
        if self.runtime._registered_prefill_enabled():
            from sglang.kernels.ops.moe.qwen38_expert_cache import _module

            mailbox = self.runtime.miss_mailbox
            if hasattr(mailbox, "pack_address"):
                # Reuse the legacy device fill kernel. A single descriptor
                # transfer/launch replaces per-expert, per-tensor host dispatch.
                with torch.cuda.stream(self.fill_stream):
                    self.fill_stream.wait_event(done)
                    if fills:
                        layouts = getattr(self, "fill_layouts", None)
                        if layouts is None:
                            self.fill_layouts = layouts = {}
                        table = layouts.get(layer)
                        if table is None:
                            rows = []
                            for name, tensor in self.pool.items():
                                record = self.runtime.manifest.tensors[layer][name]
                                rows.append(
                                    [
                                        tensor.data_ptr(),
                                        record.nbytes
                                        // self.runtime.manifest.num_experts,
                                        mailbox.pack_address + record.offset,
                                    ]
                                )
                            table = torch.tensor(
                                rows, dtype=torch.int64, pin_memory=True
                            ).to(self.runtime.device, non_blocking=True)
                            layouts[layer] = table
                        for start in range(0, len(fills), 32):
                            batch = fills[start : start + 32]
                            values = [0, len(batch), 0, 0]
                            values.extend(value for pair in batch for value in pair)
                            plan = torch.tensor(
                                values, dtype=torch.int32, pin_memory=True
                            ).to(self.runtime.device, non_blocking=True)
                            _module().mailbox_fill_grid(
                                plan,
                                table,
                                len(batch),
                                mailbox.failed_address,
                                mailbox.fill_cta,
                            )
                            self.stats["mailbox_fill_calls"] += 1
                        self.stats["fill_bytes"] += len(fills) * sum(
                            record.nbytes // self.runtime.manifest.num_experts
                            for name, record in self.runtime.manifest.tensors[
                                layer
                            ].items()
                            if name in self.pool
                        )
                        self.stats["insertions"] += len(fills)
                    copied = torch.cuda.Event()
                    copied.record(self.fill_stream)
                return copied
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
        # No registered pack: a pollable task leases the bounded staging
        # buffer only after its previous DMA completes. Other layers can still
        # plan, submit registered transfers and publish while this task waits.
        return _StagedCopy(self, layer, fills, done)

    def _complete_copies(self, copies):
        remaining = []
        for layer, epoch, fills, copied, future in copies:
            try:
                if not copied.query():
                    remaining.append((layer, epoch, fills, copied, future))
                    continue
                self.policy.commit(layer, epoch, fills)
                future.set_result(None)
            except BaseException as error:
                self.failure = error
                future.set_exception(error)
            finally:
                if future.done():
                    self.jobs.task_done()
        return remaining

    def _submit_ready(self, readers, copies):
        remaining = []
        for layer, epoch, requested, done, future in readers:
            deferred = False
            try:
                if not done.query():
                    remaining.append((layer, epoch, requested, done, future))
                    deferred = True
                    continue
                # Releasing cold slots to the global free pool is safe only
                # after their old GPU readers finish. Query; never host-wait.
                evicted, fills = self.policy.plan(layer, epoch, requested)
                self.stats["evictions"] += len(evicted)
                self.readable[layer] = dict(self.policy.layers[layer].mapping)
                if fills:
                    copied = self._copy(layer, fills, done)
                    copies.append((layer, epoch, fills, copied, future))
                    deferred = True
                else:
                    self.policy.commit(layer, epoch, fills)
                    future.set_result(None)
            except BaseException as error:
                self.failure = error
                future.set_exception(error)
            finally:
                if not deferred:
                    self.jobs.task_done()
        return remaining

    def _work(self):
        # Keep several layers' DMA in flight. Neither reader completion nor copy
        # completion stalls dispatch of another layer's maintenance.
        copies = []
        readers = []
        with torch.cuda.device(self.runtime.device):
            while True:
                copies = self._complete_copies(copies)
                readers = self._submit_ready(readers, copies)
                try:
                    job = self.jobs.get(timeout=0.0001 if copies or readers else None)
                except queue.Empty:
                    continue
                deferred = False
                try:
                    if job is None:
                        # A timed-out drain must not let close unmap a source
                        # while DMA still reads it. Keep the worker alive until
                        # all submitted work retires; close retains the mapping
                        # if joining this worker times out.
                        while copies or readers:
                            copies = self._complete_copies(copies)
                            readers = self._submit_ready(readers, copies)
                            if copies or readers:
                                time.sleep(0.0001)
                        return
                    if job[0] == "observe":
                        _, layer, requested = job
                        self.policy.observe(layer, requested)
                        continue
                    readers.append(job)
                    deferred = True
                except BaseException as error:
                    self.failure = error
                    if job is not None and job[0] != "observe":
                        job[-1].set_exception(error)
                finally:
                    if not deferred:
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
        # Each table belongs to an immutable compute snapshot. In-flight fills
        # stay absent; survivor snapshots can also be used during maintenance.
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
            # Scalar Torch assignments dispatch once per resident expert.
            # Populate the pinned int32 buffer in one host array operation.
            if mapping:
                lookup.numpy()[list(mapping)] = list(mapping.values())
            device_lookup = lookup.to(self.runtime.device, non_blocking=True)
            # Bound retained pointer tables; no weights or activations retained.
            if len(self.lookups) >= self.runtime.manifest.num_layers * 2:
                self.lookups.clear()
            self.lookups[key] = (lookup, device_lookup)
        else:
            _, device_lookup = entry
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes

        return cache_partition_routes(ids, device_lookup, len(self.policy.keys))[0]

    def run(self, layer, x, ids, weights, shared=None):
        self._seed()
        if self.current is not None:
            current_layer, epoch, mapping = self.current
            if current_layer != layer:
                raise RuntimeError("hybrid layer-entry/expert order differs")
            self.current = None
        else:
            epoch, mapping = self._begin_layer(layer)
        compute = torch.cuda.current_stream(self.runtime.device)
        # CPU jobs finish before return. The next readback fence also covers
        # the previous result upload before its pinned output is reused.
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
            # One readback fence instead of a route decision round trip followed
            # by a second activation/weight readback. Resident GPU work overlaps it.
            for host, device in ((host_ids, ids), (host_x, x), (host_weights, weights)):
                host.copy_(device, non_blocking=True)
                device.record_stream(self.readback)
            ready = torch.cuda.Event()
            ready.record(self.readback)
        future = Future()
        self.cpu_jobs.put_nowait(
            (layer, host_x, host_ids, host_weights, ready, host_result, mapping, future)
        )
        if shared is not None:
            shared()
        gpu = (
            self._gpu(x, self._remap(ids, mapping), weights, layer)
            if mapping
            else torch.zeros_like(x, dtype=torch.float32)
        )
        if not future.done():
            started = time.perf_counter()
            self.stats["result_join_waits"] += 1
            has_misses, requested = future.result(timeout=self.timeout)
            self.stats["result_join_seconds"] += time.perf_counter() - started
        else:
            has_misses, requested = future.result()
        if has_misses:
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
        self.pending = (layer, epoch, requested, done)
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
        for layer in list(self.active):
            self._wait_layer(layer)
        self._check()

    def prefill(self, layer, ids, execute, shared=None):
        """Prefill uses the same global mapping and a disjoint GPU arena.

        CPU A16 is reserved for tiny decode/verify. Large prefill fills its
        misses on GPU; cache retention and subsequent admission stay global.
        """
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts

        self._seed()
        if self.current is not None:
            current_layer, epoch, mapping = self.current
            if current_layer != layer:
                raise RuntimeError("prefill layer-entry/expert order differs")
            self.current = None
        else:
            epoch, mapping = self._begin_layer(layer)
        r = self.runtime
        if r.arena is None:
            raise ValueError("global cache prefill requires an arena")
        compute = torch.cuda.current_stream(r.device)
        host_ids = ids.cpu()
        requested = {int(e) for row in host_ids.tolist() for e in row if e >= 0}
        if bool(((host_ids < -1) | (host_ids >= r.manifest.num_experts)).any()):
            raise ValueError("prefill route outside expert range")
        # The borrowed tail is reserved at handoff/resize, so no layer's
        # persistent readers or admissions can overlap the prefill arena.
        limit = len(self.policy.keys)
        if r.borrowed_arena:
            limit -= r.manifest.num_experts
        if r.arena_last_use is not None:
            compute.wait_event(r.arena_last_use)
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
            # Scalar Torch assignments dispatch once per resident expert.
            # Populate the pinned int32 buffer in one host array operation.
            if mapping:
                lookup.numpy()[list(mapping)] = list(mapping.values())
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
        self.pending = (layer, epoch, tuple(sorted(requested)), done)
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
            plan = self.policy.plan_resize(
                budget, reserved=self._reserved_slots(budget)
            )
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
            self.stats["usable_cache_slots"] = budget - len(self.policy.reserved)
            self.readable.clear()
            r.cache_pool = r.staged_pool.pool_views(budget)
            self.pool = r.cache_pool
            self.lookups.clear()
            self.fill_layouts = {}
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
            self.stats["resize_retained_experts"] += sum(
                key is not None for key in plan.keys
            )
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
