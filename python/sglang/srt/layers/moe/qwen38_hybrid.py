# SPDX-License-Identifier: Apache-2.0
"""Optional CPU A16 contribution for LFU-rejected GPU cache misses."""

import collections
import hashlib
import logging
import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.qwen38_hybrid_policy import GlobalLfu, HybridCostModel


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
        cpu_name = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        self.cost_metadata = dict(
            backend=gpu_backend,
            hidden=m.hidden_size,
            width=m.intermediate_size,
            top_k=m.top_k,
            cores=len(self.cpu.cores),
            gpu=torch.cuda.get_device_name(runtime.device),
            cpu=cpu_name,
            transport=(
                "registered_dma"
                if runtime._registered_prefill_enabled()
                else f"staged:{runtime.packed_prefill}:{runtime.staging.numel()}"
            ),
            runtime_version=self._cost_version(),
        )
        # Emit the fingerprint before validation so a calibration author can
        # recover the required metadata even from a mismatched table.
        logging.getLogger(__name__).info(
            "Qwen38 hybrid cost calibration metadata: %s", self.cost_metadata
        )
        self.cost_model = HybridCostModel(
            envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.get(), **self.cost_metadata
        )
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
        self.mapped_batch_pages = []
        self.lookups = {}
        self.fill_layouts = {}
        self.prepared_windows = set()

        from sglang.srt.layers.moe.qwen38_hybrid_device import DeviceHybridPlanner

        if not hasattr(runtime.miss_mailbox, "pack_address"):
            raise ValueError("device CPU hybrid requires registered mailbox transport")
        self.device_planner = DeviceHybridPlanner(self)
        self.worker = threading.Thread(
            target=self._work, name="qwen38-hybrid", daemon=True
        )
        self.worker.start()
        self.thread = threading.Thread(
            target=self._cpu_work, name="qwen38-cpu-a16", daemon=True
        )
        self.cpu_jobs = queue.Queue()
        self.thread.start()
        logging.getLogger(__name__).info(
            "Qwen38 CPU hybrid enabled: backend=%s cores=%s library=%s global_slots=%d "
            "calibrated_workloads=%d uncalibrated_misses=gpu",
            gpu_backend,
            list(self.cpu.cores),
            self.cpu.library._name,
            len(self.policy.keys),
            len(self.cost_model.samples),
        )

    def _cost_version(self):
        """Invalidate measurements when the native CPU or GPU adapter changes."""
        from importlib.metadata import version

        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in (
            Path(__file__),
            root / "moe/qwen38_hybrid_policy.py",
            root / "moe/qwen38_hybrid_device.py",
            root / "moe/qwen38_heap_index.py",
            root.parents[1] / "kernels/jit/csrc/moe/qwen38_hybrid_plan.cuh",
            root.parents[1] / "kernels/jit/csrc/moe/qwen38_mailbox.cc",
            root.parents[1] / "kernels/ops/moe/qwen38_hybrid_plan.py",
            root.parents[1] / "kernels/jit/csrc/moe/qwen38_heap_index.cuh",
            root / "quantization/qwen38_cached_b12x.py",
            root / "quantization/qwen38_b12x_hybrid.py",
            root / "quantization/qwen38_cached_w4a16.py",
            root.parents[1] / "kernels/ops/moe/cpu_nvfp4/__init__.py",
            Path(self.cpu.library._name),
        ):
            digest.update(source.read_bytes())
        digest.update(
            repr(sorted(self.runtime.manifest.scale_encoding.items())).encode()
        )
        digest.update((root / "quantization/qwen38_b12x_csf.py").read_bytes())
        digest.update((root.parents[1] / "kernels/ops/moe/qwen38_csf.py").read_bytes())
        digest.update(torch.__version__.encode())
        if self.gpu_backend == "b12x":
            digest.update(version("flashinfer-python").encode())
        return digest.hexdigest()

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
        return reserved

    def _release_prefill_slots(self):
        slots = getattr(self, "prefill_reserved", set())
        if not slots:
            return
        if self.runtime.arena_last_use is not None:
            self.runtime.arena_last_use.synchronize()
        for slot in slots:
            self.policy.reserved.remove(slot)
            self.policy.set_owner(slot, None)
        self.prefill_reserved = set()
        self.stats["usable_cache_slots"] = len(self.policy.keys) - len(
            self.policy.reserved
        )

    def _borrow_prefill_slots(self):
        r = self.runtime
        if not r.borrowed_arena:
            return
        # Prefill alone borrows the legacy full-layer view. Decode never keeps
        # this suffix reserved; retire old readers before invalidating payloads.
        self.drain()
        for event in r.layer_computed.values():
            event.synchronize()
        slots = set(
            range(len(self.policy.keys) - r.manifest.num_experts, len(self.policy.keys))
        ) - self._reserved_slots(len(self.policy.keys))
        for slot in slots:
            key = self.policy.keys[slot]
            if key is not None:
                self.policy.layers[key[0]].mapping.pop(key[1])
                self.stats["prefill_borrow_evictions"] += 1
            self.policy.keys[slot] = None
            self.policy.set_owner(slot, -1)
            self.policy.reserved.add(slot)
        self.prefill_reserved = slots
        self.lookups.clear()

    def release_previous(self):
        self._check()
        if self.pending is None:
            return
        layer, epoch, requested, done, *options = self.pending
        prefill_arena = bool(options and options[0])
        self.pending = None
        if (layer, epoch) in getattr(self, "prepared_windows", ()):
            # Admission and observation already happened before this computation.
            # Retire its compute window without a second fill/heat update.
            self.prepared_windows.remove((layer, epoch))
            self.policy.release(layer, epoch)
            self.policy.commit(layer, epoch, [])
            return
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
        self.jobs.put_nowait(
            (layer, epoch, requested, done, future, False, prefill_arena)
        )

    def _wait_layer(self, layer):
        # Used by exclusive boundaries and admission preparation.
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
        # Defer admission preparation to run, after its actual routes are known.
        self.release_previous()
        if self.seeded and not hasattr(self, "device_planner"):
            epoch, mapping = self._begin_layer(layer)
            self.current = (layer, epoch, mapping)

    def _cpu_work(self):
        with torch.cuda.device(self.runtime.device):
            while True:
                polling = self.mapped_batch_pages and self.device_planner.valid
                if polling:
                    request = self.device_planner.poll()
                    if request is not None:
                        self.device_planner.cpu_task(*request)
                        continue
                try:
                    job = self.cpu_jobs.get_nowait()
                except queue.Empty:
                    if not polling:
                        time.sleep(0.001)
                    continue
                try:
                    if job is None:
                        self.cpu.release_worker()
                        return
                    layer, x, ids, weights, ready, result, future = job
                    ready.synchronize()
                    self.cpu.run(layer, x, ids, weights, result)
                    self.stats["cpu_calls"] += 1
                    future.set_result(None)
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

    def _copy_prefill_admissions(self, fills, done):
        """Publish prefill weights from the borrowed arena before it is reused.

        The tail is reserved outside the cache pool during prefill. The next
        borrower (or decode upload) drains admission before releasing that tail,
        so both source and destination remain valid until this event completes.
        """
        from sglang.kernels.ops.moe.qwen38_expert_cache import packed_scatter

        r = self.runtime
        table = torch.empty(
            (len(self.pool), len(fills), 3), dtype=torch.int64, pin_memory=True
        )
        experts, slots = zip(*fills)
        for row, (name, tensor) in zip(table, self.pool.items()):
            nbytes = tensor[0].numel() * tensor.element_size()
            row[:, 0].numpy()[:] = [tensor.data_ptr() + s * nbytes for s in slots]
            row[:, 1].numpy()[:] = [e * nbytes for e in experts]
            row[:, 2].fill_(nbytes)
        with torch.cuda.stream(self.fill_stream):
            self.fill_stream.wait_event(done)
            for row, name in zip(table, self.pool):
                packed_scatter(
                    r.arena[name].view(torch.uint8).reshape(-1),
                    row.to(r.device, non_blocking=True),
                )
            copied = torch.cuda.Event()
            copied.record(self.fill_stream)
        self.stats["prefill_admission_d2d_bytes"] += len(fills) * sum(
            t[0].numel() * t.element_size() for t in self.pool.values()
        )
        self.stats["insertions"] += len(fills)
        return copied

    def _complete_copies(self, copies):
        remaining = []
        for entry in copies:
            layer, epoch, fills, copied, future, *options = entry
            prepare = bool(options and options[0])
            try:
                if not copied.query():
                    remaining.append(entry)
                    continue
                self.policy.commit(layer, epoch, fills)
                future.set_result(self.policy.begin(layer) if prepare else None)
            except BaseException as error:
                self.failure = error
                future.set_exception(error)
            finally:
                if future.done():
                    self.jobs.task_done()
        return remaining

    def _submit_ready(self, readers, copies):
        remaining = []
        for entry in readers:
            layer, epoch, requested, done, future, *options = entry
            prepare = bool(options and options[0])
            deferred = False
            try:
                if not done.query():
                    remaining.append(entry)
                    deferred = True
                    continue
                # Releasing cold slots to the global free pool is safe only
                # after their old GPU readers finish. Query; never host-wait.
                safe_layers = None
                if hasattr(self, "device_planner"):
                    # Prefill still maintains cache on its background stream.
                    # Certify other layers here without blocking decode frames.
                    safe_layers = {
                        other
                        for other, state in enumerate(self.policy.layers)
                        if state.phase == "locked"
                        and (
                            other not in self.runtime.layer_computed
                            or self.runtime.layer_computed[other].query()
                        )
                    }
                evicted, fills = self.policy.plan(
                    layer, epoch, requested, safe_layers=safe_layers
                )
                if safe_layers is not None:
                    for other in safe_layers:
                        self.readable[other] = dict(self.policy.layers[other].mapping)
                self.stats["evictions"] += len(evicted)
                self.readable[layer] = dict(self.policy.layers[layer].mapping)
                if fills:
                    prefill_arena = len(options) > 1 and options[1]
                    copied = (
                        self._copy_prefill_admissions(fills, done)
                        if prefill_arena
                        else self._copy(layer, fills, done)
                    )
                    copies.append((layer, epoch, fills, copied, future, prepare))
                    deferred = True
                else:
                    self.policy.commit(layer, epoch, fills)
                    future.set_result(self.policy.begin(layer) if prepare else None)
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
                        job[4].set_exception(error)
                finally:
                    if not deferred:
                        self.jobs.task_done()

    def _gpu(self, x, slot_ids, weights, layer, tensors=None, hybrid=None):
        tensors = {
            **(self.pool if tensors is None else tensors),
            **self.runtime.layer_scales[layer],
        }
        if self.gpu_backend == "b12x":
            native = self.b12x.get(layer)
            if native is None:
                raise RuntimeError(
                    f"B12X was selected but layer {layer} is not initialized"
                )
            kwargs = {"output_dtype": torch.float32}
            if hybrid is not None:
                kwargs = {"output_dtype": torch.bfloat16, "hybrid": hybrid}
            return native.run(x, slot_ids, weights, tensors, **kwargs)
        # Explicit triton backend retains the FP32 partition contract.
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe

        return cached_w4a16_moe(
            x,
            slot_ids,
            weights,
            tensors,
            split_k=0,
            verify_mode="baseline",
            output_dtype=torch.float32,
        )

    def _lookup(self, layer, mapping):
        # Each table belongs to an immutable compute snapshot. In-flight fills
        # stay absent; survivor snapshots can also be used during maintenance.
        key = tuple(sorted(mapping.items()))
        entries = self.lookups.setdefault(layer, collections.OrderedDict())
        entry = entries.get(key)
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
            entries[key] = (lookup, device_lookup)
            if len(entries) > 2:
                entries.popitem(last=False)
        else:
            _, device_lookup = entry
        entries.move_to_end(key)
        return device_lookup

    def _remap(self, ids, mapping, layer):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes

        return cache_partition_routes(
            ids, self._lookup(layer, mapping), len(self.policy.keys)
        )[0]

    def _prepare_window(self, layer, epoch, requested, compute):
        """Fix LFU admission before selecting the optional CPU contribution.

        The manager owns the actual plan, reserves its destinations once, and
        publishes them after their copies. There is no speculative second plan
        or post-compute admission of weights already fetched for this window.
        """
        try:
            # Forwards submit serially on the compute stream. Its latest event
            # covers all preceding readers of this layer, including bypasses.
            # A None epoch denotes old post-compute maintenance, never a second
            # concurrent admission preparation for the same layer.
            if epoch is None:
                self._wait_layer(layer)
                epoch, _ = self.policy.begin(layer)
            if layer in self.active:
                raise RuntimeError("concurrent hybrid admission preparation")
            self.policy.release(layer, epoch)
            done = self.runtime.layer_computed.get(layer)
            if done is None:
                done = torch.cuda.Event()
                done.record(compute)
            self.readable[layer] = {}
            future = Future()
            self.active[layer] = future
            self.jobs.put_nowait((layer, epoch, requested, done, future, True))
            epoch, mapping = future.result(timeout=self.timeout)
            if self.active.get(layer) is not future:
                raise RuntimeError("hybrid admission future replaced")
            self.active.pop(layer)
        except BaseException as error:
            self.failure = error
            raise
        self.prepared_windows.add((layer, epoch))
        self.stats["admission_prepared_calls"] += 1
        return epoch, mapping

    def _gpu_misses(self, layer, x, ids, weights, mapping, misses, shared):
        """Compute missing experts in the reserved arena without mutating LFU."""
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes

        r = self.runtime
        compute = torch.cuda.current_stream(r.device)
        if r.arena_last_use is not None:
            r.transfer.wait_event(r.arena_last_use)
        r.transfer.wait_stream(compute)
        r._fill_arena_selected(layer, sorted(misses))
        if mapping:
            cached_ids, arena_ids = cache_partition_routes(
                ids, self._lookup(layer, mapping), len(self.policy.keys)
            )
        else:
            cached_ids, arena_ids = None, ids
        if shared is not None:
            shared()
        cached = self._gpu(x, cached_ids, weights, layer) if mapping else None
        compute.wait_stream(r.transfer)
        try:
            missing = self._gpu(x, arena_ids, weights, layer, r.arena)
        finally:
            # Protect arena readers even if the adapter enqueues work then raises.
            done = torch.cuda.Event()
            done.record(compute)
            r.arena_last_use = done
        self.stats["gpu_miss_calls"] += 1
        self.stats["gpu_miss_experts"] += len(misses)
        return missing if cached is None else cached + missing

    def run(self, layer, x, ids, weights, shared=None):
        if hasattr(self, "device_planner"):
            return self.device_planner.run(layer, x, ids, weights, shared)
        self._seed()
        if self.current is not None:
            current_layer, epoch, mapping = self.current
            if current_layer != layer:
                raise RuntimeError("hybrid layer-entry/expert order differs")
            self.current = None
        else:
            epoch, mapping = self._begin_layer(layer)
        compute = torch.cuda.current_stream(self.runtime.device)
        # Read only routes before choosing a branch. Full hits and GPU misses
        # never enqueue a CPU computation or read back activations/weights.
        host_ids = ids.cpu()
        rows = host_ids.tolist()
        if any(
            e < -1 or e >= self.runtime.manifest.num_experts
            for row in rows
            for e in row
        ):
            error = ValueError("hybrid route outside expert range")
            # layer_start may already hold this window. Reject further use
            # explicitly rather than leave an apparently healthy computing epoch.
            self.failure = error
            raise error
        requested = tuple(sorted({e for row in rows for e in row if e >= 0}))
        misses = set(requested) - mapping.keys()
        hit_routes = sum(e >= 0 and e in mapping for row in rows for e in row)
        self.stats["hits"] += len(requested) - len(misses)
        self.stats["misses"] += len(misses)
        if misses:
            epoch, mapping = self._prepare_window(layer, epoch, requested, compute)
        rejected = set(requested) - mapping.keys()
        cpu_experts = self.cost_model.select_cpu(
            x.shape[0], rows, mapping, rejected, admitted_experts=len(misses - rejected)
        )
        gpu_misses = rejected - cpu_experts
        self.stats["lfu_rejected_experts"] += len(rejected)
        self.stats["selected_cpu_experts"] += len(cpu_experts)
        self.stats["gpu_admitted_experts"] += len(misses - rejected)
        mode = "cpu" if cpu_experts else ("gpu" if misses else "hit")
        self.stats[f"selected_{mode}_calls"] += 1
        if mode == "hit":
            if shared is not None:
                shared()
            gpu = (
                self._gpu(x, self._remap(ids, mapping, layer), weights, layer)
                if hit_routes
                else torch.zeros_like(x, dtype=torch.float32)
            )
            return self._finish(layer, epoch, requested, ids, x, gpu)
        if mode == "gpu":
            if gpu_misses:
                gpu = self._gpu_misses(
                    layer, x, ids, weights, mapping, gpu_misses, shared
                )
            else:
                if shared is not None:
                    shared()
                gpu = self._gpu(x, self._remap(ids, mapping, layer), weights, layer)
            return self._finish(
                layer, epoch, requested, ids, x, gpu, arena=bool(gpu_misses)
            )
        # CPU jobs finish before return. The next readback fence also covers
        # the previous result upload before its pinned output is reused.
        shape = (layer, tuple(x.shape), tuple(ids.shape))
        buffers = self.host_buffers.get(shape)
        if buffers is None:
            buffers = (
                torch.empty_like(x, device="cpu", pin_memory=True),
                torch.empty_like(
                    weights, device="cpu", dtype=torch.float32, pin_memory=True
                ),
                torch.empty_like(x, device="cpu", dtype=torch.float32, pin_memory=True),
            )
            self.host_buffers[shape] = buffers
        host_x, host_weights, host_result = buffers
        host_ids = host_ids.clone()
        masked = host_ids.numpy()
        for t, row in enumerate(rows):
            for k, expert in enumerate(row):
                if expert not in cpu_experts:
                    masked[t, k] = -1
        with torch.cuda.stream(self.readback):
            self.readback.wait_stream(compute)
            for host, device in ((host_x, x), (host_weights, weights)):
                host.copy_(device, non_blocking=True)
                device.record_stream(self.readback)
            ready = torch.cuda.Event()
            ready.record(self.readback)
        future = Future()
        self.cpu_jobs.put_nowait(
            (layer, host_x, host_ids, host_weights, ready, host_result, future)
        )
        # Exclude only the selected CPU experts. Admitted and other rejected
        # experts retain GPU execution, even when CPU handles just one miss.
        gpu_ids = ids.clone()
        for expert in cpu_experts:
            gpu_ids.masked_fill_(ids == expert, -1)
        if gpu_misses:
            gpu = self._gpu_misses(
                layer, x, gpu_ids, weights, mapping, gpu_misses, shared
            )
        else:
            if shared is not None:
                shared()
            gpu = (
                self._gpu(x, self._remap(gpu_ids, mapping, layer), weights, layer)
                if set(requested) - cpu_experts
                else torch.zeros_like(x, dtype=torch.float32)
            )
        if not future.done():
            started = time.perf_counter()
            self.stats["result_join_waits"] += 1
            future.result(timeout=self.timeout)
            self.stats["result_join_seconds"] += time.perf_counter() - started
        else:
            future.result()
        self.stats["cpu_miss_experts"] += len(cpu_experts)
        with torch.cuda.stream(self.result_stream):
            result = host_result.to(self.runtime.device, non_blocking=True)
            delivered = torch.cuda.Event()
            delivered.record(self.result_stream)
        compute.wait_event(delivered)
        result.record_stream(compute)
        gpu = gpu + result
        return self._finish(
            layer, epoch, requested, ids, x, gpu, arena=bool(gpu_misses)
        )

    def _finish(self, layer, epoch, requested, ids, x, gpu, *, arena=False):
        compute = torch.cuda.current_stream(self.runtime.device)
        done = torch.cuda.Event()
        done.record(compute)
        self.runtime.layer_computed[layer] = done
        self.pending = (layer, epoch, requested, done)
        self.runtime._record_invocation(arena, ids, layer)
        self.runtime.finish_layer(layer)
        return gpu.to(x.dtype)

    def drain(self):
        if self.closed:
            return
        if self.failure is None:
            self.release_previous()
        self._join_queue(self.jobs)
        self._join_queue(self.cpu_jobs)
        planner = getattr(self, "device_planner", None)
        if planner is not None:
            planner.sync_host()
        self._release_prefill_slots()
        for layer in list(self.active):
            self._wait_layer(layer)
        self._check()

    def _copy_prefill_hits(self, hits, mapping):
        """Reuse resident rows in the existing compact prefill arena."""
        from sglang.kernels.ops.moe.qwen38_expert_cache import packed_scatter

        r = self.runtime
        pinned = r.device.type == "cuda"
        experts = sorted(hits)
        table = torch.empty(
            (len(self.pool), len(experts), 3),
            dtype=torch.int64,
            device="cpu",
            pin_memory=pinned,
        )
        for row, (name, tensor) in zip(table, self.pool.items()):
            nbytes = tensor[0].numel() * tensor.element_size()
            row[:, 0].numpy()[:] = [
                r.arena[name].data_ptr() + e * nbytes for e in experts
            ]
            row[:, 1].numpy()[:] = [mapping[e] * nbytes for e in experts]
            row[:, 2].fill_(nbytes)
        with torch.cuda.stream(r.transfer):
            for row, tensor in zip(table, self.pool.values()):
                packed_scatter(
                    tensor.view(torch.uint8).reshape(-1),
                    row.to(r.device, non_blocking=True),
                )
        r.prefill_reused_experts += len(experts)
        r.prefill_reuse_d2d_bytes += len(experts) * sum(
            t[0].numel() * t.element_size() for t in self.pool.values()
        )

    def prefill(self, layer, ids, execute, shared=None):
        """Prefill uses the same global mapping and a disjoint GPU arena.

        CPU A16 is reserved for tiny decode/verify. Large prefill fills its
        misses on GPU; cache retention and subsequent admission stay global.
        """
        from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts

        planner = getattr(self, "device_planner", None)
        if planner is not None:
            planner.sync_host()
            planner.valid = False
        self._seed()
        self._borrow_prefill_slots()
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
        if ids.is_cuda and ids.dtype == torch.int32:
            from sglang.kernels.ops.moe.qwen38_expert_cache import cache_route_counts

            # Prefill needs a set, not all M*top_k Python integers. Reuse the
            # existing layer histogram after draining legacy/device readers.
            counts = r.device_states[layer]["counts"]
            cache_route_counts(ids, counts)
            counts = counts.cpu().tolist()
            if counts[-1]:
                raise ValueError("prefill route outside expert range")
            requested = {e for e, count in enumerate(counts[:-1]) if count}
        else:
            host_ids = ids.cpu()
            requested = {int(e) for row in host_ids.tolist() for e in row if e >= 0}
            if bool(((host_ids < -1) | (host_ids >= r.manifest.num_experts)).any()):
                raise ValueError("prefill route outside expert range")
        # Prefill temporarily reserves the borrowed tail after retiring readers;
        # decode returns those slots to the global pool.
        limit = len(self.policy.keys)
        if r.borrowed_arena:
            limit -= r.manifest.num_experts
        if r.arena_last_use is not None:
            compute.wait_event(r.arena_last_use)
        misses = sorted(requested - mapping.keys())
        hits = requested & mapping.keys()
        compact = hits and not getattr(r, "prefill_direct_cache", True)
        if misses or compact:
            r.transfer.wait_stream(compute)
            if misses:
                r._fill_arena_selected(layer, misses)
            if compact:
                self._copy_prefill_hits(hits, mapping)
                self.stats["prefill_compact_runs"] += 1
            compute.wait_stream(r.transfer)
        if shared is not None:
            shared()
        if hits and not compact:
            prepared = PreparedExperts(
                {**r.arena, **r.layer_scales[layer]},
                ids,
                r.manifest.num_experts,
                True,
                cache_tensors={**self.pool, **r.layer_scales[layer]},
                cache_experts={e: mapping[e] for e in hits},
                arena_experts=misses,
                cache_lookup=self._lookup(layer, mapping),
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
        self.pending = (
            layer,
            epoch,
            tuple(sorted(requested)),
            done,
            bool(r.borrowed_arena),
        )
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
            if hasattr(self, "device_planner"):
                self.device_planner._validate_capacity(budget)
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
            if hasattr(self, "device_planner"):
                self.device_planner.valid = False
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
            for page in self.mapped_batch_pages:
                page.close(self.runtime.device)
            self.mapped_batch_pages.clear()
            self.cpu.close()
            self.closed = True
        if error is not None:
            raise error
