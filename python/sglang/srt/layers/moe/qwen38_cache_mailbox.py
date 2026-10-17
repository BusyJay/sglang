# SPDX-License-Identifier: Apache-2.0
"""GPU-planned direct fills for the threaded expert cache.

The planner and fill run on the same stream without CPU acknowledgement.
Both hits and misses accumulate device counters, collected at stats snapshots.
Registered immutable mmap bytes are read directly into GPU cache slots.
Prior GPU readers finish before a layer's plan or cache slots are reused.
"""

import collections
import os
import threading
import time

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import (
    _module,
    cache_alpha_scatter,
)
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts


class DecodeMissMailbox:
    def __init__(self, runtime):
        try:
            with torch.device("cpu"):
                self._initialize(runtime)
        except BaseException:
            sampler = getattr(self, "sampler", None)
            if sampler is not None:
                sampler.close()
            if hasattr(self, "stopping"):
                self.stopping.set()
            watchdog = getattr(self, "watchdog", None)
            if watchdog is not None and watchdog.is_alive():
                watchdog.join()
            if hasattr(self, "pack_address"):
                with torch.cuda.device(runtime.device):
                    _module().mailbox_unregister(self.pack_view)
            if hasattr(self, "pack_view"):
                del self.pack_view
            raise

    def _initialize(self, runtime):
        from cuda.bindings import driver

        from sglang.srt.utils.cuda_vmm_utils import check_drv

        self.runtime = runtime
        # Zero selects a device-side policy; no miss count crosses to the CPU.
        fill_cta = os.getenv("SGLANG_QWEN38_FILL_CTA", "8")
        if fill_cta not in tuple(str(i) for i in range(1, 9)) + ("adaptive",):
            raise ValueError("SGLANG_QWEN38_FILL_CTA must be 1..8 or adaptive")
        self.fill_cta = 0 if fill_cta == "adaptive" else int(fill_cta)
        layers, experts = runtime.manifest.num_layers, runtime.manifest.num_experts
        # hits, all-hit calls, misses, miss calls; one row per layer.
        self.hit_stats = torch.zeros(
            (layers, 4), dtype=torch.int64, device=runtime.device
        )
        self.observed = [[0, 0, 0, 0] for _ in range(layers)]
        self.native_active = False
        self.failed = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.failed_address = int(
            check_drv(
                driver.cuMemHostGetDevicePointer(self.failed.data_ptr(), 0),
                "qwen38 mailbox failure",
            )
        )
        self.sizes = {
            layer: tuple(
                runtime.manifest.tensors[layer][name].nbytes // experts
                for name in runtime.decode_copy_names
            )
            for layer in range(layers)
        }
        # Every transport shares the pack's one private source mapping. CUDA
        # materializes COW pages once; no second read-only mapping is retained.
        from sglang.srt.layers.moe.qwen38_observation import observe
        with observe("source_mapping", device=runtime.device, startup=True):
            self.pack_mapping = runtime.pack.cuda_source_mapping()
        self.pack_view = torch.frombuffer(self.pack_mapping, dtype=torch.uint8)
        with torch.cuda.device(runtime.device):
            with observe("host_register_and_device_pointer", device=runtime.device,
                         startup=True, source_bytes=self.pack_view.nbytes):
                self.pack_address = _module().mailbox_register(self.pack_view)
        with observe("discard_source_cache", device=runtime.device, startup=True):
            runtime.pack.discard_source_file_cache()
        self.refresh_targets()
        self.last_computed = None
        # Reuse completion events instead of allocating/destroying them in
        # serving frames. Keep external warm-up events until the GPU drains.
        self.events = [
            torch.cuda.Event() for _ in range(runtime.jobs.maxsize + layers + 1)
        ]
        for event in self.events:
            event.record(torch.cuda.current_stream(runtime.device))
        self.fill_stream = torch.cuda.Stream(device=runtime.device)
        self.overlap_events = {
            id(event): (torch.cuda.Event(), torch.cuda.Event()) for event in self.events
        }
        for pair in self.overlap_events.values():
            for event in pair:
                event.record(torch.cuda.current_stream(runtime.device))
        self.overlap_invocations = 0
        self.available = collections.deque(self.events)
        self.retained_external = {}
        # Preload the fill kernel before serving frames.
        warm_plan = torch.zeros(4, dtype=torch.int32, device=runtime.device)
        _module().mailbox_fill_grid(
            warm_plan,
            self.layouts[0],
            runtime.manifest.top_k,
            self.failed_address,
            self.fill_cta,
        )
        torch.cuda.current_stream(runtime.device).synchronize()
        self.pending = collections.deque()
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.watchdog = threading.Thread(
            target=self._watch, name="qwen38-mailbox-watchdog", daemon=True
        )
        self.watchdog.start()
        self.sampler = None
        if runtime.adaptive_cache and runtime.fused_decode_plan:
            from sglang.srt.layers.moe.qwen38_cache_sampling import DeviceRouteSampler

            self.sampler = DeviceRouteSampler(runtime, self._abort)

    def run(self, layer, ids, execute, timeout, sample=None, during_fill=None):
        r = self.runtime
        compute = torch.cuda.current_stream(r.device)
        previous = r.layer_computed.get(layer)
        if self.last_computed is not None:
            compute.wait_event(self.last_computed)
        if previous is not None:
            if previous not in self.events:
                self.retained_external[id(previous)] = previous
            compute.wait_event(previous)
        if layer in r.warm_ready:
            compute.wait_event(r.warm_ready[layer])
        with self.lock:
            self._retire_completed()
            oldest = self.pending[0][0] if len(self.pending) >= r.jobs.maxsize else None
        if oldest is not None:
            oldest.synchronize()
            with self.lock:
                self._retire_completed()
        with self.lock:
            live = set(r.layer_computed.values())
            for _ in range(len(self.available)):
                computed = self.available.popleft()
                if computed not in live:
                    break
                self.available.append(computed)
            else:
                raise RuntimeError("decode completion event pool exhausted")
        state = r.device_states[layer]
        remapped = torch.empty_like(ids)
        capture = (
            self.sampler.begin(layer, sample, compute)
            if self.sampler is not None
            else 0
        )
        _module().decode_plan_parallel(
            ids,
            remapped,
            state["counts"],
            state["lookup"],
            state["keys"],
            state["frequency"],
            state["ages"],
            state["generation"],
            state["clock"],
            state["plan"],
            0.95,
            int(r.cache_policy == "slru"),
            r.cache_protected_fraction,
            self.hit_stats.data_ptr() + self.hit_stats.stride(0) * 8 * layer,
            capture,
            os.getenv("SGLANG_QWEN38_PARALLEL_MISS_PLAN", "1") == "1",
        )
        self.native_active = True
        if capture:
            self.sampler.commit(layer, sample, compute)
        r.fused_decode_plan_invocations += 1
        r.decode_device_gate_invocations += 1
        if r.resident_alphas:
            tables, tensors = r.alpha_tables[layer], r.caches[layer]
            cache_alpha_scatter(
                tables["g1_alphas"],
                tables["g2_alphas"],
                tensors["g1_alphas"],
                tensors["g2_alphas"],
                state["plan"],
            )
        try:

            def submit_fill():
                _module().mailbox_fill_grid(
                    state["plan"],
                    self.layouts[layer],
                    r.manifest.top_k,
                    self.failed_address,
                    self.fill_cta,
                )

            if during_fill is None:
                submit_fill()
            else:
                from sglang.srt.layers.moe.qwen38_cache_overlap import fill_with_overlap

                ready, filled = self.overlap_events[id(computed)]
                fill_with_overlap(
                    compute, self.fill_stream, ready, filled, submit_fill, during_fill
                )
                self.overlap_invocations += 1
            return execute(
                PreparedExperts(
                    {**r.caches[layer], **r.layer_scales[layer]},
                    remapped,
                    r.slots_for_layer(layer),
                    False,
                )
            )
        finally:
            computed.record(compute)
            r.layer_computed[layer] = computed
            self.last_computed = computed
            with self.lock:
                self.pending.append(
                    (computed, time.monotonic() + timeout if timeout else None, layer)
                )

    def _abort(self, error):
        r = self.runtime
        r.worker_error = error
        r.poisoned = True
        # Queued fills check this mapped flag before reading expert sources.
        # The watchdog never submits CUDA work.
        self.failed[0] = 1

    def refresh_targets(self):
        r = self.runtime
        self.layouts = {
            layer: torch.tensor(
                [
                    [
                        r.caches[layer][name].data_ptr(),
                        size,
                        self.pack_address + r.manifest.tensors[layer][name].offset,
                    ]
                    for name, size in zip(r.decode_copy_names, self.sizes[layer])
                ],
                dtype=torch.int64,
                device=r.device,
            )
            for layer in self.sizes
        }

    @property
    def device_bytes(self):
        return (
            (self.sampler.device_bytes if self.sampler is not None else 0)
            + self.hit_stats.nbytes
            + sum(table.nbytes for table in self.layouts.values())
        )

    @property
    def pinned_bytes(self):
        return (self.sampler.pinned_bytes if self.sampler is not None else 0) + sum(
            tensor.nbytes
            for tensor in (
                self.failed,
                self.pack_view,
            )
        )

    def _watch(self):
        with torch.cuda.device(self.runtime.device):
            while not self.stopping.wait(0.01):
                try:
                    with self.lock:
                        self._retire_completed()
                        expired = (
                            self.pending
                            and self.pending[0][1] is not None
                            and time.monotonic() >= self.pending[0][1]
                        )
                        layer = self.pending[0][2] if expired else None
                    if expired:
                        self._abort(
                            TimeoutError(
                                f"expert cache GPU completion timed out at layer {layer}"
                            )
                        )
                        return
                except BaseException as error:
                    self.runtime.worker_error = error
                    self.runtime.poisoned = True
                    return

    def _retire_completed(self):
        # Keep every event alive in self.events; recycle without destroying it.
        while self.pending and self.pending[0][0].query():
            self.available.append(self.pending.popleft()[0])

    def synchronize(self):
        if not self.native_active:
            return
        for event in list(self.runtime.layer_computed.values()):
            event.synchronize()
        if self.sampler is not None:
            self.sampler.synchronize()
        self.flush_stats()
        with self.lock:
            self.pending.clear()
            self.available = collections.deque(self.events)
        self.retained_external.clear()
        self.native_active = False

    def flush_stats(self):
        compute = torch.cuda.current_stream(self.runtime.device)
        for event in list(self.runtime.layer_computed.values()):
            compute.wait_event(event)
        values = self.hit_stats.cpu().tolist()
        r = self.runtime
        for layer, (current, previous) in enumerate(zip(values, self.observed)):
            hits, calls, misses, miss_calls = (
                value - old for value, old in zip(current, previous)
            )
            size = sum(self.sizes[layer]) * misses
            r.hits += hits
            r.layer_hits[layer] += hits
            r.decode_device_gate_hits += calls
            r.misses += misses
            r.layer_misses[layer] += misses
            r.transferred_bytes += size
            r.decode_device_gate_copy_calls += miss_calls
            r.decode_device_gate_copy_bytes += size
            if r.resident_alphas:
                r.decode_alpha_h2d_calls_avoided += 2 * misses
        self.observed = values

    def close(self):
        if self.sampler is not None:
            self.sampler.close()
        self.stopping.set()
        self.watchdog.join()
        # runtime.close() drains compute before stopping mailbox threads.
        with torch.cuda.device(self.runtime.device):
            _module().mailbox_unregister(self.pack_view)
        del self.pack_view
