# SPDX-License-Identifier: Apache-2.0
"""Miss-only GPU to host mailboxes for the threaded expert cache.

The decode planner releases hits entirely on device. Misses publish their
plan into mapped pinned memory. The fill reads registered immutable mmap
bytes directly; the CPU worker only publishes metadata and counters.
No background CUDA submission is needed, even during synchronous host reads.
Per-layer mailboxes and the shared payload are reused after prior GPU readers.
"""

import collections
import mmap
import sysconfig
import threading
import time

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe.qwen38_expert_cache import (
    _module,
    cache_alpha_scatter,
)
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts


@cache_once
def _poll_module():
    return load_jit(
        "qwen38_mailbox",
        cpp_files=["moe/qwen38_mailbox.cc"],
        cpp_wrappers=[("poll", "Qwen38MailboxPoll::run")],
        extra_include_paths=[sysconfig.get_path("include")],
    )


class DecodeMissMailbox:
    def __init__(self, runtime):
        try:
            with torch.device("cpu"):
                self._initialize(runtime)
        except BaseException:
            if hasattr(self, "stopping"):
                self.stopping.set()
            for name in ("worker", "watchdog"):
                worker = getattr(self, name, None)
                if worker is not None and worker.is_alive():
                    worker.join()
            if hasattr(self, "pack_address"):
                with torch.cuda.device(runtime.device):
                    _module().mailbox_unregister(self.pack_view)
            if hasattr(self, "pack_view"):
                del self.pack_view
            if hasattr(self, "pack_mapping"):
                self.pack_mapping.close()
            raise

    def _initialize(self, runtime):
        from cuda.bindings import driver

        from sglang.srt.utils.cuda_vmm_utils import check_drv

        self.runtime = runtime
        self.poll = _poll_module().poll
        layers, experts = runtime.manifest.num_layers, runtime.manifest.num_experts
        self.heads = torch.zeros(layers, dtype=torch.int32, pin_memory=True)
        self.plans = torch.empty(
            (layers, 4 + 3 * experts), dtype=torch.int32, pin_memory=True
        )
        self.seen = torch.zeros(layers, dtype=torch.int32)
        self.ready = torch.zeros(layers, dtype=torch.int64, pin_memory=True)
        self.ready_address = int(
            check_drv(
                driver.cuMemHostGetDevicePointer(self.ready.data_ptr(), 0),
                "qwen38 mailbox readiness",
            )
        )
        self.hit_stats = torch.zeros(
            (layers, 2), dtype=torch.int64, device=runtime.device
        )
        self.observed = [[0, 0] for _ in range(layers)]
        self.epochs = [0] * layers
        self.native_active = False
        self.head_address = int(
            check_drv(
                driver.cuMemHostGetDevicePointer(self.heads.data_ptr(), 0),
                "qwen38 mailbox heads",
            )
        )
        self.plan_address = int(
            check_drv(
                driver.cuMemHostGetDevicePointer(self.plans.data_ptr(), 0),
                "qwen38 mailbox plans",
            )
        )
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
        # GeForce lacks read-only host registration. A private writable mapping
        # keeps checkpoint files immutable; CUDA may materialize COW pages when
        # pinning, so account for a pack-sized host memory cost.
        self.pack_mapping = mmap.mmap(
            runtime.pack._stream.fileno(), 0, access=mmap.ACCESS_COPY
        )
        self.pack_view = torch.frombuffer(self.pack_mapping, dtype=torch.uint8)
        with torch.cuda.device(runtime.device):
            self.pack_address = _module().mailbox_register(self.pack_view)
        self.payload_address = self.pack_address
        self.refresh_targets()
        self.last_computed = None
        # Reuse completion events instead of allocating/destroying them in
        # serving frames. Keep external warm-up events until the GPU drains.
        self.events = [
            torch.cuda.Event() for _ in range(runtime.jobs.maxsize + layers + 1)
        ]
        for event in self.events:
            event.record(torch.cuda.current_stream(runtime.device))
        self.available = collections.deque(self.events)
        self.retained_external = {}
        # Preload every kernel/driver operation before parking a stream.
        warm_plan = torch.zeros(4, dtype=torch.int32, device=runtime.device)
        _module().mailbox_fill(
            warm_plan,
            self.layouts[0],
            self.payload_address,
            sum(self.sizes[0]),
            runtime.manifest.top_k,
            self.failed_address,
            self.ready_address,
            0,
            0,
        )
        torch.cuda.current_stream(runtime.device).synchronize()
        self.pending = collections.deque()
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.worker = threading.Thread(
            target=self._work, name="qwen38-miss-mailbox", daemon=True
        )
        self.watchdog = threading.Thread(
            target=self._watch, name="qwen38-mailbox-watchdog", daemon=True
        )
        self.worker.start()
        self.watchdog.start()

    def run(self, layer, ids, execute, timeout):
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
        self.epochs[layer] += 1
        epoch = self.epochs[layer]
        if epoch >= 0x7FFFFFFF:
            raise RuntimeError("decode mailbox sequence exhausted; recreate runtime")
        state = r.device_states[layer]
        remapped = torch.empty_like(ids)
        _module().decode_plan_mailbox(
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
            self.head_address + 4 * layer,
            self.plan_address + self.plans.stride(0) * 4 * layer,
            self.hit_stats.data_ptr() + 16 * layer,
            epoch,
        )
        self.native_active = True
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
        _module().mailbox_fill(
            state["plan"],
            self.layouts[layer],
            self.payload_address,
            sum(self.sizes[layer]),
            r.manifest.top_k,
            self.failed_address,
            self.ready_address + 8 * layer,
            epoch,
            int((timeout + 0.1) * 1e9) if timeout else 0,
        )
        try:
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
        # The already-enqueued device fill checks failure before any reads.
        # No CUDA submission is needed to release a parked stream on failure.
        self.failed[0] = 1
        self.ready.fill_(0x7FFFFFFFFFFFFFFF)

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
        return self.hit_stats.nbytes + sum(
            table.nbytes for table in self.layouts.values()
        )

    @property
    def pinned_bytes(self):
        return sum(
            tensor.nbytes
            for tensor in (
                self.heads,
                self.plans,
                self.ready,
                self.failed,
                self.pack_view,
            )
        )

    def _record_misses(self, layer, values, misses):
        r = self.runtime
        size = sum(self.sizes[layer]) * misses
        r.transferred_bytes += size
        r.decode_device_gate_copy_calls += 1
        r.decode_device_gate_copy_bytes += size
        if r.resident_alphas:
            r.decode_alpha_h2d_calls_avoided += 2 * misses
        r.misses += misses
        r.hits += values[2] - misses
        r.layer_misses[layer] += misses
        r.layer_hits[layer] += values[2] - misses

    def stage(self, layer, values, misses, epoch):
        # Immutable source addresses are precomputed; publish metadata only.
        self._record_misses(layer, values, misses)
        self.ready[layer] = epoch * (self.runtime.manifest.top_k + 1) + misses

    def _work(self):
        r = self.runtime
        # CPU-only: publishes metadata and counters; never submits CUDA work.
        try:
            while not self.stopping.is_set():
                layer = self.poll(self.heads, self.seen)
                if layer < 0:
                    continue
                epoch = int(self.heads[layer])
                capacity = r.slots_for_layer(layer)
                values = self.plans[layer, : 4 + 3 * capacity].tolist()
                status, misses, active, arena = values[:4]
                if status or arena or not 0 < misses <= active <= capacity:
                    raise ValueError("invalid miss mailbox plan")
                self.stage(layer, values, misses, epoch)
                self.seen[layer] = epoch
        except BaseException as error:
            self._abort(error)

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
                                f"expert transfer worker timed out at layer {layer}"
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
        for layer, ((hits, calls), (old_hits, old_calls)) in enumerate(
            zip(values, self.observed)
        ):
            r.hits += hits - old_hits
            r.layer_hits[layer] += hits - old_hits
            r.decode_device_gate_hits += calls - old_calls
        self.observed = values

    def close(self):
        self.stopping.set()
        self.worker.join()
        self.watchdog.join()
        # runtime.close() drains compute before stopping mailbox threads.
        with torch.cuda.device(self.runtime.device):
            _module().mailbox_unregister(self.pack_view)
        del self.pack_view
        self.pack_mapping.close()
