# SPDX-License-Identifier: Apache-2.0
"""Main-thread asynchronous expert cache with a background mmap/H2D worker.

CPU transport remains, but plan/event waits run on the worker. Slot readiness
is expressed with CUDA events only: the worker records a per-job ready event on
the transfer stream after dispatching copies and notifies the caller through a
lightweight flag; the compute stream then waits on that event before MoE reads.
A bounded queue may apply host backpressure when the producer outruns transport.
Startup and explicit synchronize/close wait intentionally. CUDA Graph capture is
not supported. Raw driver stream memory operations are deliberately avoided so
every CUDA state transition goes through PyTorch's own locking.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import cache_plan, cache_route
from sglang.srt.layers.moe.qwen38_cache_device import Qwen38DeviceExpertRuntime
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts


def _ready_timeout_seconds() -> float | None:
    value = os.environ.get("QWEN38_READY_TIMEOUT_SECONDS")
    if value in (None, ""):
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError("QWEN38_READY_TIMEOUT_SECONDS must be a number") from None
    if parsed <= 0:
        raise ValueError("QWEN38_READY_TIMEOUT_SECONDS must be positive or zero")
    return parsed


@dataclass
class TransferJob:
    layer: int
    arena: bool
    plan: torch.Tensor | None
    planned: object
    computed: object
    launched: threading.Event
    ready: object = None  # CUDA event recorded on the transfer stream by worker
    error: BaseException | None = None  # set instead of ready when transport fails
    ready_set: threading.Event = field(
        default_factory=threading.Event
    )  # notification: exactly one of ready/error


class Qwen38ThreadedExpertRuntime(Qwen38DeviceExpertRuntime):
    def __init__(self, pack, device, *, max_pending=64, **kwargs):
        super().__init__(pack, device, **kwargs)
        if not 1 <= max_pending <= 1024:
            raise ValueError("max_pending must be between 1 and 1024")
        self.jobs = queue.Queue(maxsize=max_pending)
        self.capacity = threading.BoundedSemaphore(max_pending)
        self.layer_computed = {}
        self.worker_error = None
        self.closed = False
        self.worker = None
        self.retired = queue.Queue()
        self.reaper = None

    def reserve(self):
        super().reserve()
        # Precompile the JIT routing/planner before creating the worker or
        # issuing any runtime job so failures surface at startup.
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        _module()
        self.worker = threading.Thread(
            target=self._work, name="qwen38-expert-h2d", daemon=True
        )
        self.reaper = threading.Thread(
            target=self._retire, name="qwen38-expert-retire", daemon=True
        )
        self.reaper.start()
        self.worker.start()

    def _raise_error(self):
        if self.worker_error is not None:
            raise RuntimeError(
                "background expert transfer failed; recreate runtime"
            ) from self.worker_error
        if self.closed or self.worker is None:
            raise RuntimeError("threaded expert runtime must be reserved and open")

    def _fail_job(self, job, error):
        self.worker_error = error
        job.error = error
        try:
            # Drain queued work so pending copies do not race later reusers of
            # this device state; ignore failures of already-broken contexts.
            self.transfer.synchronize()
        except BaseException:
            pass
        job.ready_set.set()
        # Retire failed jobs exactly like finished ones so permit release and
        # queue bookkeeping stay uniform across the retirement thread.
        self.retired.put(job)

    def _finish_job(self, job, record_event=True):
        if record_event:
            job.ready = torch.cuda.Event()
            job.ready.record(self.transfer)
        job.ready_set.set()
        # Retire on a separate thread so staging the next job can overlap this
        # job's computation.
        self.retired.put(job)

    def _work(self):
        with torch.cuda.device(self.device), torch.no_grad():
            while True:
                job = self.jobs.get()
                if job is None:
                    self.jobs.task_done()
                    return
                try:
                    job.planned.synchronize()
                    self.transfer.wait_event(job.planned)  # policy/allocation order
                    if self.worker_error is not None:
                        raise RuntimeError("previous transfer failed")
                    if job.arena:
                        if job.plan.item():
                            raise ValueError("invalid prefill route ID")
                        if self.arena_layer != job.layer:
                            if self.arena_last_use is not None:
                                self.transfer.wait_event(self.arena_last_use)
                            self.arena_layer = None
                            for expert in range(self.manifest.num_experts):
                                self._fill(job.layer, expert, self.arena, expert)
                            self.arena_layer = job.layer
                            self.arena_loads += 1
                        elif self.arena_last_use is not None:
                            self.transfer.wait_event(self.arena_last_use)
                        # Metadata (route/plan kernels) ran on the metadata
                        # stream; nothing else needs to move for full reads.
                        self._finish_job(job)
                    else:
                        values = job.plan.tolist()
                        status, misses, active, arena = values[:4]
                        if status or arena:
                            raise ValueError(
                                f"unexpected device transfer plan: status={status}, arena={arena}"
                            )
                        active_slots = values[
                            4
                            + 2 * self.slots_per_layer : 4
                            + 2 * self.slots_per_layer
                            + active
                        ]
                        for index in range(misses):
                            expert, slot = values[4 + 2 * index : 6 + 2 * index]
                            last = self.slot_last_use[job.layer][slot]
                            if last is not None:
                                # The overwritten slot must have finished its
                                # previous reader before its bytes change.
                                self.transfer.wait_event(last)
                            self._fill(job.layer, expert, self.caches[job.layer], slot)
                        # Metadata is ordered after the previous same-layer
                        # compute; read-only hits are complete with no CPU wait.
                        for slot in active_slots:
                            self.slot_last_use[job.layer][slot] = job.computed
                        self.misses += misses
                        self.hits += active - misses
                        # Hits may need no copy at all; the ready event still
                        # has to exist and order any misses that were issued.
                        self._finish_job(job, record_event=bool(misses))
                    job = None
                except BaseException as error:
                    if job is None:
                        # Success was already retired above; keep worker state
                        # consistent for later run() calls.
                        self.worker_error = error
                    else:
                        self._fail_job(job, error)
                        job = None
                finally:
                    if job is not None:
                        self.capacity.release()
                        self.jobs.task_done()

    def _retire(self):
        with torch.cuda.device(self.device):
            while True:
                job = self.retired.get()
                if job is None:
                    self.retired.task_done()
                    return
                try:
                    # Retention until compute completion keeps slot/plan
                    # storage alive for every consumer of this invocation.
                    job.launched.wait()
                    job.computed.synchronize()
                except BaseException as error:
                    self.worker_error = error
                finally:
                    self.capacity.release()
                    self.jobs.task_done()
                    self.retired.task_done()

    def close(self):
        if self.closed:
            return
        try:
            self.synchronize()
        finally:
            self.closed = True
            if self.worker is not None:
                self.jobs.put(None)
                self.worker.join()
                self.retired.put(None)
                self.reaper.join()

    def memory_budget(self):
        result = super().memory_budget()
        # Each outstanding decode job owns one pinned transfer plan.
        result["max_pending_plan_bytes"] = (
            self.jobs.maxsize * (4 + 3 * self.slots_per_layer) * 4
        )
        return result

    def run(self, layer_id, topk_ids, execute):
        with self.lock, torch.no_grad(), torch.cuda.device(self.device):
            self._raise_error()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "threaded host transport cannot be captured in CUDA Graph"
                )
            if layer_id not in self.caches:
                raise ValueError("invalid/unreserved layer")
            if (
                topk_ids.device != self.device
                or topk_ids.dtype != torch.int32
                or topk_ids.ndim != 2
            ):
                raise ValueError("threaded cache requires CUDA int32 route IDs")
            if topk_ids.shape[1] != self.manifest.top_k:
                raise ValueError("top-k differs from the pack")
            arena = topk_ids.numel() > self.slots_per_layer
            self._record_invocation(arena, topk_ids)
            if arena and self.arena is None:
                raise ValueError("route bound exceeds cache; enable prefill arena")
            self.capacity.acquire()  # bounded producer backpressure, not a per-layer CUDA wait
            queued = False
            launched = threading.Event()
            compute = torch.cuda.current_stream(self.device)
            planned = torch.cuda.Event()
            computed = torch.cuda.Event()
            plan = None
            previous = self.layer_computed.get(layer_id)
            try:
                if previous is not None:
                    compute.wait_event(previous)
                state = self.device_states[layer_id]
                self.metadata_stream.wait_stream(compute)
                if arena:
                    remapped = topk_ids
                    with torch.cuda.stream(self.metadata_stream):
                        scratch = torch.empty_like(topk_ids)
                        cache_route(topk_ids, state["lookup"], scratch, state["counts"])
                        plan = torch.empty(
                            1, dtype=torch.int32, device="cpu", pin_memory=True
                        )
                        plan.copy_(state["counts"][-1:], non_blocking=True)
                        planned.record(self.metadata_stream)
                    topk_ids.record_stream(self.metadata_stream)
                else:
                    with torch.cuda.stream(self.metadata_stream):
                        remapped = torch.empty_like(topk_ids)
                        cache_route(
                            topk_ids, state["lookup"], remapped, state["counts"]
                        )
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
                        cache_route(
                            topk_ids, state["lookup"], remapped, state["counts"]
                        )
                        plan = torch.empty_like(
                            state["plan"], device="cpu", pin_memory=True
                        )
                        plan.copy_(state["plan"], non_blocking=True)
                        planned.record(self.metadata_stream)
                    topk_ids.record_stream(self.metadata_stream)
                    remapped.record_stream(compute)
                compute.wait_event(planned)
                job = TransferJob(layer_id, arena, plan, planned, computed, launched)
                self.jobs.put_nowait(job)
                queued = True
                timeout = _ready_timeout_seconds()
                # Host stall here is bounded by worker pickup latency, not by
                # copy completion; the copies themselves stay asynchronous.
                if timeout == 0:
                    job.ready_set.wait()
                elif not job.ready_set.wait(timeout):
                    self.poisoned = True
                    raise RuntimeError(
                        f"expert transfer worker did not report layer {layer_id} "
                        "within the readiness deadline"
                    )
                if job.error is not None:
                    self.poisoned = True
                    raise RuntimeError(
                        "background expert transfer failed; recreate runtime"
                    ) from job.error
                if job.ready is not None:
                    compute.wait_event(job.ready)
                prepared = PreparedExperts(
                    {
                        **(self.arena if arena else self.caches[layer_id]),
                        **self.layer_scales[layer_id],
                    },
                    remapped,
                    self.manifest.num_experts if arena else self.slots_per_layer,
                    arena,
                )
                result = execute(prepared)
                return result
            except BaseException:
                if not queued:
                    self.capacity.release()
                raise
            finally:
                if queued:
                    computed.record(compute)
                    self.layer_computed[layer_id] = computed
                    launched.set()
