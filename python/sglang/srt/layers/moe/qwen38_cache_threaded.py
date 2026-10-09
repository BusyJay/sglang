# SPDX-License-Identifier: Apache-2.0
"""Main-thread asynchronous expert cache with a background mmap/H2D worker.

CPU transport remains, but plan/event waits run on the worker. Stream MemOps
hold compute until writes complete without consuming SMs. A bounded queue may
apply host backpressure when the producer outruns transport. Startup and explicit
synchronize/close wait intentionally. CUDA Graph capture is not supported.
"""

from __future__ import annotations

import os as _os
import sys
import queue
import threading
import time as _time

def _dbg(*a):
    if _os.environ.get("QWEN38_THREAD_DEBUG"):
        t = _time.strftime("%H:%M:%S")
        sys.stderr.write("[threaded %s] " % t + " ".join(str(x) for x in a))
        sys.stderr.flush()
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import (
    cache_plan,
    cache_ready_guard,
    cache_route,
)
from sglang.srt.layers.moe.qwen38_cache_device import Qwen38DeviceExpertRuntime
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
from sglang.srt.layers.moe.qwen38_stream_memops import StreamMemoryOps


@dataclass
class TransferJob:
    layer: int
    arena: bool
    gate: torch.Tensor
    plan: torch.Tensor | None
    planned: object
    computed: object
    launched: threading.Event


class Qwen38ThreadedExpertRuntime(Qwen38DeviceExpertRuntime):
    def __init__(self, pack, device, *, max_pending=64, **kwargs):
        super().__init__(pack, device, **kwargs)
        if not 1 <= max_pending <= 1024:
            raise ValueError("max_pending must be between 1 and 1024")
        self.memops = StreamMemoryOps()
        self.jobs = queue.Queue(maxsize=max_pending)
        self.capacity = threading.BoundedSemaphore(max_pending)
        self.layer_computed = {}
        self.worker_error = None
        self.closed = False
        self.worker = None
        self.retired = queue.Queue()
        self.reaper = None

    def _work_copy_sync(self, view, destination):
        src = torch.frombuffer(bytearray(view), dtype=torch.uint8)
        staging = self.staging[: src.numel()]
        staging.copy_(src)
        destination.reshape(-1).view(torch.uint8).copy_(staging, non_blocking=False)
        view.release()

    def _fill_sync(self, layer_id, expert_id, tensors, slot_id):
        from sglang.srt.layers.moe.qwen38_cache_runtime import REQUIRED_TENSORS

        for name in REQUIRED_TENSORS:
            self._work_copy_sync(
                self.pack.expert_tensor_bytes(layer_id, expert_id, name),
                tensors[name][slot_id],
            )

    def reserve(self):
        super().reserve()
        # Precompile guards/planner and verify driver wait/write support before
        # creating the worker or issuing any runtime job.
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        _module()
        gate = torch.zeros(1, dtype=torch.int32, device=self.device)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self.memops.signal(stream, gate)
            self.memops.wait_ready(stream, gate)
            stream.synchronize()
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

    def _retire(self):
        with torch.cuda.device(self.device):
            while True:
                job = self.retired.get()
                if job is None:
                    self.retired.task_done()
                    return
                try:
                    job.launched.wait()
                    job.computed.synchronize()
                except BaseException as error:
                    self.worker_error = error
                finally:
                    if job is not None:
                        self.capacity.release()
                        self.jobs.task_done()
                    self.retired.task_done()

    def _work(self):
        with torch.cuda.device(self.device), torch.no_grad():
            while True:
                job = self.jobs.get()
                if job is None:
                    self.jobs.task_done()
                    return
                try:
                    job.planned.synchronize()
                    if not _os.environ.get("QWEN38_NO_PLAN_WAIT"):
                        self.transfer.wait_event(
                            job.planned
                        )  # gate initialization/allocation
                    _dbg("worker.planned_ok", job.layer, job.arena)
                    if self.worker_error is not None:
                        raise RuntimeError("previous transfer failed")
                    if job.arena:
                        if job.plan.item():
                            raise ValueError("invalid prefill route ID")
                        if self.arena_layer != job.layer:
                            if self.arena_last_use is not None:
                                self.transfer.wait_event(self.arena_last_use)
                            self.arena_layer = None
                            if _os.environ.get("QWEN38_SYNC_WORKER_COPY"):
                                for expert in range(self.manifest.num_experts):
                                    self._fill_sync(job.layer, expert, self.arena, expert)
                            else:
                                for expert in range(self.manifest.num_experts):
                                    self._fill(job.layer, expert, self.arena, expert)
                            self.arena_layer = job.layer
                            self.arena_loads += 1
                        elif self.arena_last_use is not None:
                            self.transfer.wait_event(self.arena_last_use)
                        self.memops.signal(self.transfer, job.gate)
                        self.arena_last_use = job.computed
                    else:
                        values = job.plan.tolist()
                        status, misses, active, arena = values[:4]
                        _dbg("worker.decode_plan", job.layer, "misses", misses, "active", active)
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
                            if last is not None and not _os.environ.get("QWEN38_NO_LASTUSE_WAIT"):
                                _dbg("worker.wait_last", job.layer, slot)
                                self.transfer.wait_event(last)
                            _dbg("worker.fill_begin", job.layer, expert, slot)
                            if _os.environ.get("QWEN38_SYNC_WORKER_COPY"):
                                self._fill_sync(job.layer, expert, self.caches[job.layer], slot)
                            else:
                                self._fill(job.layer, expert, self.caches[job.layer], slot)
                            _dbg("worker.fill_done", job.layer, expert, slot)
                        # Metadata is ordered after the previous same-layer compute;
                        # read-only hits are complete and need no further CPU wait.
                        self.memops.signal(self.transfer, job.gate)
                        for slot in active_slots:
                            self.slot_last_use[job.layer][slot] = job.computed
                        self.misses += misses
                        self.hits += active - misses
                    _dbg("worker.signaled_and_requeue", job.layer, job.arena)
                    # Retire on a separate thread so staging the next job can
                    # overlap this job's computation.
                    self.retired.put(job)
                    job = None
                except BaseException as error:
                    self.worker_error = error
                    try:
                        self.memops.signal(self.transfer, job.gate, failed=True)
                        self.transfer.synchronize()
                    except BaseException:
                        # Driver failure can prevent releasing a queued wait. A
                        # failed CUDA context must be terminated by the caller.
                        pass
                    self.retired.put(job)
                    job = None
                finally:
                    if job is not None:
                        self.capacity.release()
                        self.jobs.task_done()

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
            if arena and self.arena is None:
                raise ValueError("route bound exceeds cache; enable prefill arena")
            self.capacity.acquire()  # bounded producer backpressure, not a per-layer CUDA wait
            call_id = getattr(self, "_dbg_seq", 0)
            self._dbg_seq = call_id + 1
            queued = False
            launched = threading.Event()
            compute = torch.cuda.current_stream(self.device)
            gate = torch.zeros(1, dtype=torch.int32, device=self.device)
            planned = torch.cuda.Event()
            computed = torch.cuda.Event()
            plan = None
            previous = self.layer_computed.get(layer_id)
            try:
                if previous is not None:
                    compute.wait_event(previous)
                state = self.device_states[layer_id]
                if arena:
                    remapped = topk_ids
                    self.metadata_stream.wait_stream(compute)
                    with torch.cuda.stream(self.metadata_stream):
                        scratch = torch.empty_like(topk_ids)
                        cache_route(topk_ids, state["lookup"], scratch, state["counts"])
                        plan = torch.empty(
                            1, dtype=torch.int32, device="cpu", pin_memory=True
                        )
                        plan.copy_(state["counts"][-1:], non_blocking=True)
                        planned.record(self.metadata_stream)
                    topk_ids.record_stream(self.metadata_stream)
                    compute.wait_event(planned)
                else:
                    self.metadata_stream.wait_stream(compute)
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
                self.jobs.put_nowait(
                    TransferJob(
                        layer_id, arena, gate, plan, planned, computed, launched
                    )
                )
                queued = True
                _dbg("run.enqueued", call_id, "layer", layer_id, "arena" if arena else "decode")
                if _os.environ.get("QWEN38_SKIP_GATE"):
                    _dbg("run.skip_gate", call_id)
                else:
                    self.memops.wait_ready(compute, gate)
                _dbg("run.gate_ok", call_id)
                if not _os.environ.get("QWEN38_SKIP_GATE"):
                    cache_ready_guard(gate)
                prepared = PreparedExperts(
                    {
                        **(self.arena if arena else self.caches[layer_id]),
                        **self.layer_scales[layer_id],
                    },
                    remapped,
                    self.manifest.num_experts if arena else self.slots_per_layer,
                    arena,
                )
                _dbg("run.execute_begin", call_id)
                result = execute(prepared)
                _dbg("run.execute_end", call_id)
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
                    _dbg("run.final_done", call_id)

    def synchronize(self):
        if self.worker is None:
            return super().synchronize()
        self.jobs.join()
        super().synchronize()
        if self.worker_error is not None:
            raise RuntimeError(
                "background expert transfer failed"
            ) from self.worker_error

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
        # Each outstanding decode job owns a tiny pinned transfer plan.
        result["max_pending_plan_bytes"] = (
            self.jobs.maxsize * (4 + 3 * self.slots_per_layer) * 4
        )
        return result
