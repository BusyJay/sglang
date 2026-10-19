# SPDX-License-Identifier: Apache-2.0
"""Main-thread asynchronous expert cache with a background mmap/H2D worker.

Single-token decode can bypass the worker. The device planner and direct
mapped-source fill run on the compute stream without a CPU acknowledgement;
hit/miss statistics accumulate on device until a snapshot. Sampling and prefill use
the established event-based transport. Full CUDA Graph capture is unsupported.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import (
    cache_alpha_scatter,
    cache_decode_plan,
    cache_plan,
    cache_route,
    cache_scatter,
)
from sglang.srt.layers.moe.qwen38_borrowed_arena import BorrowedArenaMixin
from sglang.srt.layers.moe.qwen38_cache_device import Qwen38DeviceExpertRuntime
from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
from sglang.srt.layers.moe.qwen38_expert_cache import LayerExpertCachePolicy
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

logger = logging.getLogger(__name__)
_DENSE_STARTUP_PREFILL_TOKENS = 128


def _prefer_dense_prefill(missing, num_experts):
    """Choose bulk H2D when sparse bytes plus fragmentation lose their edge.

    On a 5090 with the 64 MiB staging ring, each extra run costs roughly
    2.3 expert rows until the overhead flattens near 48 runs. This boundary
    keeps one large missing interval sparse even when most experts are absent.
    """
    if not missing:
        return False
    runs = 1 + sum(right != left + 1 for left, right in zip(missing, missing[1:]))
    fragment_penalty = 7 * num_experts * min(runs, 48) // (3 * 512)
    return len(missing) + fragment_penalty >= num_experts


def _ready_timeout_seconds() -> float | None:
    value = os.environ.get("QWEN38_READY_TIMEOUT_SECONDS")
    if value in (None, ""):
        return 120.0
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError("QWEN38_READY_TIMEOUT_SECONDS must be a number") from None
    if parsed < 0:
        raise ValueError("QWEN38_READY_TIMEOUT_SECONDS must be positive or zero")
    return parsed


def _sample_rows(item):
    """Expand a GPU batch without changing the analyzer's ordered route input."""
    if len(item) == 4:
        yield item
    else:
        layer, cycle, first, end, rows = item
        if end - first != len(rows):
            raise ValueError("incomplete route sample batch")
        for index, active in enumerate(rows, first):
            yield layer, cycle, index, active


@dataclass
class TransferJob:
    layer: int
    arena: bool
    plan: torch.Tensor | None
    planned: object
    computed: object
    launched: threading.Event
    sample: tuple[int, int] | None = None  # observation cycle, offset within window
    force_dense: bool = False
    verify: bool = False
    ready: object = None  # CUDA event recorded on the transfer stream by worker
    error: BaseException | None = None  # set instead of ready when transport fails
    ready_set: threading.Event = field(
        default_factory=threading.Event
    )  # notification: exactly one of ready/error
    plan_values: list[int] | None = None  # already synchronized on the caller
    sample_published: object = None  # optional observer handoff at transport changes
    cache_experts: dict[int, int] | None = None
    arena_experts: list[int] | None = None
    plan_ready: threading.Event = field(default_factory=threading.Event)
    prefill_w4a4: bool = False


class Qwen38ThreadedExpertRuntime(BorrowedArenaMixin, Qwen38DeviceExpertRuntime):
    def __init__(
        self,
        pack,
        device,
        *,
        max_pending=64,
        adaptive_cache=False,
        adaptive_cache_window=512,
        adaptive_cache_interval=4096,
        adaptive_cache_mode="rebalance",
        cache_policy="lfu",
        cache_protected_fraction=0.7,
        prefill_reuse_cache=True,
        prefill_direct_cache=False,
        decode_early_cache=False,
        cache_overlap_policy="auto",
        preserve_cache_on_resize=False,
        prefill_seed_cache=True,
        resident_alphas=True,
        prefill_copy_mode="auto",
        fused_decode_plan=True,
        decode_hit_fast_path=True,
        decode_device_gate=True,
        packed_decode="direct",
        packed_prefill="direct",
        packed_min_bytes=4096,
        packed_max_bytes=64 << 20,
        packed_min_copies=2,
        **kwargs,
    ):
        from sglang.srt.layers.moe.qwen38_packed_transport import validate_packed_config

        validate_packed_config(
            packed_decode,
            packed_prefill,
            packed_min_bytes,
            packed_max_bytes,
            packed_min_copies,
        )
        self.packed_decode, self.packed_prefill = packed_decode, packed_prefill
        self.packed_min_bytes, self.packed_max_bytes = (
            packed_min_bytes,
            packed_max_bytes,
        )
        self.packed_min_copies = packed_min_copies
        self.packed_transport = None
        self.profiler = None
        super().__init__(pack, device, **kwargs)
        # Extra adaptive/borrowed slots are a cache budget, not a reason to
        # switch short prefills away from the batched arena transport.
        self.prefill_layer_slots = tuple(self.layer_cache_slots)
        if not isinstance(adaptive_cache, bool):
            raise ValueError("adaptive_cache must be a boolean")
        self.adaptive_cache = adaptive_cache
        if type(adaptive_cache_window) is not int or adaptive_cache_window < 16:
            raise ValueError("adaptive_cache_window must be an integer >= 16")
        self.adaptive_cache_window = adaptive_cache_window
        if type(adaptive_cache_interval) is not int or adaptive_cache_interval < 0:
            raise ValueError("adaptive_cache_interval must be a non-negative integer")
        if adaptive_cache_mode not in ("rebalance", "observe", "sample"):
            raise ValueError("adaptive_cache_mode must be rebalance, observe or sample")
        if cache_policy not in ("lfu", "slru") or not 0 <= cache_protected_fraction < 1:
            raise ValueError("invalid cache policy or protected fraction")
        self.adaptive_cache_interval = adaptive_cache_interval
        self.adaptive_cache_mode = adaptive_cache_mode
        self.cache_policy = cache_policy
        self.cache_protected_fraction = cache_protected_fraction
        if (
            type(prefill_reuse_cache) is not bool
            or type(prefill_seed_cache) is not bool
        ):
            raise ValueError("prefill reuse and seed options must be booleans")
        self.prefill_reuse_cache = prefill_reuse_cache
        if type(prefill_direct_cache) is not bool:
            raise ValueError("prefill_direct_cache must be a boolean")
        if prefill_direct_cache and not prefill_reuse_cache:
            raise ValueError("prefill_direct_cache requires prefill_reuse_cache")
        self.prefill_direct_cache = prefill_direct_cache
        for name, value in (
            ("decode_early_cache", decode_early_cache),
            ("preserve_cache_on_resize", preserve_cache_on_resize),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            setattr(self, name, value)
        if cache_overlap_policy not in ("auto", "always"):
            raise ValueError("cache_overlap_policy must be auto or always")
        self.cache_overlap_policy = cache_overlap_policy
        self.prefill_seed_cache = prefill_seed_cache
        if type(resident_alphas) is not bool:
            raise ValueError("resident_alphas must be a boolean")
        if prefill_copy_mode not in ("auto", "dense", "sparse"):
            raise ValueError("prefill_copy_mode must be auto, dense or sparse")
        self.resident_alphas = resident_alphas
        self.prefill_copy_mode = prefill_copy_mode
        # Resident per-expert alpha tables let one fused scatter fill both slot
        # alphas, so those two rows leave the per-expert decode copy set.
        self.decode_copy_names = tuple(
            name
            for name in REQUIRED_TENSORS
            if not resident_alphas or name not in ("g1_alphas", "g2_alphas")
        )
        if type(fused_decode_plan) is not bool:
            raise ValueError("fused_decode_plan must be a boolean")
        self.fused_decode_plan = fused_decode_plan
        self.fused_decode_plan_invocations = 0
        if type(decode_hit_fast_path) is not bool:
            raise ValueError("decode_hit_fast_path must be a boolean")
        self.decode_hit_fast_path = decode_hit_fast_path
        self.decode_hit_fast_path_invocations = 0
        if type(decode_device_gate) is not bool:
            raise ValueError("decode_device_gate must be a boolean")
        self.decode_device_gate = decode_device_gate
        self.decode_device_gate_invocations = 0
        self.decode_device_gate_hits = 0
        self.decode_device_gate_copy_calls = 0
        self.decode_device_gate_copy_bytes = 0
        self.gate_warmed_layers = set()
        self.miss_mailbox = None
        self.alpha_tables = {}
        self.decode_alpha_h2d_calls_avoided = 0
        self.prefill_copy_stats = [
            dict(
                dense=0,
                sparse=0,
                resident=0,
                startup_dense=0,
                forced_dense=0,
                heuristic_dense=0,
                missing=0,
                host_missing=0,
                missing_runs=0,
                reuse_candidates=0,
                direct_cache_runs=0,
                direct_cache_experts=0,
            )
            for _ in range(self.manifest.num_layers)
        ]
        self.prefill_reused_experts = self.prefill_reuse_d2d_bytes = 0
        self.prefill_seed_runs = self.prefill_h2d_bytes = 0
        self.prefill_host_dispatch_seconds = 0.0
        self.prefill_tokens_seen = 0
        self.prefill_dense_current = False
        self.adaptive_steps = [0] * self.manifest.num_layers
        self.adaptive_process = None
        self.analysis_stopping = threading.Event()
        self.adaptive_evaluations = 0
        self.adaptive_analysis_seconds = 0.0
        self.adaptive_last_evaluation = None
        self.host_plans = {}
        self.host_plan_allocations = 0
        self.cache_pool = None
        self.borrowed_arena = False
        self.borrowed_records = None
        self.borrowed_layers = {}
        self.borrowed_restore_runs = self.borrowed_restore_experts = 0
        self.adaptive_samples = None
        self.adaptive_worker = None
        self.adaptive_target = None
        self.adaptive_pending = None
        self.adaptive_lock = threading.Lock()
        self.adaptive_rebalances = 0
        self.adaptive_sample_drops = 0
        self.sample_publication = None
        self.adaptive_migrated_experts = 0
        self.adaptive_evicted_experts = 0
        self.adaptive_last_rebalance_seconds = 0.0
        self.layer_hits = [0] * self.manifest.num_layers
        self.layer_misses = [0] * self.manifest.num_layers
        # None means the shared prefill arena contains every expert of its
        # current layer; otherwise this set tracks only loaded expert IDs.
        self.arena_residents = set()
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
        from sglang.srt.layers.moe.qwen38_startup_memory import (
            get_startup_memory_probe,
            memory_mark,
        )

        probe = get_startup_memory_probe(self.device)
        memory_mark(
            probe,
            "runtime.reserve.begin",
            metadata={"adaptive": self.adaptive_cache, "slots": self.layer_cache_slots},
        )
        if self.adaptive_cache:
            budget = sum(self.layer_cache_slots)
            first = self.manifest.tensors[0]
            if getattr(self, "kv_cache_stages", None):
                from sglang.srt.layers.moe.qwen38_staged_pool import StagedExpertPool

                self.staged_pool = StagedExpertPool(
                    self.manifest,
                    self.device,
                    self.layer_cache_slots,
                    getattr(self, "adaptive_min_slots", self.manifest.top_k),
                )
                self.cache_pool = self.staged_pool.pool_views(budget)
            else:
                self.cache_pool = {}
            for name in REQUIRED_TENSORS:
                memory_mark(probe, "adaptive.pool." + name + ".before")
                if hasattr(self, "staged_pool"):
                    continue
                self.cache_pool[name] = torch.zeros(
                    (budget, *first[name].shape[1:]),
                    dtype=getattr(torch, first[name].dtype),
                    device=self.device,
                )
                memory_mark(
                    probe,
                    "adaptive.pool." + name + ".after",
                    metadata={
                        "tensor_bytes": self.cache_pool[name].numel()
                        * self.cache_pool[name].element_size()
                    },
                )
            self.transfer.wait_stream(torch.cuda.current_stream(self.device))
            self._assign_pool_views(self.layer_cache_slots)
            from sglang.srt.layers.moe.qwen38_cache_buffers import (
                AdaptiveMetadataBuffers,
            )

            # Every legal future layout fits without growing policy buffers.
            minimum = getattr(self, "adaptive_min_slots", self.manifest.top_k)
            maximum = min(
                self.manifest.num_experts,
                budget - (self.manifest.num_layers - 1) * minimum,
            )
            if hasattr(self, "staged_pool"):
                maximum = self.manifest.num_experts
            self.adaptive_metadata = AdaptiveMetadataBuffers(
                self.manifest.num_layers,
                self.manifest.num_experts,
                maximum,
                self.device,
            )
            self.migration_scratch = {
                name: torch.empty_like(pool[0])
                for name, pool in self.cache_pool.items()
            }
            self.adaptive_samples = queue.Queue(maxsize=4096)
        if (
            self.prefill_arena
            and self.cache_pool
            and os.getenv("SGLANG_QWEN38_BORROW_PREFILL_ARENA", "1") == "1"
            and budget >= self.manifest.num_experts
        ):
            self.arena = {
                name: tensor[-self.manifest.num_experts :]
                for name, tensor in self.cache_pool.items()
            }
            self.borrowed_arena = True
            logger.info(
                "Qwen38 prefill arena borrows %d expert slots from pool tail",
                self.manifest.num_experts,
            )
        super().reserve()
        memory_mark(probe, "runtime.after_base_reserve")
        # Precompile the JIT routing/planner before creating the worker or
        # issuing any runtime job so failures surface at startup.
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        _module()
        memory_mark(probe, "runtime.after_routing_jit")
        if self.resident_alphas:
            # Reserve before KV allocation. Copies finish before worker startup;
            # immutable expert-ID tables survive cache eviction and migration.
            for layer in self.manifest.tensors:
                tables = {}
                for name in ("g1_alphas", "g2_alphas"):
                    tables[name] = torch.empty(
                        self.manifest.num_experts,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.transfer.wait_stream(torch.cuda.current_stream(self.device))
                    self._copy(self.pack.tensor_bytes(layer, name), tables[name])
                self.alpha_tables[layer] = tables
            self.transfer.synchronize()
        memory_mark(probe, "runtime.after_resident_alphas")
        if self.adaptive_cache:
            # Extra plan tail exports compact original expert IDs. No second
            # histogram D2H, memset or routing kernel is needed for sampling.
            self.metadata_stream.wait_stream(torch.cuda.current_stream(self.device))
            if self.adaptive_cache_mode != "sample":
                self.adaptive_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(
                            Path(__file__).with_name("qwen38_cache_analysis_worker.py")
                        ),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
        memory_mark(probe, "runtime.after_adaptive_metadata")
        if self.packed_decode != "direct" or self.packed_prefill != "direct":
            from sglang.srt.layers.moe.qwen38_packed_transport import (
                PackedExpertTransport,
            )

            self.packed_transport = PackedExpertTransport(self)
            self.packed_transport.refresh_targets()
        self.worker = threading.Thread(
            target=self._work, name="qwen38-expert-h2d", daemon=True
        )
        self.reaper = threading.Thread(
            target=self._retire, name="qwen38-expert-retire", daemon=True
        )
        self.reaper.start()
        self.worker.start()
        if self.decode_device_gate:
            from sglang.srt.layers.moe.qwen38_cache_mailbox import DecodeMissMailbox

            self.miss_mailbox = DecodeMissMailbox(self)
        if self.adaptive_cache:
            self.adaptive_worker = threading.Thread(
                target=self._analyze_routes, name="qwen38-cache-adaptive", daemon=True
            )
            self.adaptive_worker.start()

    def _state(self, layer_id):
        if not hasattr(self, "adaptive_metadata"):
            return super()._state(layer_id)
        if layer_id not in self.device_states:
            slots = self.slots_for_layer(layer_id)
            self.device_states[layer_id] = self.adaptive_metadata.reset(layer_id, slots)
            self.slot_ready[layer_id] = [None] * slots
            self.slot_last_use[layer_id] = [None] * slots
        return self.device_states[layer_id]

    def _host_range(self, kind):
        from sglang.srt.layers.moe.qwen38_observation import observe

        return (
            self.profiler.host_range(kind)
            if self.profiler is not None
            else observe(kind)
        )

    def _gpu_range(self, kind, stream):
        return (
            self.profiler.gpu_range(kind, stream)
            if self.profiler is not None
            else contextlib.nullcontext()
        )

    def _copy_batch(self, entries):
        with self._host_range("cpu_staging_and_enqueue"):
            return super()._copy_batch(entries)

    def _fill(self, layer_id, expert_id, tensors, slot_id, names=REQUIRED_TENSORS):
        transport = self.packed_transport
        if transport is not None:
            kind = "prefill" if tensors is self.arena else "decode"
            expected = self.arena if kind == "prefill" else self.caches[layer_id]
            mode = self.packed_prefill if kind == "prefill" else self.packed_decode
            if (
                mode == "packed"
                and tensors is expected
                and tuple(names) == transport.layouts[kind][0]
            ):
                if transport.eligible(kind):
                    with self._host_range("packed_cpu_staging_and_enqueue"):
                        return transport.fill(layer_id, expert_id, slot_id, kind)
                transport.counters[kind + "_fallbacks"] += 1
        return super()._fill(layer_id, expert_id, tensors, slot_id, names=names)

    def _assign_pool_views(self, layout):
        offset = 0
        for layer, slots in enumerate(layout):
            self.caches[layer] = {
                name: tensor.narrow(0, offset, slots)
                for name, tensor in self.cache_pool.items()
            }
            offset += slots
        if offset > next(iter(self.cache_pool.values())).shape[0]:
            raise ValueError("adaptive layout exceeds expert-cache GPU budget")

    def _prefill_plan(self, state):
        if "prefill_plan" not in state:
            state["prefill_plan"] = torch.empty(
                state["counts"].numel() + state["keys"].numel(),
                dtype=torch.int32,
                device=self.device,
            )
        return state["prefill_plan"]

    def _host_plan(self, source):
        size = source.numel()
        pool = self.host_plans.get(size)
        if pool is None:
            pool = self.host_plans[size] = queue.SimpleQueue()
        try:
            return pool.get_nowait()
        except queue.Empty:
            self.host_plan_allocations += 1
            return torch.empty_like(source, device="cpu", pin_memory=True)

    def _sample_position(self, layer, topk_ids):
        if not self.adaptive_cache:
            return None
        if topk_ids.shape[0] != 1 and not getattr(self, "sampling_verify", False):
            # Do not join decode observations across unobserved prefill/batches.
            period = self.adaptive_cache_window + self.adaptive_cache_interval
            self.adaptive_steps[layer] = (
                self.adaptive_steps[layer] // period + 1
            ) * period
            return None
        step = self.adaptive_steps[layer]
        self.adaptive_steps[layer] += 1
        cycle, index = divmod(
            step, self.adaptive_cache_window + self.adaptive_cache_interval
        )
        return (cycle, index) if index < self.adaptive_cache_window else None

    def _migrate_pool(self, target, snapshot):
        from sglang.srt.layers.moe.qwen38_cache_adaptive import relocate_slots

        old_offsets = []
        new_offsets = []
        old_cursor = new_cursor = 0
        for old_slots, new_slots in zip(self.layer_cache_slots, target):
            old_offsets.append(old_cursor)
            new_offsets.append(new_cursor)
            old_cursor += old_slots
            new_cursor += new_slots
        moves = {}
        retained = {}
        occupied = evicted = 0
        for layer, new_slots in enumerate(target):
            keys, frequency, ages = snapshot[layer]
            residents = [slot for slot, expert in enumerate(keys) if expert >= 0]
            occupied += len(residents)
            residents.sort(key=lambda slot: (-frequency[slot], -ages[slot]))
            selected = residents[:new_slots]
            evicted += len(residents) - len(selected)
            retained[layer] = [
                (keys[slot], frequency[slot], ages[slot]) for slot in selected
            ]
            for new_slot, old_slot in enumerate(selected):
                moves[old_offsets[layer] + old_slot] = new_offsets[layer] + new_slot
        with torch.cuda.stream(torch.cuda.current_stream(self.device)):
            for name, pool in self.cache_pool.items():
                scratch = self.migration_scratch[name]
                relocate_slots(
                    moves,
                    lambda src, dst: pool[dst].copy_(pool[src]),
                    lambda src: scratch.copy_(pool[src]),
                    lambda dst: pool[dst].copy_(scratch),
                )
        self.transfer.wait_stream(torch.cuda.current_stream(self.device))
        return retained, occupied - evicted, evicted

    def _restore_device_policy(self, retained):
        for layer, records in retained.items():
            state = self.device_states[layer]
            self.adaptive_metadata.restore(layer, state, records, non_blocking=True)

    def _registered_prefill_enabled(self):
        return (
            os.getenv("SGLANG_QWEN38_PREFILL_REGISTERED_DMA", "1") == "1"
            and self.miss_mailbox is not None
        )

    def _registered_prefill_copy(self, layer, runs):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        with self._host_range("registered_prefill_enqueue"):
            with torch.cuda.stream(self.transfer):
                # pack_view/arena belong to the runtime and remain alive until drain.
                source = self.miss_mailbox.pack_view
                for name in REQUIRED_TENSORS:
                    record = self.manifest.tensors[layer][name]
                    row_bytes = record.nbytes // self.manifest.num_experts
                    for first, end in runs:
                        begin = record.offset + first * row_bytes
                        count = (end - first) * row_bytes
                        target = (
                            self.arena[name][first:end].reshape(-1).view(torch.uint8)
                        )
                        _module().registered_copy(target, source[begin : begin + count])
                        self.transferred_bytes += count
                        self.h2d_copy_calls += 1
                self.ready = torch.cuda.Event()
                self.ready.record(self.transfer)

    def _fill_arena(self, layer_id):
        if self._registered_prefill_enabled():
            return self._registered_prefill_copy(
                layer_id, [(0, self.manifest.num_experts)]
            )
        return super()._fill_arena(layer_id)

    def _fill_arena_selected(self, layer, experts):
        """Copy contiguous runs of routed experts into their original IDs."""
        if not experts:
            return
        runs = []
        start = previous = experts[0]
        for expert in experts[1:]:
            if expert != previous + 1:
                runs.append((start, previous + 1))
                start = expert
            previous = expert
        runs.append((start, previous + 1))
        if self._registered_prefill_enabled():
            return self._registered_prefill_copy(layer, runs)
        if (
            self.packed_transport is not None
            and self.packed_prefill == "packed"
            and self.packed_transport.eligible("prefill")
            and len(runs) > 1
        ):
            # One routed expert per fill. Dense bulk prefill remains direct.
            for expert in experts:
                self._fill(layer, expert, self.arena, expert)
            return
        entries = []
        try:
            for name in REQUIRED_TENSORS:
                record = self.manifest.tensors[layer][name]
                expert_bytes = record.nbytes // self.manifest.num_experts
                source = self.pack.tensor_bytes(layer, name)
                try:
                    for first, end in runs:
                        entries.append(
                            (
                                source[first * expert_bytes : end * expert_bytes],
                                self.arena[name][first:end],
                            )
                        )
                finally:
                    source.release()
            self._copy_batch(entries)
        except BaseException:
            for source, _ in entries:
                source.release()
            raise

    def _ensure_arena_experts(
        self, layer, counts, keys=None, *, force_dense=False, verify=False, job=None
    ):
        stats = self.prefill_copy_stats[layer]
        if self.arena_layer != layer:
            self.arena_residents = set()
        if (
            getattr(self, "prefill_direct_cache", False)
            and getattr(self, "cache_overlap_policy", "always") == "auto"
            and not verify
            and (
                self.arena_residents is None
                or not any(
                    count and expert not in self.arena_residents
                    for expert, count in enumerate(counts[:-1])
                )
            )
        ):
            # A ready arena already needs no copies. Avoid building cache
            # partitions and paying a second expert dispatch on a warm call.
            stats["auto_arena_runs"] = stats.get("auto_arena_runs", 0) + 1
            stats["resident"] += 1
            return False
        if (
            getattr(self, "prefill_direct_cache", False)
            and self.prefill_reuse_cache
            and not verify
            and keys is not None
            and job is not None
        ):
            # The borrowed arena overwrites only the overlapping tail slots.
            # Prefix slots remain valid and must stay protected until computed.
            overlap = self.borrowed_layers.get(layer)
            limit = overlap[0] if overlap is not None else len(keys)
            resident = {
                expert: slot
                for slot, expert in enumerate(keys[:limit])
                if expert >= 0 and counts[expert]
            }
            if getattr(self, "cache_overlap_policy", "always") == "auto":
                path = "arena"
                if resident:
                    from sglang.srt.layers.moe.qwen38_cache_overlap_policy import (
                        select_prefill_cache_path,
                    )

                    path = select_prefill_cache_path(
                        counts,
                        resident,
                        self.arena_residents,
                        hidden=self.manifest.hidden_size,
                        width=self.manifest.intermediate_size,
                        top_k=self.manifest.top_k,
                        w4a4=job.prefill_w4a4,
                    )
                counter = f"auto_{path}_runs"
                stats[counter] = stats.get(counter, 0) + 1
                if path == "arena":
                    resident = {}
            if resident:
                stats["direct_cache_runs"] = stats.get("direct_cache_runs", 0) + 1
                stats["direct_cache_experts"] = stats.get(
                    "direct_cache_experts", 0
                ) + len(resident)
                job.cache_experts = resident
                job.arena_experts = [
                    e
                    for e, count in enumerate(counts[:-1])
                    if count and e not in resident
                ]
                # Publish immutable route metadata BEFORE staging/enqueueing
                # missing weights; cache readers need not wait for arena H2D.
                job.plan_ready.set()
                missing = [
                    e
                    for e in job.arena_experts
                    if self.arena_residents is not None
                    and e not in self.arena_residents
                ]
                stats["missing"] += len(missing)
                stats["host_missing"] += len(missing)
                stats["missing_runs"] += bool(missing) + sum(
                    right != left + 1 for left, right in zip(missing, missing[1:])
                )
                if missing:
                    self._fill_arena_selected(layer, missing)
                    self.arena_residents.update(missing)
                    stats["sparse"] += 1
                else:
                    stats["resident"] += 1
                # Do not seed/warm cache from partially populated arena here.
                return bool(missing)
        if self.arena_residents is None:
            stats["resident"] += 1
            return False
        missing = [
            expert
            for expert, count in enumerate(counts[:-1])
            if count and expert not in self.arena_residents
        ]
        if not missing:
            stats["resident"] += 1
            return False
        stats["missing"] += len(missing)
        if force_dense:
            pack = getattr(self, "pack", None)
            if pack is not None:
                pack.wait_prefetch()
            self._fill_arena(layer)
            self.arena_residents = None
            stats["dense"] += 1
            stats["startup_dense"] += 1
            return True
        resident = set()
        if (
            self.prefill_reuse_cache
            and keys is not None
            and layer not in self.borrowed_layers
        ):
            resident = {expert for expert in keys if expert >= 0}
        reused = [expert for expert in missing if expert in resident]
        host_missing = [expert for expert in missing if expert not in resident]
        stats["host_missing"] += len(host_missing)
        stats["reuse_candidates"] += len(reused)
        stats["missing_runs"] += bool(host_missing) + sum(
            right != left + 1 for left, right in zip(host_missing, host_missing[1:])
        )
        # Decide before launching scatter; a fragmented, nearly full layer is
        # cheaper to replace than to merge with the decode cache.
        dense = not verify and (
            self.prefill_copy_mode == "dense"
            or (
                self.prefill_copy_mode == "auto"
                and _prefer_dense_prefill(host_missing, self.manifest.num_experts)
            )
        )
        if dense:
            self._fill_arena(layer)
            self.arena_residents = None
            stats["dense"] += 1
            stats[
                (
                    "forced_dense"
                    if self.prefill_copy_mode == "dense"
                    else "heuristic_dense"
                )
            ] += 1
            return True
        stats["sparse"] += 1
        if reused:
            # planned completion orders lookup/keys; prior fills on this
            # same transfer stream precede scatter. Later same-layer work
            # waits for the arena computation, so source slots stay stable.
            state = self.device_states[layer]
            with torch.cuda.stream(self.transfer):
                for name in REQUIRED_TENSORS:
                    cache_scatter(
                        self.caches[layer][name],
                        self.arena[name],
                        state["keys"],
                        state["counts"],
                    )
            self.prefill_reused_experts += len(reused)
            self.prefill_reuse_d2d_bytes += sum(
                self.manifest.tensors[layer][name].nbytes
                // self.manifest.num_experts
                * len(reused)
                for name in REQUIRED_TENSORS
            )
            self.arena_residents.update(reused)
        # Sparse copies avoid transferring the other experts on short prefills.
        if host_missing:
            self._fill_arena_selected(layer, host_missing)
            self.arena_residents.update(host_missing)
        return True

    def _analyze_routes(self):
        cycle = None
        routes = None
        while not self.analysis_stopping.is_set():
            try:
                item = self.adaptive_samples.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.adaptive_samples.task_done()
                return
            try:
                for layer, sample_cycle, index, active in _sample_rows(item):
                    if cycle != sample_cycle:
                        cycle = sample_cycle
                        routes = [[] for _ in range(self.manifest.num_layers)]
                    if routes is None:
                        continue
                    if len(routes[layer]) != index:
                        # Never replay a sequence with dropped samples as contiguous.
                        routes = None
                        continue
                    routes[layer].append(active)
                    if min(map(len, routes)) < self.adaptive_cache_window:
                        continue
                    if self.adaptive_cache_mode == "sample":
                        routes = None
                        continue
                    with self.adaptive_lock:
                        current = list(self.layer_cache_slots)
                        budget = sum(current)
                    request = dict(
                        routes=routes,
                        current=current,
                        num_experts=self.manifest.num_experts,
                        top_k=self.manifest.top_k,
                        min_slots=getattr(
                            self, "adaptive_min_slots", self.manifest.top_k
                        ),
                        budget=budget,
                        policy=self.cache_policy,
                        protected_fraction=self.cache_protected_fraction,
                    )
                    self.adaptive_process.stdin.write(json.dumps(request) + "\n")
                    self.adaptive_process.stdin.flush()
                    deadline = time.monotonic() + 120
                    while not select.select([self.adaptive_process.stdout], [], [], 1)[
                        0
                    ]:
                        if self.analysis_stopping.is_set():
                            return
                        if time.monotonic() >= deadline:
                            raise TimeoutError("adaptive cache analysis timed out")
                    reply = json.loads(self.adaptive_process.stdout.readline())
                    routes = None
                    self.adaptive_evaluations += 1
                    self.adaptive_analysis_seconds += reply["seconds"]
                    self.adaptive_last_evaluation = reply["evaluation"]
                    logger.info(
                        "Qwen38 adaptive cache evaluation: %s; analysis=%.3fs",
                        reply["evaluation"],
                        reply["seconds"],
                    )
                    if (
                        reply["target"] is not None
                        and self.adaptive_cache_mode == "rebalance"
                    ):
                        with self.adaptive_lock:
                            if (
                                self.layer_cache_slots == current
                                and self.adaptive_pending is None
                            ):
                                self.adaptive_target = reply["target"]
            except BaseException:
                logger.exception("Disabling adaptive expert-cache analysis")
                self.adaptive_cache = False
                return
            finally:
                release = getattr(item, "release", None)
                if release is not None:
                    release()
                self.adaptive_samples.task_done()

    def initialize_staged_layout(self):
        """Publish initial adaptive budget after target KV construction."""
        from sglang.srt.mem_cache.qwen38_kv_stages import distribute_slots

        budget = self.staged_pool.budgets[0]
        target = distribute_slots(
            self.layer_cache_slots,
            budget,
            getattr(self, "adaptive_min_slots", self.manifest.top_k),
            self.manifest.num_experts,
        )
        self.cache_pool = self.staged_pool.pool_views(budget)
        self._assign_pool_views(target)
        self.migration_scratch = {
            name: torch.empty_like(pool[0]) for name, pool in self.cache_pool.items()
        }
        self._refresh_staged_arena()
        self.layer_cache_slots = list(target)
        self.max_layer_slots = max(target)
        self.device_states.clear()
        for layer in range(self.manifest.num_layers):
            self.policies[layer] = LayerExpertCachePolicy(
                self.manifest.num_experts, target[layer]
            )
            self._state(layer)
        if self.miss_mailbox is not None:
            self.miss_mailbox.refresh_targets()
        if self.packed_transport is not None:
            self.packed_transport.refresh_targets()
        self.pending_kv_stage = None
        self.staged_trim_enqueued = False

    def _refresh_staged_arena(self):
        if self.borrowed_arena:
            self.arena = {
                name: tensor[-self.manifest.num_experts :]
                for name, tensor in self.cache_pool.items()
            }

    def _preserve_staged_layout(self, target, budget, snapshot):
        """Move retained hot rows before shrinking physical pool backing.

        Caller owns runtime/adaptive locks and has drained existing readers.
        Expose both old sources and new destinations during overlapping moves.
        """
        old_budget = sum(self.layer_cache_slots)
        self.cache_pool = self.staged_pool.pool_views(max(old_budget, budget))
        retained, migrated, evicted = self._migrate_pool(target, snapshot)
        self.cache_pool = self.staged_pool.pool_views(budget)
        self._assign_pool_views(target)
        self.layer_cache_slots = list(target)
        self.max_layer_slots = max(target)
        self.device_states.clear()
        self.slot_ready.clear()
        self.slot_last_use.clear()
        self.warm_ready.clear()
        self.layer_computed.clear()
        self.policies = {
            layer: LayerExpertCachePolicy(self.manifest.num_experts, slots)
            for layer, slots in enumerate(target)
        }
        for layer in range(self.manifest.num_layers):
            self._state(layer)
        self._restore_device_policy(retained)
        if self.cache_policy == "slru":
            for layer, slots in enumerate(target):
                self.device_states[layer]["frequency"][
                    int(slots * self.cache_protected_fraction) :
                ].zero_()
        self._refresh_staged_arena()
        self.arena_layer = None
        self.arena_residents = set()
        self.arena_last_use = None
        self.gate_warmed_layers.clear()
        if self.miss_mailbox is not None:
            self.miss_mailbox.refresh_targets(non_blocking=True)
        if self.packed_transport is not None:
            self.packed_transport.refresh_targets(non_blocking=True)
        stream = torch.cuda.current_stream(self.device)
        self.metadata_stream.wait_stream(stream)
        self.transfer.wait_stream(stream)
        self.adaptive_migrated_experts += migrated
        self.adaptive_evicted_experts += evicted

    def complete_staged_trim(self):
        """Retain hottest residents after the existing result snapshot fence.

        Migration writes must finish before KV unmaps/reuses the expert tail.
        This opt-in path pays one migration completion wait at stage changes.
        """
        pending = getattr(self, "staged_preserve_pending", None)
        if pending is None:
            return
        target, budget, stream = pending
        if not self.staged_trim_done.query():
            raise RuntimeError(
                "staged policy snapshot was not covered by result completion"
            )
        with self.adaptive_lock, self.lock, torch.no_grad(), torch.cuda.stream(stream):
            try:
                self._return_arena()
                self._drain_arena_readers()
                snapshot = [
                    tuple(
                        self.adaptive_metadata.host[layer][name][:slots].tolist()
                        for name in ("keys", "frequency", "ages")
                    )
                    for layer, slots in enumerate(self.layer_cache_slots)
                ]
                self._preserve_staged_layout(target, budget, snapshot)
                self.staged_trim_done.record(stream)
                self.staged_trim_done.synchronize()
            except BaseException:
                self.poisoned = True
                raise
            self.staged_preserve_pending = None

    def grow_staged_layout(self, index):
        """Publish newly backed expert slots after KV tail release."""
        from sglang.srt.mem_cache.qwen38_kv_stages import distribute_slots

        budget = self.staged_pool.budgets[index]
        target = distribute_slots(
            self.layer_cache_slots,
            budget,
            getattr(self, "adaptive_min_slots", self.manifest.top_k),
            self.manifest.num_experts,
        )
        with self.adaptive_lock, self.lock:
            self.adaptive_target = None
            if self.preserve_cache_on_resize:
                self._return_arena()
                self._drain_arena_readers()
                snapshot = [
                    tuple(
                        self.device_states[layer][name].cpu().tolist()
                        for name in ("keys", "frequency", "ages")
                    )
                    for layer in range(self.manifest.num_layers)
                ]
                self._preserve_staged_layout(target, budget, snapshot)
                return
            borrowed = set(self.borrowed_layers)
            self.borrowed_records = None
            self.borrowed_layers = {}
            self.arena_layer = None
            self.arena_residents = set()
            self.arena_last_use = None
            old = list(self.layer_cache_slots)
            offset_old = offset_new = 0
            for layer, slots in enumerate(target):
                state = self.device_states[layer]
                if offset_old != offset_new or layer in borrowed:
                    self.device_states[layer] = self.adaptive_metadata.reset(
                        layer, slots
                    )
                    self.slot_ready[layer] = [None] * slots
                    self.slot_last_use[layer] = [None] * slots
                else:
                    for name in ("keys", "frequency", "ages", "generation"):
                        full = self.adaptive_metadata.device[layer][name]
                        full[old[layer] : slots].fill_(-1 if name == "keys" else 0)
                        state[name] = full[:slots]
                    state["plan"] = self.adaptive_metadata.device[layer]["plan"][
                        : 4 + 4 * slots
                    ]
                    state["prefill_plan"] = self.adaptive_metadata.device[layer][
                        "prefill_plan"
                    ][: state["counts"].numel() + slots]
                    self.slot_ready[layer] += [None] * (slots - old[layer])
                    self.slot_last_use[layer] += [None] * (slots - old[layer])
                self.policies[layer] = LayerExpertCachePolicy(
                    self.manifest.num_experts, slots
                )
                offset_old += old[layer]
                offset_new += slots
            self.cache_pool = self.staged_pool.pool_views(budget)
            self._assign_pool_views(target)
            self._refresh_staged_arena()
            self.layer_cache_slots = list(target)
            self.max_layer_slots = max(target)
            self.gate_warmed_layers.clear()
            if self.miss_mailbox is not None:
                self.miss_mailbox.refresh_targets(non_blocking=True)
            if self.packed_transport is not None:
                self.packed_transport.refresh_targets(non_blocking=True)
            stream = torch.cuda.current_stream(self.device)
            self.metadata_stream.wait_stream(stream)
            self.transfer.wait_stream(stream)

    def enqueue_staged_trim(self):
        """Shrink/repartition behind current readers, without a new host wait.

        Layers whose pool offsets change are invalidated rather than migrated.
        Unchanged prefix layers preserve policy and payload in place.
        """
        from sglang.srt.mem_cache.qwen38_kv_stages import distribute_slots

        index = self.pending_kv_stage
        if index is None or self.staged_trim_enqueued:
            return
        budget = self.staged_pool.budgets[index]
        target = distribute_slots(
            self.layer_cache_slots,
            budget,
            getattr(self, "adaptive_min_slots", self.manifest.top_k),
            self.manifest.num_experts,
        )
        if self.preserve_cache_on_resize:
            # Snapshot on the batch stream; its existing result fence makes
            # host policy available before the KV tail is physically returned.
            with self.adaptive_lock:
                self.adaptive_target = None
                stream = torch.cuda.current_stream(self.device)
                for layer, state in self.device_states.items():
                    for name in ("keys", "frequency", "ages"):
                        self.adaptive_metadata.host[layer][name][
                            : state[name].numel()
                        ].copy_(state[name], non_blocking=True)
                self.staged_preserve_pending = (list(target), budget, stream)
                self.staged_trim_done = torch.cuda.Event()
                self.staged_trim_done.record(stream)
                self.staged_trim_enqueued = True
            return
        borrowed = set(self.borrowed_layers)
        self.borrowed_records = None
        self.borrowed_layers = {}
        self.arena_layer = None
        self.arena_residents = set()
        self.arena_last_use = None
        old_offset = new_offset = 0
        with self.adaptive_lock:
            self.adaptive_target = None
            for layer, slots in enumerate(target):
                state = self.device_states[layer]
                old_slots = self.layer_cache_slots[layer]
                if old_offset != new_offset or layer in borrowed:
                    self.device_states[layer] = self.adaptive_metadata.reset(
                        layer, slots
                    )
                    self.slot_ready[layer] = [None] * slots
                    self.slot_last_use[layer] = [None] * slots
                else:
                    state["lookup"].masked_fill_(state["lookup"] >= slots, -1)
                    for name in ("keys", "frequency", "ages", "generation"):
                        state[name] = state[name][:slots]
                    state["plan"] = state["plan"][: 4 + 4 * slots]
                    state["prefill_plan"] = state["prefill_plan"][
                        : state["counts"].numel() + slots
                    ]
                    self.slot_ready[layer] = self.slot_ready[layer][:slots]
                    self.slot_last_use[layer] = self.slot_last_use[layer][:slots]
                self.policies[layer] = LayerExpertCachePolicy(
                    self.manifest.num_experts, slots
                )
                old_offset += old_slots
                new_offset += slots
            self.cache_pool = self.staged_pool.pool_views(budget)
            self._assign_pool_views(target)
            self._refresh_staged_arena()
            self.layer_cache_slots = list(target)
            self.max_layer_slots = max(target)
            self.gate_warmed_layers.clear()
            # Address tables upload on this stream before result completion.
            if self.miss_mailbox is not None:
                self.miss_mailbox.refresh_targets(non_blocking=True)
            if self.packed_transport is not None:
                self.packed_transport.refresh_targets(non_blocking=True)
            stream = torch.cuda.current_stream(self.device)
            self.metadata_stream.wait_stream(stream)
            self.transfer.wait_stream(stream)
        self.staged_trim_done = torch.cuda.Event()
        self.staged_trim_done.record(stream)
        self.staged_trim_enqueued = True

    def enqueue_adaptive_snapshot(self):
        """Read policy behind this batch, covered by its existing result fence."""
        if self.borrowed_records is not None or self.adaptive_pending is not None:
            return
        with self.adaptive_lock:
            target = self.adaptive_target
            self.adaptive_target = None
            if target is None or target == self.layer_cache_slots:
                return
            minimum = getattr(self, "adaptive_min_slots", self.manifest.top_k)
            if (
                len(target) != self.manifest.num_layers
                or any(
                    type(slots) is not int
                    or not minimum <= slots <= self.manifest.num_experts
                    for slots in target
                )
                or sum(target) != next(iter(self.cache_pool.values())).shape[0]
            ):
                raise ValueError(
                    "invalid adaptive layout or incomplete verify active-set capacity"
                )
            stream = torch.cuda.current_stream(self.device)
            for layer, state in self.device_states.items():
                for name in ("keys", "frequency", "ages"):
                    self.adaptive_metadata.host[layer][name][
                        : state[name].numel()
                    ].copy_(state[name], non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
            self.adaptive_pending = (list(target), stream, done)

    def adaptive_after_result(self):
        """Apply after result processing, submitting all writes on the old stream."""
        pending = self.adaptive_pending
        if pending is None:
            return
        target, stream, done = pending
        # Query only: unsupported result paths without an existing completion
        # wait defer publication. Never create a host wait for maintenance.
        if not done.query():
            self.poisoned = True
            raise RuntimeError("adaptive snapshot was not covered by result completion")
        with self.adaptive_lock, self.lock, torch.no_grad(), torch.cuda.stream(stream):
            try:
                started = time.perf_counter()
                old = list(self.layer_cache_slots)
                # Consume every D2H buffer before restore overwrites them.
                snapshot = [
                    tuple(
                        self.adaptive_metadata.host[layer][name][:slots].tolist()
                        for name in ("keys", "frequency", "ages")
                    )
                    for layer, slots in enumerate(old)
                ]
                retained, migrated, evicted = self._migrate_pool(target, snapshot)
                self._assign_pool_views(target)
                if hasattr(self, "staged_pool"):
                    self._refresh_staged_arena()
                if self.miss_mailbox is not None:
                    self.miss_mailbox.refresh_targets(non_blocking=True)
                if self.packed_transport is not None:
                    self.packed_transport.refresh_targets(non_blocking=True)
                self.layer_cache_slots = list(target)
                self.max_layer_slots = max(target)
                self.policies = {
                    layer: LayerExpertCachePolicy(self.manifest.num_experts, slots)
                    for layer, slots in enumerate(target)
                }
                self.device_states.clear()
                self.slot_ready.clear()
                self.slot_last_use.clear()
                self.warm_ready.clear()
                self.layer_computed.clear()
                for layer in range(self.manifest.num_layers):
                    self._state(layer)
                self._restore_device_policy(retained)
                if self.cache_policy == "slru":
                    for layer, state in self.device_states.items():
                        state["frequency"][
                            int(target[layer] * self.cache_protected_fraction) :
                        ].zero_()
                self.metadata_stream.wait_stream(stream)
                self.transfer.wait_stream(stream)
            except BaseException:
                self.poisoned = True
                raise
            self.adaptive_pending = None
            self.adaptive_rebalances += 1
            self.gate_warmed_layers.clear()
            self.adaptive_migrated_experts += migrated
            self.adaptive_evicted_experts += evicted
            self.adaptive_last_rebalance_seconds = time.perf_counter() - started
            logger.info(
                "Qwen38 adaptive cache: layer slots %s -> %s; "
                "migrated=%d evicted=%d enqueue=%.3fs",
                old,
                target,
                migrated,
                evicted,
                self.adaptive_last_rebalance_seconds,
            )

    def stats_snapshot(self):
        if getattr(self, "miss_mailbox", None) is not None:
            self.miss_mailbox.flush_stats()
        result = super().stats_snapshot()
        result["decode_shared_fill_overlap_invocations"] = getattr(
            getattr(self, "miss_mailbox", None), "overlap_invocations", 0
        )
        if hasattr(self, "staged_pool"):
            result["kv_cache_stages"] = self.kv_cache_stages
            result["expert_loan_budgets"] = getattr(self, "expert_loan_budgets", None)
            result["expert_physical_bytes"] = (
                self.staged_pool.arena.backed_bytes + self.staged_pool.auxiliary_bytes
            )
        result["borrowed_prefill_arena"] = self.borrowed_arena
        result["borrowed_restore_runs"] = self.borrowed_restore_runs
        result["borrowed_restore_experts"] = self.borrowed_restore_experts
        result["packed_decode"] = self.packed_decode
        result["packed_prefill"] = self.packed_prefill
        result["packed_transport"] = (
            dict(self.packed_transport.counters)
            if self.packed_transport is not None
            else None
        )
        if self.adaptive_cache:
            result.update(
                adaptive_rebalances=self.adaptive_rebalances,
                adaptive_sample_drops=self.adaptive_sample_drops,
                adaptive_migrated_experts=self.adaptive_migrated_experts,
                adaptive_evicted_experts=self.adaptive_evicted_experts,
                adaptive_last_rebalance_seconds=self.adaptive_last_rebalance_seconds,
                layer_hits=list(self.layer_hits),
                layer_misses=list(self.layer_misses),
                adaptive_evaluations=self.adaptive_evaluations,
                adaptive_analysis_seconds=self.adaptive_analysis_seconds,
                adaptive_last_evaluation=self.adaptive_last_evaluation,
            )
        result.update(
            cache_policy=self.cache_policy,
            cache_protected_fraction=self.cache_protected_fraction,
            layer_cache_slots=list(self.layer_cache_slots),
            cache_slots_total=(
                next(iter(self.cache_pool.values())).shape[0]
                if self.cache_pool
                else sum(self.layer_cache_slots)
            ),
            layer_hits=list(self.layer_hits),
            layer_misses=list(self.layer_misses),
            adaptive_cache_mode=self.adaptive_cache_mode,
            adaptive_cache_window=self.adaptive_cache_window,
            adaptive_cache_interval=self.adaptive_cache_interval,
            host_plan_allocations=self.host_plan_allocations,
            prefill_reuse_cache=self.prefill_reuse_cache,
            prefill_direct_cache=getattr(self, "prefill_direct_cache", False),
            cache_overlap_policy=getattr(self, "cache_overlap_policy", "always"),
            decode_early_cache=getattr(self, "decode_early_cache", False),
            preserve_cache_on_resize=getattr(self, "preserve_cache_on_resize", False),
            decode_early_cache_invocations=getattr(
                getattr(self, "miss_mailbox", None), "early_cache_invocations", 0
            ),
            prefill_seed_cache=self.prefill_seed_cache,
            fused_decode_plan=self.fused_decode_plan,
            fused_decode_plan_invocations=self.fused_decode_plan_invocations,
            decode_hit_fast_path=self.decode_hit_fast_path,
            decode_hit_fast_path_invocations=self.decode_hit_fast_path_invocations,
            decode_device_gate=self.decode_device_gate,
            decode_device_gate_invocations=self.decode_device_gate_invocations,
            decode_device_gate_hits=self.decode_device_gate_hits,
            decode_device_gate_copy_calls=self.decode_device_gate_copy_calls,
            decode_device_gate_copy_bytes=self.decode_device_gate_copy_bytes,
            resident_alphas=self.resident_alphas,
            decode_alpha_h2d_calls_avoided=self.decode_alpha_h2d_calls_avoided,
            prefill_copy_mode=self.prefill_copy_mode,
            prefill_copy_stats=[dict(stats) for stats in self.prefill_copy_stats],
            prefill_reused_experts=self.prefill_reused_experts,
            prefill_reuse_d2d_bytes=self.prefill_reuse_d2d_bytes,
            prefill_seed_runs=self.prefill_seed_runs,
            prefill_h2d_bytes=self.prefill_h2d_bytes,
            prefill_host_dispatch_seconds=self.prefill_host_dispatch_seconds,
            prefill_tokens_seen=self.prefill_tokens_seen,
            prefill_dense_startup_tokens=_DENSE_STARTUP_PREFILL_TOKENS,
        )
        return result

    def _raise_error(self):
        if self.poisoned:
            raise RuntimeError("threaded expert runtime is poisoned; recreate runtime")
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
        job.plan_ready.set()
        # Retire failed jobs exactly like finished ones so permit release and
        # queue bookkeeping stay uniform across the retirement thread.
        self.retired.put(job)

    def _finish_job(self, job, record_event=True):
        if record_event:
            job.ready = torch.cuda.Event()
            job.ready.record(self.transfer)
        job.ready_set.set()
        job.plan_ready.set()
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
                sample_published = getattr(job, "sample_published", None)
                try:
                    if job.plan_values is None:
                        job.planned.synchronize()
                    self.transfer.wait_event(job.planned)  # policy/allocation order
                    if self.worker_error is not None:
                        raise RuntimeError("previous transfer failed")
                    if job.arena:
                        values = job.plan.tolist()
                        counts = values[: self.manifest.num_experts + 1]
                        keys = values[self.manifest.num_experts + 1 :]
                        if counts[-1]:
                            raise ValueError("invalid prefill route ID")
                        started = time.perf_counter()
                        bytes_before = self.transferred_bytes
                        if self.arena_layer != job.layer:
                            if self.arena_last_use is not None:
                                self.transfer.wait_event(self.arena_last_use)
                            self.arena_layer = None
                            loaded = self._ensure_arena_experts(
                                job.layer,
                                counts,
                                keys,
                                force_dense=job.force_dense,
                                verify=job.verify,
                                job=job,
                            )
                            self.arena_layer = job.layer
                        elif self.arena_last_use is not None:
                            self.transfer.wait_event(self.arena_last_use)
                            loaded = self._ensure_arena_experts(
                                job.layer,
                                counts,
                                keys,
                                force_dense=job.force_dense,
                                verify=job.verify,
                                job=job,
                            )
                        else:
                            loaded = self._ensure_arena_experts(
                                job.layer,
                                counts,
                                keys,
                                force_dense=job.force_dense,
                                verify=job.verify,
                                job=job,
                            )
                        if loaded:
                            self.arena_loads += 1
                        seed = (
                            not job.verify
                            and self.prefill_seed_cache
                            and bool(keys)
                            and all(key < 0 for key in keys)
                        )
                        if (
                            not job.verify
                            and job.cache_experts is None
                            and (self.warm_decode_cache or seed)
                            and job.layer not in self.borrowed_layers
                        ):
                            self._warm_from_arena(job.layer, counts[:-1])
                            if seed:
                                self.prefill_seed_runs += 1
                            if self.cache_policy == "slru":
                                # Prefill counts seed probation; decode reuse promotes.
                                with torch.cuda.stream(self.transfer):
                                    self.device_states[job.layer]["frequency"].zero_()
                        self.prefill_h2d_bytes += self.transferred_bytes - bytes_before
                        self.prefill_host_dispatch_seconds += (
                            time.perf_counter() - started
                        )
                        # Metadata (route/plan kernels) ran on the metadata
                        # stream; nothing else needs to move for full reads.
                        self._finish_job(job)
                        self.arena_last_use = job.computed
                    else:
                        capacity = self.slots_for_layer(job.layer)
                        values = (
                            job.plan_values
                            if job.plan_values is not None
                            else job.plan.tolist()
                        )
                        status, misses, active, arena = values[:4]
                        if status or arena:
                            raise ValueError(
                                f"unexpected device transfer plan: status={status}, arena={arena}"
                            )
                        sample = None
                        if job.sample is not None:
                            sample = (
                                job.layer,
                                *job.sample,
                                values[4 + 3 * capacity : 4 + 3 * capacity + active],
                            )
                        active_slots = values[
                            4 + 2 * capacity : 4 + 2 * capacity + active
                        ]
                        for index in range(misses):
                            expert, slot = values[4 + 2 * index : 6 + 2 * index]
                            last = self.slot_last_use[job.layer][slot]
                            if last is not None:
                                # The overwritten slot must have finished its
                                # previous reader before its bytes change.
                                self.transfer.wait_event(last)
                            self._fill(
                                job.layer,
                                expert,
                                self.caches[job.layer],
                                slot,
                                names=self.decode_copy_names,
                            )
                        if misses and self.resident_alphas:
                            # All victim readers were joined above. The job
                            # event covers both weight H2D and alpha writes.
                            tables = self.alpha_tables[job.layer]
                            tensors = self.caches[job.layer]
                            with torch.cuda.stream(self.transfer):
                                cache_alpha_scatter(
                                    tables["g1_alphas"],
                                    tables["g2_alphas"],
                                    tensors["g1_alphas"],
                                    tensors["g2_alphas"],
                                    self.device_states[job.layer]["plan"],
                                )
                            self.decode_alpha_h2d_calls_avoided += 2 * misses
                        # Metadata is ordered after the previous same-layer
                        # compute; read-only hits are complete with no CPU wait.
                        for slot in active_slots:
                            self.slot_last_use[job.layer][slot] = job.computed
                        self.misses += misses
                        self.hits += active - misses
                        self.layer_misses[job.layer] += misses
                        self.layer_hits[job.layer] += active - misses
                        # Hits may need no copy at all; the ready event still
                        # has to exist and order any misses that were issued.
                        self._finish_job(job, record_event=bool(misses))
                        # Publish readiness before any analysis queue work.
                        if sample is not None and self.adaptive_cache:
                            try:
                                self.adaptive_samples.put_nowait(sample)
                            except queue.Full:
                                self.adaptive_sample_drops += 1
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
                    if sample_published is not None:
                        sample_published.set()
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
                    if job.plan is not None and hasattr(self, "host_plans"):
                        self.host_plans[job.plan.numel()].put(job.plan)
                    self.capacity.release()
                    self.jobs.task_done()
                    self.retired.task_done()

    def synchronize(self):
        # CUDA completion alone does not drain the retirement thread. Its
        # event synchronization must finish before a caller starts graph capture.
        self.jobs.join()
        if self.miss_mailbox is not None:
            self.miss_mailbox.synchronize()
        super().synchronize()

    def close(self):
        if self.closed:
            return
        try:
            self.synchronize()
        finally:
            self.closed = True
            if self.miss_mailbox is not None:
                self.miss_mailbox.close()
            if self.worker is not None:
                self.jobs.put(None)
                self.worker.join()
                self.retired.put(None)
                self.reaper.join()
            if self.adaptive_worker is not None and self.adaptive_worker.is_alive():
                self.analysis_stopping.set()
                self.adaptive_worker.join()
            if self.adaptive_process is not None:
                self.adaptive_process.stdin.close()
                try:
                    self.adaptive_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.adaptive_process.kill()
                    self.adaptive_process.wait()
                self.adaptive_process.stdout.close()
            if hasattr(self, "staged_pool"):
                self.caches.clear()
                self.cache_pool = None
                self.arena = None
                self.staged_pool.close()

    def memory_budget(self):
        result = super().memory_budget()
        if self.borrowed_arena:
            result["borrowed_prefill_arena_bytes"] = result["prefill_arena_bytes"]
            result["prefill_arena_bytes"] = 0
        result["adaptive_metadata_bytes"] = (
            self.adaptive_metadata.nbytes() if hasattr(self, "adaptive_metadata") else 0
        )
        result["migration_scratch_bytes"] = sum(
            t.numel() * t.element_size()
            for t in getattr(self, "migration_scratch", {}).values()
        )
        result["resident_alpha_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for tables in self.alpha_tables.values()
            for tensor in tables.values()
        )
        result["packed_scratch_bytes"] = (
            self.packed_transport.scratch.numel()
            if self.packed_transport is not None
            else 0
        )
        result["packed_descriptor_bytes"] = (
            sum(
                t.numel() * t.element_size()
                for pages in self.packed_transport.tables.values()
                for t in pages
            )
            if self.packed_transport is not None
            else 0
        )
        # Each outstanding decode job owns one pinned transfer plan.
        result["max_pending_plan_bytes"] = (
            self.jobs.maxsize
            * max(
                4 + (4 if self.adaptive_cache else 3) * self.max_layer_slots,
                self.manifest.num_experts + 1 + self.max_layer_slots,
            )
            * 4
        )
        result["decode_device_gate_bytes"] = (
            self.miss_mailbox.device_bytes if self.miss_mailbox is not None else 0
        )
        result["decode_mailbox_pinned_bytes"] = (
            self.miss_mailbox.pinned_bytes if self.miss_mailbox is not None else 0
        )
        return result

    def _use_prefill_arena(self, layer_id, route_bound):
        capacity = self.slots_for_layer(layer_id)
        if self.prefill_arena:
            capacity = min(capacity, self.prefill_layer_slots[layer_id])
        return route_bound > capacity

    def _dense_startup_prefill(self, layer_id, arena, tokens):
        if not arena:
            return False
        if layer_id == 0:
            # Each layer sees the same chunk; count it once at the first layer.
            self.prefill_dense_current = (
                self.prefill_tokens_seen < _DENSE_STARTUP_PREFILL_TOKENS
            )
            self.prefill_tokens_seen += tokens
        return self.prefill_dense_current

    supports_execution_phase = True
    supports_fill_overlap = True

    def run(
        self,
        layer_id,
        topk_ids,
        execute,
        *,
        execution_phase=None,
        during_fill=None,
        early_cache=None,
        prefill_w4a4=False,
    ):
        result = self._run(
            layer_id,
            topk_ids,
            execute,
            execution_phase=execution_phase,
            during_fill=during_fill,
            early_cache=early_cache,
            prefill_w4a4=prefill_w4a4,
        )
        if layer_id == self.manifest.num_layers - 1:
            if hasattr(self, "staged_pool"):
                self.enqueue_staged_trim()
                loan = getattr(self, "kv_loan", None)
                if loan is not None:
                    loan.enqueue_reborrow_probe()
            if (
                self.adaptive_cache
                and getattr(self, "pending_kv_stage", None) is None
                and not getattr(self, "kv_reborrow_probe", False)
            ):
                self.enqueue_adaptive_snapshot()
        return result

    def _run(
        self,
        layer_id,
        topk_ids,
        execute,
        *,
        execution_phase=None,
        during_fill=None,
        early_cache=None,
        prefill_w4a4=False,
    ):
        with self.lock, torch.no_grad(), torch.cuda.device(self.device):
            self._raise_error()
            publication = getattr(self, "sample_publication", None)
            if publication is not None:
                # Legacy verify can signal weight readiness before emitting
                # its route. Order observer messages across transport changes.
                if not publication.wait(_ready_timeout_seconds() or None):
                    self.poisoned = True
                    raise RuntimeError("legacy route sample publication timed out")
                self.sample_publication = None
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
            capacity = self.slots_for_layer(layer_id)
            arena = self._use_prefill_arena(layer_id, topk_ids.numel())
            force_dense = execution_phase != "verify" and self._dense_startup_prefill(
                layer_id, arena, topk_ids.shape[0]
            )
            self._record_invocation(arena, topk_ids, layer_id)
            self.sampling_verify = execution_phase == "verify"
            if arena and self.arena is None:
                raise ValueError("route bound exceeds cache; enable prefill arena")
            if self.borrowed_arena:
                if arena:
                    self._borrow_arena()
                else:
                    self._return_arena()
            sample = self._sample_position(layer_id, topk_ids)
            if (
                self.miss_mailbox is not None
                and (
                    execution_phase == "verify"
                    or early_cache is not None
                    or layer_id in self.gate_warmed_layers
                    or sample is not None
                )
                and self.fused_decode_plan
                and not arena
                and (topk_ids.shape[0] == 1 or execution_phase == "verify")
                and 0 < topk_ids.numel() <= 32
            ):
                result = self.miss_mailbox.run(
                    layer_id,
                    topk_ids,
                    execute,
                    _ready_timeout_seconds(),
                    sample=sample,
                    during_fill=during_fill,
                    early_cache=early_cache,
                )
                self.gate_warmed_layers.add(layer_id)
                return result
            if during_fill is not None:
                # Cold non-sampling decode may still warm via the legacy path.
                # Compute the shared branch once, with its original math.
                during_fill()
            if self.miss_mailbox is not None:
                # Drain readers and hand off partial route samples before
                # entering prefill, multi-token verify, or another legacy path.
                self.miss_mailbox.synchronize()
            self.capacity.acquire()  # bounded producer backpressure, not a per-layer CUDA wait
            queued = False
            fast_hit = False
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
                if layer_id in self.warm_ready:
                    self.metadata_stream.wait_event(self.warm_ready[layer_id])
                if arena:
                    remapped = topk_ids
                    with torch.cuda.stream(self.metadata_stream):
                        scratch = torch.empty_like(topk_ids)
                        cache_route(topk_ids, state["lookup"], scratch, state["counts"])
                        if self.prefill_reuse_cache or self.prefill_seed_cache:
                            source = self._prefill_plan(state)
                            source[: state["counts"].numel()].copy_(state["counts"])
                            source[state["counts"].numel() :].copy_(state["keys"])
                        else:
                            source = state["counts"]
                        plan = self._host_plan(source)
                        plan.copy_(source, non_blocking=True)
                        planned.record(self.metadata_stream)
                    topk_ids.record_stream(self.metadata_stream)
                else:
                    with torch.cuda.stream(self.metadata_stream):
                        remapped = torch.empty_like(topk_ids)
                        if (
                            self.fused_decode_plan
                            and topk_ids.shape[0] == 1
                            and 0 < topk_ids.shape[1] <= 32
                        ):
                            cache_decode_plan(
                                topk_ids,
                                remapped,
                                state["counts"],
                                state["lookup"],
                                state["keys"],
                                state["frequency"],
                                state["ages"],
                                state["generation"],
                                state["clock"],
                                state["plan"],
                                policy=int(self.cache_policy == "slru"),
                                protected_fraction=self.cache_protected_fraction,
                            )
                            self.fused_decode_plan_invocations += 1
                        else:
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
                                policy=int(self.cache_policy == "slru"),
                                protected_fraction=self.cache_protected_fraction,
                            )
                            cache_route(
                                topk_ids, state["lookup"], remapped, state["counts"]
                            )
                        source = state["plan"]
                        if sample is None and source.numel() != 4 + 3 * capacity:
                            source = source[: 4 + 3 * capacity]
                        plan = self._host_plan(source)
                        plan.copy_(source, non_blocking=True)
                        planned.record(self.metadata_stream)
                    topk_ids.record_stream(self.metadata_stream)
                    remapped.record_stream(compute)
                compute.wait_event(planned)
                plan_values = None
                if self.decode_hit_fast_path and not arena and topk_ids.shape[0] == 1:
                    # Replace the worker's plan wait with the caller's wait.
                    # The bounded D2H remains, but all-hit decode needs no queue
                    # handoff, worker wakeup or retirement job.
                    planned.synchronize()
                    plan_values = plan.tolist()
                    if (
                        plan_values[0] == 0
                        and plan_values[1] == 0
                        and plan_values[3] == 0
                    ):
                        fast_hit = True
                        active = plan_values[2]
                        for slot in plan_values[
                            4 + 2 * capacity : 4 + 2 * capacity + active
                        ]:
                            self.slot_last_use[layer_id][slot] = computed
                        self.hits += active
                        self.layer_hits[layer_id] += active
                        if sample is not None and self.adaptive_cache:
                            try:
                                self.adaptive_samples.put_nowait(
                                    (
                                        layer_id,
                                        *sample,
                                        plan_values[
                                            4 + 3 * capacity : 4 + 3 * capacity + active
                                        ],
                                    )
                                )
                            except queue.Full:
                                self.adaptive_sample_drops += 1
                        self.decode_hit_fast_path_invocations += 1
                        prepared = PreparedExperts(
                            {**self.caches[layer_id], **self.layer_scales[layer_id]},
                            remapped,
                            capacity,
                            False,
                        )
                        result = execute(prepared)
                        if self.decode_device_gate:
                            self.gate_warmed_layers.add(layer_id)
                        return result
                job = TransferJob(
                    layer_id,
                    arena,
                    plan,
                    planned,
                    computed,
                    launched,
                    sample,
                    force_dense=force_dense,
                    prefill_w4a4=prefill_w4a4,
                    verify=execution_phase == "verify",
                    plan_values=plan_values,
                )
                if sample is not None:
                    job.sample_published = self.sample_publication = threading.Event()
                timeout = _ready_timeout_seconds()
                self.jobs.put_nowait(job)
                queued = True

                def wait_for_weights():
                    if timeout == 0:
                        job.ready_set.wait()
                    elif not job.ready_set.wait(timeout):
                        self.poisoned = True
                        raise RuntimeError("expert weights readiness deadline expired")
                    if job.error is not None:
                        self.poisoned = True
                        raise RuntimeError(
                            "background expert transfer failed"
                        ) from job.error
                    if job.ready is not None:
                        compute.wait_event(job.ready)

                direct = (
                    arena
                    and getattr(self, "prefill_direct_cache", False)
                    and execution_phase != "verify"
                )
                notification = job.plan_ready if direct else job.ready_set
                # Includes plan completion, host staging, and ring backpressure;
                # the final H2Ds need only be enqueued, not completed.
                if timeout == 0:
                    notification.wait()
                elif not notification.wait(timeout):
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
                deferred = direct and job.cache_experts is not None
                if not deferred and job.ready is not None:
                    compute.wait_event(job.ready)
                prepared = PreparedExperts(
                    {
                        **(self.arena if arena else self.caches[layer_id]),
                        **self.layer_scales[layer_id],
                    },
                    remapped,
                    self.manifest.num_experts if arena else capacity,
                    arena,
                    cache_tensors=(
                        {**self.caches[layer_id], **self.layer_scales[layer_id]}
                        if job.cache_experts is not None
                        else None
                    ),
                    cache_experts=job.cache_experts,
                    cache_lookup=state["lookup"]
                    if job.cache_experts is not None
                    else None,
                    cache_slot_limit=(
                        self.borrowed_layers[layer_id][0]
                        if layer_id in self.borrowed_layers
                        else capacity
                    ),
                    arena_experts=job.arena_experts,
                    wait_for_weights=wait_for_weights if deferred else None,
                )
                try:
                    result = execute(prepared)
                finally:
                    # Even all-hit/error paths join any queued writes before
                    # recording completion and allowing storage reuse.
                    if deferred:
                        wait_for_weights()
                if self.decode_device_gate and not arena and topk_ids.shape[0] == 1:
                    self.gate_warmed_layers.add(layer_id)
                return result
            except BaseException:
                if not queued and not fast_hit:
                    if plan is not None:
                        # A failed copy/launch must finish before reusing its host buffer.
                        self.metadata_stream.synchronize()
                        self.host_plans[plan.numel()].put(plan)
                    self.capacity.release()
                raise
            finally:
                if queued or fast_hit:
                    computed.record(compute)
                    self.layer_computed[layer_id] = computed
                    launched.set()
                if fast_hit:
                    # D2H completed before inline execution; GPU readers do not
                    # use this pinned plan. Preserve computed for future writers.
                    self.host_plans[plan.numel()].put(plan)
                    self.capacity.release()
