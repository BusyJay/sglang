"""Real-pack transport-only GPU replay. No real MoE compute is timed."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .core import ROOT, file_hash, read, write, subprocess_environment


def validate_trace(path, meta, suite, manifest_hash):
    if meta["trace_sha256"] != file_hash(path) or meta["manifest_sha256"] != manifest_hash:
        raise ValueError("trace/pack fingerprint mismatch")
    if meta["kind"] not in ("real", "synthetic"):
        raise ValueError("trace kind missing")
    rows = []
    last_token = -1
    for i, line in enumerate(Path(path).read_text().splitlines()):
        event = json.loads(line)
        layer, token, routes = event["layer"], event["token"], event["routes"]
        if type(layer) is not int or not 0 <= layer < suite["num_layers"]:
            raise ValueError("invalid trace layer")
        if type(token) is not int or token < last_token:
            raise ValueError("trace token boundaries must be monotonic")
        if not isinstance(routes, list) or not routes or any(not isinstance(row, list) or len(row) != suite["top_k"] or
                len(set(row)) != len(row) or any(type(e) is not int or not 0 <= e < suite["num_experts"] for e in row) for row in routes):
            raise ValueError("invalid route matrix")
        if layer != i % suite["num_layers"]:
            raise ValueError("trace must have complete ordered layer sweeps")
        if layer == 0 and rows and token != last_token + len(rows[-1]["routes"]):
            raise ValueError("trace token offset gap")
        if layer and (token != rows[-1]["token"] or len(routes) != len(rows[-1]["routes"])):
            raise ValueError("layer/token boundary mismatch")
        rows.append(event)
        last_token = token
    if not rows or len(rows) % suite["num_layers"]:
        raise ValueError("incomplete trace")
    if len(rows) != meta["events"]:
        raise ValueError("trace event count mismatch")
    return rows


def numeric_delta(before, after):
    counters = ("hits", "misses", "transferred_bytes", "h2d_copy_calls", "staging_batches",
                "run_invocations", "cache_invocations", "arena_invocations", "arena_loads")
    return {k: after[k] - before[k] for k in counters if k in before and k in after}


class ReplayProfile:
    """Profile-only instrument. Normal replay never constructs this object."""
    def __init__(self, cuda):
        self.cuda = cuda
        self.marks = []
        self.host_marks = []

    def begin(self, started):
        a, b = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
        host_begin = time.perf_counter() - started
        a.record()
        return a, b, host_begin

    def end(self, mark, event, started):
        a, b, host_begin = mark
        b.record()
        self.marks.append((a, b))
        self.host_marks.append({"layer": event["layer"], "token": event["token"],
                                "dispatch_begin_s": host_begin,
                                "dispatch_end_s": time.perf_counter() - started})

    def results(self, events, layers):
        latencies = [{"layer": e["layer"], "token": e["token"], "rows": len(e["routes"]),
                      "gpu_ready_interval_ms": a.elapsed_time(b)}
                     for e, (a, b) in zip(events, self.marks)]
        per_token = []
        for offset in range(0, len(events), layers):
            group = latencies[offset:offset + layers]
            per_token.append({"token": events[offset]["token"], "rows": len(events[offset]["routes"]),
                              "gpu_ready_sum_ms": sum(e["gpu_ready_interval_ms"] for e in group),
                              "host_dispatch_span_s": self.host_marks[offset + layers - 1]["dispatch_end_s"] - self.host_marks[offset]["dispatch_begin_s"]})
        return latencies, self.host_marks, per_token


def execute_interval(runtime, routes, events, execute, profile=None):
    """Boundary synchronization only; timing instrumentation is opt-in."""
    runtime.synchronize()
    before = runtime.boundary_stats_snapshot()
    started = time.perf_counter()
    for ids, event in zip(routes, events):
        if profile is not None:
            mark = profile.begin(started)
        execute(ids, event)
        if profile is not None:
            profile.end(mark, event, started)
    runtime.synchronize()
    elapsed = time.perf_counter() - started
    after = runtime.boundary_stats_snapshot()
    return elapsed, before, after


def run(args):
    import sys
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (Qwen38ExpertPack, REQUIRED_TENSORS, load_manifest)
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime

    suite, machine = read(args.suite), read(args.machine)
    candidate = next(c for c in suite["candidates"] if c["id"] == args.candidate)
    meta = read(args.trace_meta)
    events = validate_trace(args.trace, meta, suite, file_hash(machine["manifest"]))
    if meta.get("model_revision") != machine["model_revision"]:
        raise ValueError("trace model revision mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this is not a CPU simulation")
    manifest = load_manifest(machine["manifest"])
    if (manifest.num_layers, manifest.num_experts, manifest.top_k) != (suite["num_layers"], suite["num_experts"], suite["top_k"]):
        raise ValueError("real pack dimensions differ from suite")
    pack = Qwen38ExpertPack(manifest)
    loader = dict(candidate["loader"])
    loader["slots_per_layer"] = loader.pop("cache_slots_per_layer")
    loader.pop("runtime_mode")
    loader.pop("async_pack_prefetch", None)
    loader["stats_log_interval"] = 0  # boundary counters replace periodic logging in every replay mode
    runtime = Qwen38ThreadedExpertRuntime(pack, "cuda", **loader)
    evidence = []
    backing = None
    replay_l2 = suite.get("replay_l2", {})
    if args.profile:
        from sglang.srt.layers.moe.qwen38_cache_profile import CacheProfile
        runtime.profiler = CacheProfile(torch.cuda)
    try:
        runtime.reserve()
        if runtime.l2_mode != "off":
            from sglang.srt.utils.cuda_vmm_utils import (VmmReservation, align_up,
                get_device_granularity, make_device_allocation_prop)
            size = replay_l2.get("backing_bytes", 0)
            if type(size) is not int or size <= 0:
                raise ValueError("L2 replay requires explicit replay_l2.backing_bytes")
            device = torch.cuda.current_device()
            granularity = get_device_granularity(device)
            size = align_up(size, granularity)
            backing = VmmReservation(size, make_device_allocation_prop(device), device)
            handle = backing.map(0, size, retain_handle=True)
            runtime.set_elastic_handles([(0, size, handle)])
        # Same explicit prefill seed routes for EVERY candidate, outside timing.
        # Dedicated replay storage is not scheduler KV lending evidence.
        seed = replay_l2.get("seed", [])
        for layer in range(manifest.num_layers):
            experts = [expert for owner, expert in seed if owner == layer]
            if not experts:
                continue
            count = max(runtime.slots_for_layer(layer) + manifest.top_k,
                        len(experts))
            count = (count + manifest.top_k - 1) // manifest.top_k * manifest.top_k
            ids = torch.tensor([experts[i % len(experts)] for i in range(count)],
                               dtype=torch.int32, device="cuda").reshape(-1, manifest.top_k)
            runtime.run(layer, ids, lambda prepared: None)
        runtime.synchronize()
        if runtime.l2_mode == "fixed":
            if set(map(tuple, runtime.l2_fixed_seed)) != runtime.elastic_storage.fixed_seeded:
                raise ValueError("fixed replay seed incomplete; check shared seed/capacity")
        routes = [torch.tensor(e["routes"], dtype=torch.int32, device="cuda") for e in events]
        # Each independent worker starts empty; warmup is an explicit fixed prefix.
        for ids, event in zip(routes[:args.warmup_events], events[:args.warmup_events]):
            runtime.run(event["layer"], ids, lambda prepared: None)
        def execute(ids, event):
            if args.correctness:
                # Inspect every unique routed expert, every payload byte, including alpha.
                active = sorted(set(e for row in event["routes"] for e in row))
                positions = []
                flat = [e for row in event["routes"] for e in row]
                for expert in active:
                    positions.append(flat.index(expert))
                positions_gpu = torch.tensor(positions, dtype=torch.long, device="cuda")
                actual = runtime.run(event["layer"], ids, lambda prepared: {
                    name: prepared.tensors[name][prepared.topk_ids.reshape(-1)[positions_gpu].long()].clone()
                    for name in REQUIRED_TENSORS})
                matches = {}
                for name in REQUIRED_TENSORS:
                    record = manifest.tensors[event["layer"]][name]
                    expected = []
                    for expert in active:
                        view = pack.expert_tensor_bytes(event["layer"], expert, name)
                        try:
                            expected.append(torch.frombuffer(bytearray(view), dtype=getattr(torch, record.dtype)).reshape(record.shape[1:]))
                        finally:
                            view.release()
                    matches[name] = torch.equal(actual[name].cpu().contiguous().view(torch.uint8), torch.stack(expected).contiguous().view(torch.uint8))
                evidence.append({"layer": event["layer"], "token": event["token"], "experts": active, "matches": matches})
                if not all(matches.values()):
                    raise RuntimeError(f"reference payload mismatch: {evidence[-1]}")
            else:
                runtime.run(event["layer"], ids, lambda prepared: None)
        profiler = ReplayProfile(torch.cuda) if args.profile else None
        elapsed, before, after = execute_interval(runtime, routes, events, execute, profiler)
        latencies, host_marks, per_token = profiler.results(events, suite["num_layers"]) if profiler else ([], [], [])
        mode = "correctness" if args.correctness else "profile" if args.profile else "benchmark"
        report = {"per_token": per_token, "status": "ok", "version": suite["version"], "mode": mode,
                  "scope": "transport-only; real pack sizes/layout; no MoE compute, no model speedup inference",
                  "trace_kind": meta["kind"], "trace_sha256": file_hash(args.trace), "trace_meta": meta,
                  "wall_s": elapsed if mode == "benchmark" else None, "gpu_intervals": latencies,
                  "host_dispatch_intervals": host_marks,
                  "measurement": "benchmark: wall boundaries only; profile: per-layer timing events and host marks; correctness/profile excluded from speed comparisons",
                  "diagnostic_wall_s": elapsed if mode == "profile" else None,
                  "effective_child_env": subprocess_environment(),
                  "measured_event_range": [0, len(events)],
                  "warmup_policy": "repeat trace prefix, then measure entire trace from beginning",
                  "initial_cache": "empty before explicit L2 pack seed and shared prefill seed",
                  "l2_backing": {"kind": "dedicated-VMM; not borrowed KV", "bytes": backing.size if backing else 0,
                                 "shared_prefill_seed": seed},
                  "runtime_profile": runtime.profiler.results() if runtime.profiler else None, "warmup_events": args.warmup_events,
                  "before": before, "after": after, "counter_delta": numeric_delta(before, after),
                  "correctness": {"status": "passed" if args.correctness else "not_run",
                                  "scope": "all routed experts, all six payload tensors, compared byte-exact to real mmap pack; real MoE not executed",
                                  "evidence": evidence},
                  "actual_runtime_source": str(Path(__import__('inspect').getfile(Qwen38ThreadedExpertRuntime)).resolve()),
                  "effective_runtime": {"decode_submit_experts": runtime.decode_submit_experts, "packed_decode": runtime.packed_decode, "packed_prefill": runtime.packed_prefill, "packed_attached": runtime.packed_transport is not None, "l2_mode": runtime.l2_mode, "l2_attached": runtime.elastic_storage is not None, "stats_log_interval": runtime.stats_log_interval, "layer_cache_slots": runtime.layer_cache_slots, "adaptive_cache": runtime.adaptive_cache,
                                        "resident_alphas": runtime.resident_alphas, "fused_decode_plan": runtime.fused_decode_plan,
                                        "prefill_copy_mode": runtime.prefill_copy_mode, "cache_policy": runtime.cache_policy},
                  "candidate": args.candidate, "payload_nbytes": manifest.payload_nbytes,
                  "token_wall_s": elapsed / sum(len(e["routes"]) for e in events if e["layer"] == 0) if mode == "benchmark" else None}
        write(args.output, report)
    finally:
        try:
            runtime.close()
        finally:
            # Release physical owner only after all aliases were removed.
            if backing is not None and (runtime.elastic_storage is None or not runtime.elastic_storage.blocks):
                backing.close()
        # Runtime owns mmap views; child process exit releases the read-only pack.
