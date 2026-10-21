#!/usr/bin/env python3
"""Global GPU vs CPU-subset transfer sensitivity, never timing calibration."""

import argparse
import ctypes as ct
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from replay_transfer import GLOBAL, HEAP, ROOT, D, I, Q, compile_planners, read_cases

DEVICE = ROOT / "python/sglang/srt/layers/moe/qwen38_hybrid_device.py"
INDEX_BUILDER = ROOT / "python/sglang/srt/layers/moe/qwen38_heap_index.py"
B12X = ROOT / "python/sglang/srt/layers/quantization/qwen38_cached_b12x.py"
B12X_HYBRID = ROOT / "python/sglang/srt/layers/quantization/qwen38_b12x_hybrid.py"
MAILBOX = ROOT / "python/sglang/kernels/jit/csrc/moe/qwen38_mailbox.cc"
PLAN_WRAPPER = ROOT / "python/sglang/kernels/ops/moe/qwen38_hybrid_plan.py"


def clone(array):
    copy = type(array)()
    ct.memmove(copy, array, ct.sizeof(array))
    return copy


def replay(lib, sweeps, budget, model, warmup, cpu_limit, cases=None, case_name=None):
    layers, experts, hidden = (
        model["num_layers"],
        model["num_experts"],
        model["hidden_size"],
    )
    top_k = model["top_k"]
    lookup = (I * (layers * experts))(*([-1] * (layers * experts)))
    heat = (D * (layers * experts))()
    owners = (I * budget)(*([-2] + [-1] * (budget - 1)))
    keys = (I * budget)(*([-1] * budget))
    stats = (Q * 7)()
    metrics = Counter()
    gpu_only_streak = 0
    for step, events in enumerate(sweeps):
        for layer, rows in enumerate(events):
            flat = sum(rows, [])
            counts = Counter(e for e in flat if e >= 0)
            ids = (I * len(flat))(*flat)
            gpu = (I * len(flat))()
            cpu = (I * len(flat))()
            plan = (I * (4 + 2 * len(flat)))()
            costs = (D * 0)()
            samples = 0
            rejected = []
            admitted = []
            if cpu_limit:
                # Probe a copy of current state solely to recover this invocation's
                # rejected profile, then execute the real decision once on original.
                probe_stats = (Q * 7)()
                probe_state = [clone(a) for a in (lookup, heat, owners, keys)]
                lib.hybrid_plan_kernel(
                    ids,
                    *probe_state,
                    costs,
                    gpu,
                    cpu,
                    plan,
                    probe_stats,
                    len(rows),
                    len(flat),
                    experts,
                    budget,
                    0,
                    layer,
                    0.95,
                    1.9,
                )
                admitted = [plan[4 + 2 * n] for n in range(probe_stats[3])]
                rejected = [plan[4 + 2 * n] for n in range(probe_stats[3], plan[1])]
                selected = min(cpu_limit, len(rejected))
                if selected:
                    profile = [counts[e] for e in rejected]
                    cpu_profile = profile[:selected]
                    cpu_routes = sum(cpu_profile)
                    # Diagnostic selector input: 1 is only a rank sentinel, NOT a
                    # measured elapsed time. Never export it as calibration.
                    row = (
                        [
                            len(rows),
                            sum(counts.values()) - cpu_routes,
                            cpu_routes,
                            selected,
                            len(rejected),
                            1,
                        ]
                        + cpu_profile
                        + [0] * (32 - selected)
                        + profile
                        + [0] * (32 - len(rejected))
                    )
                    costs = (D * 70)(*row)
                    samples = 1
            previous = list(stats)
            lib.hybrid_plan_kernel(
                ids,
                lookup,
                heat,
                owners,
                keys,
                costs,
                gpu,
                cpu,
                plan,
                stats,
                len(rows),
                len(flat),
                experts,
                budget,
                samples,
                layer,
                0.95,
                1.9,
            )
            selected_experts = {e for e in cpu if e >= 0}
            assert selected_experts.isdisjoint({keys[s] for s in gpu if s >= 0})
            if step < warmup:
                continue
            if cases is not None and selected_experts:
                cpu_profile = tuple(
                    counts[e] for e in rejected[: len(selected_experts)]
                )
                key = (
                    len(rows),
                    sum(counts.values()) - sum(cpu_profile),
                    sum(cpu_profile),
                    len(selected_experts),
                    cpu_profile,
                    tuple(counts[e] for e in rejected),
                )
                variant = (
                    key,
                    len(admitted),
                    len(rejected) - len(selected_experts),
                )
                record = cases.get(variant)
                if record is None:
                    record = cases[variant] = dict(
                        key=key,
                        rows=rows,
                        layer=layer,
                        cpu_experts=rejected[: len(selected_experts)],
                        rejected_experts=rejected,
                        admitted_experts=admitted,
                        case=case_name,
                        count=0,
                    )
                record["count"] += 1
            metrics.update(
                calls=1,
                unique_accesses=len(counts),
                gpu_fills=plan[1],
                lfu_admitted=stats[3] - previous[3],
                forced_placements=stats[6] - previous[6],
                cpu_expert_calls=len(selected_experts),
                cpu_routes=sum(e >= 0 for e in cpu),
                cpu_calls=bool(selected_experts),
                gpu_only_calls=not selected_experts,
            )
            if selected_experts:
                metrics["cpu_after_gpu_only_calls"] += bool(gpu_only_streak)
                gpu_only_streak = 0
            else:
                gpu_only_streak += 1
                metrics["max_gpu_only_streak"] = max(
                    metrics["max_gpu_only_streak"], gpu_only_streak
                )
            if selected_experts:
                # hybrid_stage writes only on selected routes, with no route
                # readback or per-tensor cudaMemcpy. The worker reads these
                # bytes in place; hybrid_result reads the CPU result in place.
                metrics["mapped_header_write_bytes"] += 12
                metrics["mapped_activation_write_bytes"] += len(rows) * hidden * 2
                metrics["mapped_route_ids_write_bytes"] += len(rows) * top_k * 4
                metrics["mapped_route_weights_write_bytes"] += len(flat) * 4
                metrics["mapped_result_read_bytes"] += len(rows) * hidden * 4
                metrics["mapped_completion_write_bytes"] += 4
                metrics["mapped_completion_read_bytes_min"] += 4
                metrics["selected_tokens"] += len(rows)
    return dict(cpu_limit=cpu_limit, metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixed", type=Path, default=Path(__file__).with_name("transfer-replay.json")
    )
    p.add_argument("--cpu-limits", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--budgets", type=int, nargs="+", default=[6388, 6390, 6643, 7006])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases-output", type=Path)
    p.add_argument("--max-cases", type=int, default=100)
    args = p.parse_args()
    args.fixed = args.fixed.resolve()
    if any(not 0 <= limit <= 32 for limit in args.cpu_limits):
        raise ValueError("CPU prefix limit must be in [0, 32]")
    fixed = json.loads(args.fixed.read_text())
    m = fixed["geometry"]
    trace = (
        ROOT
        / "tools/qwen38_cache_tests/verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz"
    )
    cases = read_cases(trace, m["num_layers"], m["num_experts"], m["top_k"])
    reports = []
    measured_cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_planners(Path(tmp))
        for budget in args.budgets:
            for case, sweeps in cases.items():
                reference = next(
                    r
                    for r in fixed["reports"]
                    if r["budget"] == budget and r["case"] == case
                )
                configs = []
                for limit in args.cpu_limits:
                    r = replay(
                        lib,
                        sweeps,
                        budget,
                        m,
                        reference["warmup_sweeps"],
                        limit,
                        (
                            measured_cases
                            if args.cases_output and budget == args.budgets[0]
                            else None
                        ),
                        case,
                    )
                    metrics = r["metrics"]
                    metrics["expert_h2d_bytes"] = (
                        metrics["gpu_fills"] * fixed["expert_payload_bytes"]
                    )
                    for k in (
                        "mapped_header_write_bytes",
                        "mapped_activation_write_bytes",
                        "mapped_route_ids_write_bytes",
                        "mapped_route_weights_write_bytes",
                        "mapped_result_read_bytes",
                        "mapped_completion_write_bytes",
                        "mapped_completion_read_bytes_min",
                    ):
                        metrics.setdefault(k, 0)
                    metrics["mapped_request_write_bytes"] = sum(
                        metrics[k]
                        for k in (
                            "mapped_header_write_bytes",
                            "mapped_activation_write_bytes",
                            "mapped_route_ids_write_bytes",
                            "mapped_route_weights_write_bytes",
                        )
                    )
                    metrics["mapped_total_bytes"] = (
                        metrics["mapped_request_write_bytes"]
                        + metrics["mapped_result_read_bytes"]
                        + metrics["mapped_completion_write_bytes"]
                        + metrics["mapped_completion_read_bytes_min"]
                    )
                    assert (
                        metrics["cpu_calls"] + metrics["gpu_only_calls"]
                        == metrics["calls"]
                    )
                    if limit == 0:
                        assert metrics["mapped_total_bytes"] == 0
                    metrics["cpu_expert_weight_bytes_read"] = (
                        metrics["cpu_expert_calls"] * fixed["expert_payload_bytes"]
                    )
                    if limit == 0:
                        assert (
                            metrics["gpu_fills"]
                            == reference["results"]["global_gpu"]["gpu_fills"]
                        )
                    configs.append(r)
                    print(
                        budget,
                        case,
                        limit,
                        metrics["gpu_fills"],
                        metrics["cpu_calls"],
                        flush=True,
                    )
                reports.append(dict(budget=budget, case=case, configs=configs))
    output = dict(
        status="diagnostic_cpu_subset_mmap_sensitivity",
        production_calibration_available=False,
        production_empty_table_behavior="Same expert placement and transfers as global GPU-only; zero selected CPU experts and zero mapped request/result payload.",
        expert_payload_bytes=fixed["expert_payload_bytes"],
        reports=reports,
        provenance={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                GLOBAL,
                HEAP,
                DEVICE,
                INDEX_BUILDER,
                B12X,
                B12X_HYBRID,
                MAILBOX,
                PLAN_WRAPPER,
                trace,
                args.fixed,
            )
        },
        limits=[
            "CPU subsets are hypothetical fixed prefix limits of LFU-rejected experts, not benchmark-validated choices.",
            "Diagnostic rank sentinel only exercises actual selector branch; it is not elapsed time or a production calibration file.",
            "Each mode replays its own cache evolution: CPU experts stay absent, so later misses and forced evictions can differ.",
            "CPU weight bytes are logical native RAM reads, not PCIe or measured DRAM traffic.",
            "Mapped request bytes count 12-byte header, BF16 activation, all route IDs and FP32 route weights only for CPU-selected calls; mapped GPU result reads use FP32.",
            "Mapped byte counts are logical registered-host accesses, not measured PCIe transactions or memcpy traffic; completion reads count one minimum 4-byte poll, not repeated per-thread polls.",
            "No CUDA, no CPU native execution, no throughput or latency prediction; route-only fixed-budget target verify replay.",
        ],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.cases_output:

        def workload_key(record):
            (
                tokens,
                gpu_routes,
                cpu_routes,
                cpu_experts,
                cpu_profile,
                rejected_profile,
            ) = record["key"]
            return (
                tokens,
                gpu_routes,
                cpu_routes,
                cpu_experts,
                tuple(sorted(cpu_profile)),
                tuple(sorted(rejected_profile)),
            )

        variants = Counter(workload_key(record) for record in measured_cases.values())
        popularity = Counter()
        for record in measured_cases.values():
            popularity[workload_key(record)] += record["count"]
        chosen_keys = set()
        remaining = args.max_cases
        for key, _ in popularity.most_common():
            if variants[key] > remaining:
                continue
            chosen_keys.add(key)
            remaining -= variants[key]
        selected_cases = sorted(
            (
                record
                for record in measured_cases.values()
                if workload_key(record) in chosen_keys
            ),
            key=lambda row: row["count"],
            reverse=True,
        )
        complete_keys = [
            list(key[:4]) + [list(key[4]), list(key[5])] for key in sorted(chosen_keys)
        ]
        args.cases_output.write_text(
            json.dumps(
                dict(
                    geometry=m,
                    budget=args.budgets[0],
                    cases=selected_cases,
                    complete_keys=complete_keys,
                    provenance=output["provenance"],
                    note="Diagnostic LFU-rejected prefixes; GPU measurement required before use as calibration.",
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
