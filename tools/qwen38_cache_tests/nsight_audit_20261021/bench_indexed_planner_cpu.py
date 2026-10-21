#!/usr/bin/env python3
"""Time scan/indexed C++ planner bodies on recorded expert-ID routes (no GPU)."""

import argparse
import ctypes as ct
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from replay_transfer import D, I, Q, ROOT, compile_planners, read_cases

TRACE = (
    ROOT
    / "tools/qwen38_cache_tests/verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz"
)
FIXED = ROOT / "tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay.json"


def make_state(layers, experts, slots, max_slots):
    lookup = (I * (layers * experts))(*([-1] * (layers * experts)))
    heat = (D * (layers * experts))()
    owners = (I * slots)(*([-2] + [-1] * (slots - 1)))
    keys = (I * slots)(*([-1] * slots))
    stats = (Q * 7)()
    index = (I * (2 * layers * experts + 3 * layers + 4 + max_slots))()
    base = 2 * layers * experts + 3 * layers
    index[base + max_slots + 2] = (slots - 1) % max_slots
    index[base + max_slots + 3] = slots - 1
    for slot in range(1, slots):
        index[base + 1 + slot - 1] = slot
    for layer in range(layers):
        index[2 * layers * experts + 2 * layers + layer] = -1
    for offset in range(layers * experts):
        index[layers * experts + offset] = -1
    return lookup, heat, owners, keys, stats, index


def cost_for_call(lib, state, ids, rows, layer, experts, slots, limit):
    if not limit:
        return (D * 0)(), 0
    lookup, heat, owners, keys, _, _ = state
    copies = []
    for array in (lookup, heat, owners, keys):
        copy = type(array)()
        ct.memmove(copy, array, ct.sizeof(array))
        copies.append(copy)
    gpu, cpu = (I * len(ids))(), (I * len(ids))()
    plan, stats = (I * (4 + 2 * len(ids)))(), (Q * 7)()
    lib.hybrid_plan_kernel(
        ids,
        *copies,
        (D * 0)(),
        gpu,
        cpu,
        plan,
        stats,
        rows,
        len(ids),
        experts,
        slots,
        0,
        layer,
        0.95,
        1.9,
    )
    admitted = stats[3]
    rejected = [plan[4 + 2 * i] for i in range(admitted, plan[1])]
    selected = min(limit, len(rejected))
    if not selected:
        return (D * 0)(), 0
    counts = Counter(e for e in ids if e >= 0)
    profile = [counts[e] for e in rejected]
    cpu_profile = profile[:selected]
    cpu_routes = sum(cpu_profile)
    row = (
        [
            rows,
            sum(counts.values()) - cpu_routes,
            cpu_routes,
            selected,
            len(rejected),
            1,
        ]
        + cpu_profile
        + [0] * (32 - selected)
        + profile
        + [0] * (32 - len(profile))
    )
    return (D * 70)(*row), 1


def run(
    lib, sweeps, indexed, cpu_limit, layers, experts, slots, max_slots, verify=False
):
    state = make_state(layers, experts, slots, max_slots)
    lookup, heat, owners, keys, stats, index = state
    signature = []
    elapsed_ns = 0
    for events in sweeps:
        for layer, rows in enumerate(events):
            flat = [e for row in rows for e in row]
            ids = (I * len(flat))(*flat)
            costs, samples = cost_for_call(
                lib, state, ids, len(rows), layer, experts, slots, cpu_limit
            )
            gpu, cpu = (I * len(flat))(), (I * len(flat))()
            plan = (I * (4 + 2 * len(flat)))()
            call_args = (
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
                slots,
                samples,
                layer,
                0.95,
                1.9,
            )
            start = time.perf_counter_ns()
            if indexed:
                lib.hybrid_plan_indexed_kernel(*call_args, index, layers, max_slots)
            else:
                lib.hybrid_plan_kernel(*call_args)
            elapsed_ns += time.perf_counter_ns() - start
            if verify:
                signature.append(
                    (
                        tuple(stats),
                        tuple(e for e in cpu if e >= 0),
                        tuple((owners[s], keys[s]) for s in gpu if s >= 0),
                        tuple(plan[4 + 2 * i] for i in range(plan[1])),
                    )
                )
    if verify:
        residents = sorted(
            (owners[s], keys[s], heat[owners[s] * experts + keys[s]])
            for s in range(slots)
            if owners[s] >= 0 and keys[s] >= 0
        )
        signature.append((residents, tuple(heat)))
    return elapsed_ns / 1000, signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slots", type=int, default=6388)
    p.add_argument("--max-slots", type=int, default=7006)
    p.add_argument("--cpu-limit", type=int, default=0)
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()
    if not 1 < args.slots <= args.max_slots or not 0 <= args.cpu_limit <= 32:
        p.error("invalid slots or CPU limit")
    if args.trials < 1:
        p.error("trials must be positive")
    model = json.loads(FIXED.read_text())["geometry"]
    layers, experts = model["num_layers"], model["num_experts"]
    cases = read_cases(TRACE, layers, experts, model["top_k"])
    print("calls:", {k: sum(map(len, v)) for k, v in cases.items()}, flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_planners(Path(tmp))
        for name, sweeps in cases.items():
            old_verified = run(
                lib,
                sweeps,
                False,
                args.cpu_limit,
                layers,
                experts,
                args.slots,
                args.max_slots,
                verify=True,
            )[1]
            new_verified = run(
                lib,
                sweeps,
                True,
                args.cpu_limit,
                layers,
                experts,
                args.slots,
                args.max_slots,
                verify=True,
            )[1]
            for call, (old, new) in enumerate(zip(old_verified, new_verified)):
                if old != new:
                    raise AssertionError(
                        f"{name}: state differs at call {call}: scan={old!r} indexed={new!r}"
                    )
            cpu_calls = sum(bool(result[1]) for result in old_verified[:-1])
            timings = {False: [], True: []}
            for trial in range(args.trials):
                for indexed in ((False, True) if trial % 2 == 0 else (True, False)):
                    elapsed, _ = run(
                        lib,
                        sweeps,
                        indexed,
                        args.cpu_limit,
                        layers,
                        experts,
                        args.slots,
                        args.max_slots,
                    )
                    timings[indexed].append(elapsed)
            old_us, indexed_us = (
                statistics.median(timings[key]) for key in (False, True)
            )
            calls = sum(map(len, sweeps))
            print(
                name,
                "calls",
                calls,
                "cpu_limit",
                args.cpu_limit,
                "cpu_calls",
                cpu_calls,
                "scan_us/call",
                round(old_us / calls, 3),
                "indexed_us/call",
                round(indexed_us / calls, 3),
                "ratio",
                round(indexed_us / old_us, 3),
                flush=True,
            )


if __name__ == "__main__":
    main()
