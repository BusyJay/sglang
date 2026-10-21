#!/usr/bin/env python3
"""Causal adaptive non-hybrid baseline using the production analysis worker."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from replay_transfer import ROOT, F, I, Q, compile_planners, read_cases

WORKER = ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
ADAPTIVE = ROOT / "python/sglang/srt/layers/moe/qwen38_cache_adaptive.py"


def load_adaptive():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(
        "sglang.srt.layers.moe.qwen38_expert_cache",
        ADAPTIVE.with_name("qwen38_expert_cache.py"),
    )
    return load("transfer_adaptive", ADAPTIVE)


def state_for(slots, experts):
    return (
        (I * experts)(*([-1] * experts)),
        (I * slots)(*([-1] * slots)),
        (F * slots)(),
        (Q * slots)(),
        (Q * slots)(),
        (Q * 1)(),
        (I * (4 + 4 * slots))(),
    )


def migrate(states, old_layout, target, experts, relocate):
    old_offsets, new_offsets = [], []
    for layout, offsets in [(old_layout, old_offsets), (target, new_offsets)]:
        cursor = 0
        for slots in layout:
            offsets.append(cursor)
            cursor += slots
    moves, new_states = {}, []
    evicted = 0
    for layer, (state, slots) in enumerate(zip(states, target)):
        residents = [s for s, e in enumerate(state[1]) if e >= 0]
        residents.sort(key=lambda s: (-state[2][s], -state[3][s]))
        selected = residents[:slots]
        evicted += len(residents) - len(selected)
        new = state_for(slots, experts)
        for dst, src in enumerate(selected):
            e = state[1][src]
            new[4][dst] = 1
            new[0][e], new[1][dst], new[2][dst], new[3][dst] = (
                dst,
                e,
                state[2][src],
                state[3][src],
            )
            moves[old_offsets[layer] + src] = new_offsets[layer] + dst
        new[5][0] = max(new[3], default=0)
        new_states.append(new)
    transfers = Counter()
    values = {
        old_offsets[l] + s: (l, e)
        for l, state in enumerate(states)
        for s, e in enumerate(state[1])
        if e >= 0
    }
    original = values.copy()
    scratch = [None]

    def move(src, dst):
        values[dst] = values[src]
        transfers["slot_copies"] += 1

    def save(src):
        scratch[0] = values[src]
        transfers["scratch_saves"] += 1
        transfers["slot_copies"] += 1

    def restore(dst):
        values[dst] = scratch[0]
        transfers["slot_copies"] += 1

    relocate(moves, move, save, restore)
    assert all(values[dst] == original[src] for src, dst in moves.items())
    return new_states, dict(transfers, evicted=evicted, retained=len(moves))


def replay(
    lib,
    worker,
    relocate,
    sweeps,
    budget,
    geometry,
    warmup,
    window,
    interval,
    candidate_max,
):
    layers, experts, topk = geometry
    layout = [budget // layers + (l < budget % layers) for l in range(layers)]
    states = [state_for(s, experts) for s in layout]
    metrics = Counter()
    events = []
    sample = [[] for _ in range(layers)]
    for step, rows in enumerate(sweeps):
        for layer, routes in enumerate(rows):
            counts = Counter(e for row in routes for e in row if e >= 0)
            histogram = (I * (experts + 1))(*[counts[e] for e in range(experts)], 0)
            lib.apply_plan(
                histogram,
                *states[layer],
                experts,
                layout[layer],
                0.95,
                0,
                0,
                False,
                None,
                -1,
            )
            if states[layer][-1][0] or states[layer][-1][3]:
                raise ValueError("invalid baseline plan")
            if step >= warmup:
                metrics.update(
                    calls=1, unique_accesses=len(counts), gpu_fills=states[layer][-1][1]
                )
            if step % (window + interval) < window:
                sample[layer].append(sorted(counts))
        if step % (window + interval) == window - 1:
            request = dict(
                routes=sample,
                current=layout,
                num_experts=experts,
                top_k=topk,
                min_slots=3 * topk,
                budget=budget,
                policy="lfu",
                protected_fraction=0.7,
                candidate_max=candidate_max,
            )
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            reply = json.loads(worker.stdout.readline())
            sample = [[] for _ in range(layers)]
            event = dict(
                after_sweep=step + 1,
                evaluation=reply["evaluation"],
                target=reply["target"],
                analysis_host_seconds=reply["seconds"],
            )
            if reply["target"] is not None:
                states, moved = migrate(
                    states, layout, reply["target"], experts, relocate
                )
                layout = reply["target"]
                event["migration"] = moved
                if step >= warmup:
                    metrics.update(
                        rebalances=1,
                        migration_slot_copies=moved.get("slot_copies", 0),
                        migration_evictions=moved["evicted"],
                    )
            events.append(event)
    for key in ("migration_slot_copies", "migration_evictions", "rebalances"):
        metrics.setdefault(key, 0)
    metrics["hit_fraction"] = 1 - metrics["gpu_fills"] / metrics["unique_accesses"]
    return dict(
        window=window,
        interval=interval,
        candidate_max=candidate_max,
        metrics=metrics,
        events=events,
        final_layout=layout,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixed", type=Path, default=Path(__file__).with_name("transfer-replay.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--budgets", nargs="+", type=int, default=[6388, 6390, 6643, 7006])
    p.add_argument("--candidate-max", type=int, default=256)
    args = p.parse_args()
    fixed = json.loads(args.fixed.read_text())
    geometry = tuple(
        fixed["geometry"][k] for k in ["num_layers", "num_experts", "top_k"]
    )
    trace = (
        ROOT
        / "tools/qwen38_cache_tests/verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz"
    )
    if (
        min(args.windows) < 16
        or not max(
            geometry[2] * 3,
            max((b + geometry[0] - 1) // geometry[0] for b in args.budgets),
        )
        <= args.candidate_max
        <= geometry[1]
    ):
        raise ValueError("invalid observation window or candidate maximum")
    cases = read_cases(trace, *geometry)
    adaptive = load_adaptive()
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = compile_planners(Path(tmp))
        worker = subprocess.Popen(
            [sys.executable, str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            for budget in args.budgets:
                for case, sweeps in cases.items():
                    reference = next(
                        r
                        for r in fixed["reports"]
                        if r["budget"] == budget and r["case"] == case
                    )
                    configs = []
                    for window in args.windows:
                        interval = 4096 if window == 512 else 0
                        r = replay(
                            lib,
                            worker,
                            adaptive.relocate_slots,
                            sweeps,
                            budget,
                            geometry,
                            reference["warmup_sweeps"],
                            window,
                            interval,
                            args.candidate_max,
                        )
                        m = r["metrics"]
                        m["expert_h2d_bytes"] = (
                            m["gpu_fills"] * fixed["expert_payload_bytes"]
                        )
                        m["migration_d2d_bytes"] = (
                            m["migration_slot_copies"] * fixed["expert_payload_bytes"]
                        )
                        m["total_payload_bytes"] = (
                            m["expert_h2d_bytes"] + m["migration_d2d_bytes"]
                        )
                        configs.append(r)
                        print(
                            budget,
                            case,
                            window,
                            m["gpu_fills"],
                            m["migration_slot_copies"],
                            len(r["events"]),
                            flush=True,
                        )
                    reports.append(
                        dict(
                            budget=budget,
                            case=case,
                            configs=configs,
                            global_gpu=reference["results"]["global_gpu"],
                            fixed_per_layer_gpu=reference["results"]["per_layer_gpu"],
                        )
                    )
        finally:
            worker.stdin.close()
            worker.wait()
            worker.stdout.close()
    output = dict(
        status="causal_adaptive_host_replay",
        expert_payload_bytes=fixed["expert_payload_bytes"],
        reports=reports,
        provenance={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                WORKER,
                ADAPTIVE,
                ADAPTIVE.with_name("qwen38_expert_cache.py"),
                trace,
                args.fixed,
            ]
        },
        limits=[
            "Production analysis worker and allocation/migration rules; no future routes supplied to a decision.",
            "Window512 interval4096 is default; shorter windows interval0 are tuned causal baselines, decisions applied immediately after observed sweep.",
            "Actual async analysis/application latency is omitted, favoring baseline; recorded host analysis time is not serving timing.",
            "Best baseline configuration may be selected after replay and is explicitly a comparison envelope, not a guaranteed online policy.",
            "Fixed budget, separate request cold starts: initial production cache, prefill and draft are absent from trace.",
            "H2D and D2D are reported separately; summed payload is not a latency or bandwidth model.",
        ],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
