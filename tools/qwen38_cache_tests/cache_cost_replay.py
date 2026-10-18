#!/usr/bin/env python3
"""Offline cache-policy/slot-allocation comparison using measured fill costs.

No GPU execution or production policy change. Summed isolated fill latency is
a surrogate, not service latency or predicted token throughput.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from ._load_source import load_source
from .replay_cache import allocate

policy_module = load_source(
    "qwen38_cost_policy", "python/sglang/srt/layers/moe/qwen38_expert_cache.py"
)


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def read_sweeps(path, layers, experts, case=None):
    """Reject partial sweeps and gaps; do not concatenate different requests."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sweeps, pending, current_case = [], [], None
    with opener(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if "event" in row:
                if pending:
                    raise ValueError("request boundary inside layer sweep")
                if row["event"] == "start":
                    current_case = row["case"]
                elif row["event"] == "end":
                    current_case = None
                continue
            if case is not None and current_case != case:
                continue
            if row.get("phase", "verify") != "verify":
                raise ValueError(
                    "only verify tape supported; do not silently drop phases"
                )
            if "sweep" in row and row["sweep"] != len(sweeps):
                raise ValueError("sweep index gap or duplicate")
            if row["layer"] != len(pending):
                raise ValueError("trace must contain ordered complete layer sweeps")
            routes = row["routes"]
            if (
                not routes
                or len(routes) > 3
                or any(not isinstance(r, list) or not r for r in routes)
            ):
                raise ValueError("only 1..3-row decode/verify routes supported")
            if any(
                type(e) is not int or not -1 <= e < experts for r in routes for e in r
            ):
                raise ValueError("invalid expert ID")
            active = sorted({e for r in routes for e in r if e >= 0})
            if pending and len(routes) != pending[0][1]:
                raise ValueError("different verify shapes in one sweep")
            pending.append((active, len(routes), current_case))
            if len(pending) == layers:
                sweeps.append(pending)
                pending = []
    if pending or not sweeps:
        raise ValueError("empty trace or partial final sweep")
    if len({s[0][2] for s in sweeps}) != 1:
        raise ValueError(
            "select one request with --case; no cross-request concatenation"
        )
    return sweeps


class FillCost:
    def __init__(self, samples, variant, metric):
        buckets = {}
        for row in samples["reports"]:
            if row["variant"] == variant:
                buckets.setdefault(row["misses"], []).append(row[metric])
        if not buckets or 0 not in buckets:
            raise ValueError("cost samples must include zero misses")
        self.values = {n: statistics.median(v) for n, v in buckets.items()}
        if any(type(n) is not int or n < 0 for n in self.values):
            raise ValueError("invalid measured miss count")
        if any(not math.isfinite(v) or v < 0 for v in self.values.values()):
            raise ValueError("invalid fill cost")
        self.interpolated = set()

    def __call__(self, misses):
        if misses in self.values:
            return self.values[misses]
        lower = max((n for n in self.values if n < misses), default=None)
        upper = min((n for n in self.values if n > misses), default=None)
        if lower is None or upper is None:
            raise ValueError("miss count outside measured range; no extrapolation")
        self.interpolated.add(misses)
        return self.values[lower] + (self.values[upper] - self.values[lower]) * (
            misses - lower
        ) / (upper - lower)


def replay_layer(events, capacity, experts, name, fraction, cost, score_from, score_to):
    cache = policy_module.LayerExpertCachePolicy(
        experts,
        capacity,
        policy="lfu" if name == "lfu" else "slru",
        protected_fraction=0 if name == "lru" else fraction,
    )
    misses, accesses, all_hit, latency, runs = 0, 0, 0, 0.0, 0
    histogram = Counter()
    for index, (active, _, _) in enumerate(events[:score_to]):
        if len(active) > capacity:
            raise ValueError("capacity smaller than active set; arena not modeled")
        mapping, missing = cache.ensure(active)
        if index < score_from:
            continue
        n = len(missing)
        misses += n
        accesses += len(active)
        all_hit += n == 0
        latency += cost(n)
        histogram[n] += 1
        pairs = sorted((e, mapping[e]) for e in missing)
        runs += sum(
            i == 0 or e != pairs[i - 1][0] + 1 or slot != pairs[i - 1][1] + 1
            for i, (e, slot) in enumerate(pairs)
        )
    return dict(
        misses=misses,
        accesses=accesses,
        all_hit_calls=all_hit,
        estimated_fill_ms=latency,
        coalescible_runs=runs,
        miss_histogram=dict(sorted(histogram.items())),
    )


def aggregate(rows):
    keys = (
        "misses",
        "accesses",
        "all_hit_calls",
        "estimated_fill_ms",
        "coalescible_runs",
    )
    result = {k: sum(row[k] for row in rows) for k in keys}
    histogram = Counter()
    for row in rows:
        histogram.update(row["miss_histogram"])
    result["miss_histogram"] = dict(sorted(histogram.items()))
    result["hit_rate"] = 1 - result["misses"] / max(1, result["accesses"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--case")
    p.add_argument("--fill-samples", required=True, type=Path)
    p.add_argument("--variant", default="mapped_cta_8")
    p.add_argument(
        "--metric",
        choices=["median_gpu_ms", "wall_ms_per_call"],
        default="median_gpu_ms",
    )
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--budget", type=int, default=6390)
    p.add_argument(
        "--capacities", nargs="+", type=int, default=[96, 112, 128, 144, 160, 192, 256]
    )
    p.add_argument("--warmup", type=int, default=128)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if (
        args.layers < 1
        or args.experts < 1
        or not args.layers <= args.budget <= args.layers * args.experts
        or not 0 < args.train_fraction < 1
    ):
        p.error("invalid geometry, budget or split")
    sweeps = read_sweeps(args.trace, args.layers, args.experts, args.case)
    split = int(len(sweeps) * args.train_fraction)
    if not 0 <= args.warmup < split < len(sweeps):
        p.error("insufficient sweeps for warmup/train/test split")
    initial = [
        args.budget // args.layers + (i < args.budget % args.layers)
        for i in range(args.layers)
    ]
    capacities = sorted(set(args.capacities) | set(initial))
    if min(capacities) < 1 or max(capacities) > args.experts:
        p.error("invalid capacity grid")
    events = [[s[layer] for s in sweeps] for layer in range(args.layers)]
    cost = FillCost(read_json(args.fill_samples), args.variant, args.metric)
    reports = []
    for name, fraction in [
        ("lfu", 0.7),
        ("lru", 0),
        ("slru", 0.5),
        ("slru", 0.7),
        ("slru", 0.8),
    ]:
        train = [
            {
                c: replay_layer(
                    e, c, args.experts, name, fraction, cost, args.warmup, split
                )
                for c in capacities
            }
            for e in events
        ]
        layouts = {"fixed": initial}
        for objective, field in [
            ("min_misses", "misses"),
            ("min_fill_ms", "estimated_fill_ms"),
        ]:
            chosen = allocate(
                [{c: r[field] for c, r in curve.items()} for curve in train],
                args.budget,
            )
            # Use sampled capacities only; unused slots explicitly reported.
            layouts[objective] = chosen["layer_cache_slots"]
        results = []
        for objective, layout in layouts.items():
            test = [
                replay_layer(
                    e, c, args.experts, name, fraction, cost, split, len(sweeps)
                )
                for e, c in zip(events, layout)
            ]
            fit = [curve[c] for curve, c in zip(train, layout)]
            results.append(
                dict(
                    objective=objective,
                    layout=layout,
                    used_slots=sum(layout),
                    unused_slots=args.budget - sum(layout),
                    train=aggregate(fit),
                    held_out=aggregate(test),
                    held_out_layers=test,
                )
            )
        reports.append(dict(policy=name, protected_fraction=fraction, results=results))
        print(
            name,
            fraction,
            [
                (
                    x["objective"],
                    x["held_out"]["misses"],
                    round(x["held_out"]["estimated_fill_ms"], 2),
                )
                for x in results
            ],
            flush=True,
        )
    samples = read_json(args.fill_samples)
    output = dict(
        status="completed_offline_estimate",
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        fill_sha256=hashlib.sha256(args.fill_samples.read_bytes()).hexdigest(),
        trace=str(args.trace),
        case=args.case,
        sweeps=len(sweeps),
        warmup_sweeps=args.warmup,
        train_end=split,
        test_sweeps=len(sweeps) - split,
        budget=args.budget,
        capacities=capacities,
        initial_layout=initial,
        cost_variant=args.variant,
        cost_metric=args.metric,
        measured_cost_ms=cost.values,
        interpolated_miss_counts=sorted(cost.interpolated),
        calibration_gpu=samples.get("gpu"),
        calibration_sizes=samples.get("sizes"),
        reports=reports,
        limits=[
            "Summed isolated fill latency surrogate; no service speedup prediction",
            "Cost is pooled across synthetic address plans; not conditioned on layer, shape, locality or concurrency",
            "LRU uses SLRU with zero protected fraction; active experts cannot be evicted",
            "Warm cache state is obtained by replaying the same prefix from empty for each layout, not restored production state",
            "Validation replays full training prefix at selected capacity; migration cost excluded",
            "Whole verify active sets only; draft/prefill/intervening unrecorded work excluded",
            "No latency-aware eviction heuristic is implemented; only policy comparison and capacity-objective comparison",
            "Source/target run counts reported diagnostically; mapped kernel still copies per expert",
            "Capacity grid can leave budget unused; no fabricated residual-slot gains",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
