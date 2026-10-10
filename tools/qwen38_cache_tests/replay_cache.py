#!/usr/bin/env python3
"""Offline decode-route replay and fixed-budget capacity allocation; no torch."""

import argparse
import json
from pathlib import Path

from ._load_source import load_source

policy = load_source(
    "qwen38_replay_policy",
    "python/sglang/srt/layers/moe/qwen38_expert_cache.py",
)


def read_trace(path, num_layers, num_experts, max_tokens=1):
    layers = [[] for _ in range(num_layers)]
    skipped = 0
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        layer, rows = record["layer"], record["routes"]
        if type(layer) is not int or not 0 <= layer < num_layers:
            raise ValueError("invalid trace layer")
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("routes must be a matrix")
        ids = [expert for row in rows for expert in row]
        if any(type(e) is not int or not -1 <= e < num_experts for e in ids):
            raise ValueError("invalid trace expert")
        if len(rows) > max_tokens:
            skipped += 1
            continue
        layers[layer].append(sorted(set(e for e in ids if e >= 0)))
    if any(not layer for layer in layers):
        raise ValueError("trace must contain decode samples for every layer")
    return layers, skipped


def miss_curve(routes, capacities, num_experts):
    result = {}
    for capacity in capacities:
        cache = policy.LayerExpertCachePolicy(num_experts, capacity)
        misses = 0
        for active in routes:
            if len(active) > capacity:
                raise ValueError("capacity cannot hold a traced active set")
            _, missing = cache.ensure(active)
            misses += len(missing)
        result[capacity] = misses
    return result


def allocate(curves, budget):
    """Exact minimum replay misses over candidate capacities under slot budget.

    Unlike marginal-gain greedy, this handles non-concave LFU miss curves.
    Assumes equal bytes per slot across layers (the current runtime layout).
    """
    states = {0: (0, [])}
    for curve in curves:
        next_states = {}
        for used, (misses, choices) in states.items():
            for capacity, cost in sorted(curve.items()):
                total = used + capacity
                candidate = (misses + cost, choices + [capacity])
                if total <= budget and (total not in next_states or candidate < next_states[total]):
                    next_states[total] = candidate
        states = next_states
    if not states:
        raise ValueError("budget cannot cover minimum candidate capacities")
    used = min(states, key=lambda total: (states[total][0], total, states[total][1]))
    misses, choices = states[used]
    return {"layer_cache_slots": choices, "used_slots": used, "misses": misses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--capacities", type=int, nargs="+", default=[56, 80, 96, 112, 128])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()
    capacities = sorted(set(args.capacities))
    if min(args.layers, args.experts, args.max_tokens, args.budget, *capacities) <= 0 or max(capacities) > args.experts:
        parser.error("invalid replay dimensions or capacities")
    layers, skipped = read_trace(args.trace, args.layers, args.experts, args.max_tokens)
    curves = [miss_curve(routes, capacities, args.experts) for routes in layers]
    report = {
        "allocation": allocate(curves, args.budget),
        "budget": args.budget,
        "curves": curves,
        "samples_per_layer": [len(routes) for routes in layers],
        "skipped_large_invocations": skipped,
        "uniform_misses": {capacity: sum(curve[capacity] for curve in curves) for capacity in capacities},
        "assumptions": "cold-start decode-only CPU LFU replay, equal slot bytes, candidate-grid optimum; not GPU performance",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["allocation"]))


if __name__ == "__main__":
    main()
