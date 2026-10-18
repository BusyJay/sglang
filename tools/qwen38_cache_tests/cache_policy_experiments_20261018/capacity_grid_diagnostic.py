"""CPU diagnostic: hold the candidate grid fixed to separate grid/objective effects.

Does not select or change any GPU experiment arm.
"""

import argparse
import time
from pathlib import Path

from tools.qwen38_cache_tests.cache_cost_replay import (
    FillCost,
    aggregate,
    read_json,
    read_sweeps,
    replay_layer,
)
from tools.qwen38_cache_tests.replay_cache import allocate

from .capacity_experiment import PREVIOUS, ROOT, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cost = FillCost(
        read_json(
            ROOT
            / "tools/qwen38_cache_tests/launch_fill_experiments_20261018/results/fill-prefill-overlap.json.gz"
        ),
        "mapped_adaptive",
        "median_gpu_ms",
    )
    for case in ("coding", "raft"):
        started = time.perf_counter()
        sweeps = read_sweeps(PREVIOUS / f"{case}.jsonl", 48, 512)
        events = [[s[layer] for s in sweeps] for layer in range(48)]
        production = read_json(args.output / f"{case}-production-reply.json")
        capacities = production["candidates"]
        curves = [
            {c: replay_layer(e, c, 512, "lfu", 0.7, cost, 128, 512) for c in capacities}
            for e in events
        ]
        rows = {}
        for name, field in [
            ("min_misses", "misses"),
            ("min_fill_ms", "estimated_fill_ms"),
        ]:
            chosen = allocate(
                [{c: r[field] for c, r in curve.items()} for curve in curves], 6390
            )
            layout = chosen["layer_cache_slots"]
            rows[name] = dict(
                layout=layout,
                matches_production=layout == production["target"],
                used_slots=sum(layout),
                static_held_out=aggregate(
                    [
                        replay_layer(e, c, 512, "lfu", 0.7, cost, 512, 1024)
                        for e, c in zip(events, layout)
                    ]
                ),
            )
        save(
            args.output / f"{case}-grid-diagnostic.json",
            dict(
                scope="CPU static-layout diagnostic only, no additional GPU arm",
                candidates=capacities,
                results=rows,
                seconds=time.perf_counter() - started,
            ),
        )
        print(
            case,
            {name: row["matches_production"] for name, row in rows.items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
