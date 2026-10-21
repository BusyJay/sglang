#!/usr/bin/env python3
"""Report the best observed causal adaptive configuration per request."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).with_name(name)
            for name in (
                "transfer-replay-adaptive.json",
                "transfer-replay-adaptive-fast.json",
                "transfer-replay-adaptive-wide.json",
            )
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("transfer-replay-adaptive-best.json"),
    )
    args = parser.parse_args()
    curves, refs = defaultdict(list), {}
    for path in args.inputs:
        data = json.loads(path.read_text())
        for row in data["reports"]:
            key = row["budget"], row["case"]
            if key in refs and refs[key] != row["global_gpu"]:
                raise ValueError("incompatible global baseline")
            refs[key] = row["global_gpu"]
            curves[key].extend(row["configs"])
    aggregates, reports = defaultdict(lambda: defaultdict(int)), []
    for (budget, case), configs in sorted(curves.items()):
        best = min(configs, key=lambda c: c["metrics"]["gpu_fills"])
        best_total = min(configs, key=lambda c: c["metrics"]["total_payload_bytes"])
        global_gpu = refs[budget, case]
        metrics = best["metrics"]
        aggregates[budget]["baseline_fills"] += metrics["gpu_fills"]
        aggregates[budget]["global_fills"] += global_gpu["gpu_fills"]
        aggregates[budget]["baseline_migration_copies"] += metrics.get(
            "migration_slot_copies", 0
        )
        reports.append(
            dict(
                budget=budget,
                case=case,
                best_h2d_config=best,
                best_total_payload_config=best_total,
                global_gpu=global_gpu,
            )
        )
    output = dict(
        status="best_observed_causal_config_envelope",
        reports=reports,
        aggregates=aggregates,
        provenance={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.inputs
        },
        limits=[
            "Best configuration selected per request after replay; not an online oracle or proof of globally optimal performance."
        ],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
