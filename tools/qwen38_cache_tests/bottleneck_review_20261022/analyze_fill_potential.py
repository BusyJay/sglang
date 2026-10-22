# SPDX-License-Identifier: Apache-2.0
"""Rebucket historical paired timings; never emit a runtime calibration."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze(report, shape):
    coarse, fine = defaultdict(list), defaultdict(list)
    for row in report["cases"]:
        key = tuple(tuple(v) if isinstance(v, list) else v for v in row["key"])
        fills = (
            len(row["admitted_experts"])
            + len(row["rejected_experts"])
            - len(row["cpu_experts"])
        )
        coarse[key].append(row)
        fine[key + (fills,)].append(row)

    def eligible(groups):
        return {
            id(row)
            for rows in groups.values()
            if all(row["accepted"] for row in rows)
            for row in rows
        }

    old, new = eligible(coarse), eligible(fine)
    return dict(
        shape=shape,
        cases=len(report["cases"]),
        coarse_keys=sum(all(r["accepted"] for r in rows) for rows in coarse.values()),
        fill_keys=sum(all(r["accepted"] for r in rows) for rows in fine.values()),
        coarse_eligible_representatives=len(old),
        fill_eligible_representatives=len(new),
        recovered_representatives=len(new - old),
    )


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=here.parent / "plan_20261022"
    )
    parser.add_argument("--output", type=Path, default=here / "fill_potential.json")
    args = parser.parse_args()
    groups = [
        analyze(
            json.loads((args.source_dir / f"gate-final-{shape}.json").read_text()),
            shape,
        )
        for shape in ("m3", "m12")
    ]
    args.output.write_text(
        json.dumps(
            dict(
                note="Historical timings only; not a runtime calibration or speedup measurement.",
                groups=groups,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
