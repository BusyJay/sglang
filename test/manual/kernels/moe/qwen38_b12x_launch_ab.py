#!/usr/bin/env python3
"""Run existing independent-reference parity suite for baseline and prepared launch."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sglang.srt.environ import envs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 5:
        parser.error("rounds >=1 and iterations >=5 required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.rounds):
        order = [(False, "baseline"), (True, "prepared")]
        if repeat % 2:
            order.reverse()
        for enabled, name in order:
            with envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.override(enabled):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tools.qwen38_cache_tests.b12x_gpu_check",
                        "--output",
                        str(args.output_dir / (name + f"-{repeat}.json")),
                        "--iterations",
                        str(args.iterations),
                    ],
                    check=True,
                )
        baseline = json.loads((args.output_dir / f"baseline-{repeat}.json").read_text())
        prepared = json.loads((args.output_dir / f"prepared-{repeat}.json").read_text())
        assert len(baseline["reports"]) == len(prepared["reports"])
        for a, b in zip(baseline["reports"], prepared["reports"]):
            keys = ("tokens", "pattern", "ids_dtype")
            assert all(a[k] == b[k] for k in keys)
            rows.append(
                {
                    "repeat": repeat,
                    **{k: a[k] for k in keys},
                    "baseline_ms": a["b12x"]["median_ms"],
                    "prepared_ms": b["b12x"]["median_ms"],
                    "speedup": a["b12x"]["median_ms"] / b["b12x"]["median_ms"],
                }
            )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(
            dict(
                status="passed",
                reports=rows,
                scope="Synthetic parity and microbench; requires separate full-model A/B",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
