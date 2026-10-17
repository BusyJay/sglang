#!/usr/bin/env python3
"""Summarize the saved A/B streams without discarding slow cases."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.results.resolve()
rounds = []
for path in sorted(root.glob("round-*/result.json")):
    result = json.loads(path.read_text())
    result["folder"] = path.parent.name
    rows = list(csv.DictReader((path.parent / "gpu-memory.csv").open()))
    result["sampled_peak_mib"] = max(
        int(row[" memory.used [MiB]"].split()[0]) for row in rows
    )
    rounds.append(result)

summary = {"rounds": [], "by_mode": {}, "output_comparisons": []}
for result in rounds:
    summary["rounds"].append(
        {
            "folder": result["folder"],
            "mode": result["mode"],
            "startup_s": result["startup_s"],
            "sampled_peak_mib": result["sampled_peak_mib"],
            "quality_passed": sum(row["passed"] for row in result.get("quality", [])),
            "quality_total": len(result.get("quality", [])),
        }
    )
for mode in ("w4a16", "w4a4"):
    grouped = [result for result in rounds if result["mode"] == mode]
    summary["by_mode"][mode] = {}
    for kind in ("short", "long"):
        cases = [
            row for result in grouped for row in result["cases"] if row["kind"] == kind
        ]
        if not cases:
            continue
        summary["by_mode"][mode][kind] = {
            "n": len(cases),
            "decode_tps": sum(
                row["stream_points"][-1][1] - row["stream_points"][0][1]
                for row in cases
            )
            / sum(
                row["stream_points"][-1][0] - row["stream_points"][0][0]
                for row in cases
            ),
            "e2e_tps": sum(row["meta_info"]["completion_tokens"] for row in cases)
            / sum(row["wall_s"] for row in cases),
            "mean_ttft_s": statistics.mean(row["ttft_s"] for row in cases),
            "median_ttft_s": statistics.median(row["ttft_s"] for row in cases),
            "accept_length": sum(row["meta_info"]["completion_tokens"] for row in cases)
            / sum(row["meta_info"]["spec_verify_ct"] for row in cases),
            "prompt_tokens": [row["meta_info"]["prompt_tokens"] for row in cases],
            "decode_tps_cases": [row["decode_tokens_per_second"] for row in cases],
            "ttft_s_cases": [row["ttft_s"] for row in cases],
        }
if rounds:
    baseline = rounds[0]
    for result in rounds[1:]:
        for expected, actual in zip(baseline["cases"], result["cases"]):
            assert expected["prompt_sha256"] == actual["prompt_sha256"]
            prefix = 0
            for x, y in zip(expected["text"], actual["text"]):
                if x != y:
                    break
                prefix += 1
            summary["output_comparisons"].append(
                {
                    "folder": result["folder"],
                    "kind": actual["kind"],
                    "case": actual["case"],
                    "equal": expected["text"] == actual["text"],
                    "common_prefix_characters": prefix,
                    "baseline_sha256": hashlib.sha256(
                        expected["text"].encode()
                    ).hexdigest(),
                    "actual_sha256": hashlib.sha256(
                        actual["text"].encode()
                    ).hexdigest(),
                }
            )
(root / "comparison.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        {k: v for k, v in summary.items() if k != "output_comparisons"},
        indent=2,
        ensure_ascii=False,
    )
)
