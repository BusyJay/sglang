"""Summarize fresh-process fixed-tape repeats without treating kernels as samples."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from tools.qwen38_cache_tests.cache_cost_replay import read_sweeps, replay_layer


def summarize(folder):
    runs = json.loads((folder / "runs.json").read_text())
    grouped = {}
    baselines = {}
    for run in runs:
        data = json.loads((folder / run["file"]).read_text())
        if data["mode"] != "benchmark" or data["status"] != "passed":
            raise ValueError("Only passed timing workers can enter summary")
        grouped.setdefault((run["policy"], run["fraction"]), []).append((run, data))
        if run["policy"] == "lfu":
            baselines[run["repeat"]] = data["wall_s"]
    result = []
    for (policy, fraction), values in grouped.items():
        times = [d["wall_s"] for _, d in values]
        ratios = [d["wall_s"] / baselines[r["repeat"]] for r, d in values]
        logs = [math.log(ratio) for ratio in ratios]
        mean = statistics.mean(logs)
        ci = None
        if len(logs) == 5:
            # Descriptive paired log-ratio interval: t(4), five independent
            # process repeats conditional on this one recorded workload.
            half = 2.776445 * statistics.stdev(logs) / math.sqrt(5)
            ci = [math.exp(mean - half), math.exp(mean + half)]
        samples = []
        for run, data in values:
            before, after = data["before"], data["after"]
            delta = {
                key: after[key] - before[key]
                for key in [
                    "hits",
                    "misses",
                    "decode_device_gate_copy_bytes",
                    "decode_device_gate_invocations",
                    "decode_device_gate_hits",
                ]
            }
            samples.append(
                dict(
                    repeat=run["repeat"],
                    file=run["file"],
                    wall_s=data["wall_s"],
                    counters=delta,
                    measured_sweeps=data["measured_sweeps"],
                    trace_sha256=data["trace_sha256"],
                )
            )
        if len({s["trace_sha256"] for s in samples}) != 1:
            raise ValueError("Different route tapes in same comparison")
        counters = samples[0]["counters"]
        deterministic = all(s["counters"] == counters for s in samples)
        result.append(
            dict(
                policy=policy,
                fraction=fraction,
                samples=samples,
                layout=values[0][1]["layout"],
                median_wall_s=statistics.median(times),
                min_wall_s=min(times),
                max_wall_s=max(times),
                wall_cv=(
                    statistics.stdev(times) / statistics.mean(times)
                    if len(times) > 1
                    else None
                ),
                geometric_paired_ratio_vs_lfu=math.exp(mean),
                paired_ratios=ratios,
                descriptive_95pct_ratio_interval=ci,
                counters_identical_across_repeats=deterministic,
                measured_sweeps=samples[0]["measured_sweeps"],
            )
        )
    return sorted(result, key=lambda row: row["median_wall_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = dict(
        scope="fixed-route planner/transport, not end-to-end model inference",
        workloads={},
    )
    for case in ["coding", "raft"]:
        rows = summarize(args.results / f"{case}-timing")
        tape = args.results / f"{case}.jsonl"
        if not tape.exists():
            tape = tape.with_suffix(".jsonl.gz")
        raw = (
            gzip.decompress(tape.read_bytes())
            if tape.suffix == ".gz"
            else tape.read_bytes()
        )
        tape_sha = hashlib.sha256(raw).hexdigest()
        valid_hashes = {tape_sha, hashlib.sha256(tape.read_bytes()).hexdigest()}
        if any(
            sample["trace_sha256"] not in valid_hashes
            for row in rows
            for sample in row["samples"]
        ):
            raise ValueError("Archived tape does not match timed route workload")
        sweeps = read_sweeps(tape, 48, 512)
        for row in rows:
            expected_misses = expected_hits = 0
            for layer, capacity in enumerate(row["layout"]):
                expected = replay_layer(
                    [s[layer] for s in sweeps],
                    capacity,
                    512,
                    row["policy"],
                    row["fraction"],
                    lambda _: 0,
                    128,
                    len(sweeps),
                )
                expected_misses += expected["misses"]
                expected_hits += expected["accesses"] - expected["misses"]
            row["cpu_reference"] = dict(hits=expected_hits, misses=expected_misses)
            row["cpu_gpu_policy_counts_identical"] = all(
                sample["counters"]["hits"] == expected_hits
                and sample["counters"]["misses"] == expected_misses
                for sample in row["samples"]
            )
        parity = []
        for file in (args.results / f"{case}-parity").glob("*.json"):
            if file.name == "runs.json":
                continue
            data = json.loads(file.read_text())
            if (
                data["status"] != "passed"
                or data["mode"] != "correctness"
                or data["correctness_checks"] == 0
            ):
                raise ValueError("Parity did not pass")
            parity.append(
                dict(
                    file=file.name,
                    checks=data["correctness_checks"],
                    policy=data["policy"],
                    fraction=data["protected_fraction"],
                )
            )
        if len(parity) != 5:
            raise ValueError("Expected payload parity for five policies")
        cpu = json.loads((args.results / f"{case}-cpu.json").read_text())
        cpu_rows = [
            dict(
                policy=row["policy"],
                fraction=row["protected_fraction"],
                objectives=[
                    dict(
                        objective=r["objective"],
                        misses=r["held_out"]["misses"],
                        fill_ms_surrogate=r["held_out"]["estimated_fill_ms"],
                        layout=r["layout"],
                    )
                    for r in row["results"]
                ],
            )
            for row in cpu["reports"]
        ]
        report["workloads"][case] = dict(gpu=rows, parity=parity, cpu_held_out=cpu_rows)
        print(case)
        for row in rows:
            print(
                row["policy"],
                row["fraction"],
                round(row["median_wall_s"], 4),
                round((row["geometric_paired_ratio_vs_lfu"] - 1) * 100, 2),
                row["descriptive_95pct_ratio_interval"],
                row["samples"][0]["counters"],
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
