#!/usr/bin/env python3
"""Summarize every completed sample, keeping timing and acceptance separate."""

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    groups, rounds = {}, []
    for path in sorted(args.results.glob("round-*/result.json")):
        result = json.loads(path.read_text())
        if "server_info_after" not in result:
            continue
        log = path.with_name("service.log")
        opener = open
        if not log.exists():
            log, opener = path.with_name("service.log.gz"), gzip.open
        stats, evaluations, capacity = [], [], None
        allocator_retries = 0
        with opener(log, "rt") as stream:
            for line in stream:
                allocator_retries += "memory mapping failed with OOM" in line
                if "Qwen38ExpertCacheStats[" in line:
                    stats.append(json.loads(line[line.index("{") :]))
                if (
                    "Qwen38 adaptive cache evaluation:" in line
                    or "Qwen38 adaptive cache: layer slots" in line
                ):
                    evaluations.append(line.strip())
                if "KV Cache is allocated." in line and "#tokens:" in line:
                    capacity = int(line.split("#tokens:")[1].split(",")[0])
        memory_path = path.with_name("gpu-memory.csv")
        memory_open = open
        if not memory_path.exists():
            memory_path, memory_open = path.with_name("gpu-memory.csv.gz"), gzip.open
        with memory_open(memory_path, "rt") as memory_file:
            memory_rows = list(csv.reader(memory_file))[1:]
        peak = max(float(row[1].strip().split()[0]) for row in memory_rows)
        rounds.append(
            dict(
                folder=path.parent.name,
                variant=result["variant"],
                kv_capacity=capacity,
                quality_passed=sum(v["passed"] for v in result["quality"]),
                quality_total=len(result["quality"]),
                adaptive_logs=evaluations,
                final_stats=stats[-1] if stats else None,
                sampled_gpu_peak_mib=peak,
                allocator_mapping_retries=allocator_retries,
            )
        )
        for row in result["cases"]:
            groups.setdefault((result["variant"], row["kind"]), []).append(row)
    metrics = []
    for (variant, kind), rows in sorted(groups.items()):
        tokens = sum(r["stream_points"][-1][1] - r["stream_points"][0][1] for r in rows)
        elapsed = sum(
            r["stream_points"][-1][0] - r["stream_points"][0][0] for r in rows
        )
        total_verify = sum(r["meta_info"]["spec_verify_ct"] for r in rows)
        accepted = sum(r["meta_info"]["completion_tokens"] for r in rows)
        metrics.append(
            dict(
                variant=variant,
                kind=kind,
                samples=len(rows),
                decode_tokens_per_second=tokens / elapsed,
                ttft_s=statistics.mean(r["ttft_s"] for r in rows),
                e2e_tokens_per_second=accepted / sum(r["wall_s"] for r in rows),
                acceptance_length=accepted / total_verify,
                server_seconds_per_verify=sum(
                    r["meta_info"]["e2e_latency"] - r["ttft_s"] for r in rows
                )
                / total_verify,
                sample_decode_tps=[r["decode_tokens_per_second"] for r in rows],
            )
        )
    for record in metrics:
        baseline = next(
            (
                r
                for r in metrics
                if r["variant"] == "baseline" and r["kind"] == record["kind"]
            ),
            None,
        )
        if baseline:
            record["decode_change_percent"] = 100 * (
                record["decode_tokens_per_second"]
                / baseline["decode_tokens_per_second"]
                - 1
            )
    (args.results / "comparison.json").write_text(
        json.dumps(dict(metrics=metrics, rounds=rounds), indent=2) + "\n"
    )
    for row in metrics:
        print(
            row["variant"],
            row["kind"],
            row["samples"],
            f"{row['decode_tokens_per_second']:.2f} tok/s",
            f"{row.get('decode_change_percent', 0):+.2f}%",
            f"TTFT {row['ttft_s']:.3f}s",
            f"accept {row['acceptance_length']:.3f}",
        )


if __name__ == "__main__":
    main()
