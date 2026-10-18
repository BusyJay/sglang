"""Validate and summarize paired, migrated-cache capacity experiments (CPU only)."""

import argparse
import gzip
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

from .capacity_experiment import ALGORITHMS, ROOT, save


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder):
    runs = json.loads((folder / "runs.json").read_text())
    metadata = json.loads((folder / "experiment.json").read_text())
    rounds = metadata["rounds"]
    if len(runs) != 8 * (rounds + 1):
        raise ValueError("incomplete correctness/timing matrix")
    report = dict(scope=metadata["scope"], rounds=rounds, workloads={})
    for case in ("coding", "raft"):
        training = json.loads((folder / f"{case}-training.json").read_text())
        rows = {}
        prefix_counters = set()
        for algorithm in ALGORITHMS:
            selected = [
                r for r in runs if r["case"] == case and r["algorithm"] == algorithm
            ]
            checks = [r for r in selected if r["mode"] == "correctness"]
            timings = [r for r in selected if r["mode"] == "timing"]
            if len(checks) != 1 or sorted(r["repeat"] for r in timings) != list(
                range(rounds)
            ):
                raise ValueError("missing or duplicated process repeat")
            samples = []
            parity = None
            for run in selected:
                d = json.loads((folder / run["file"]).read_text())
                if d["status"] != "passed" or not d["migration_ready"]:
                    raise ValueError("candidate did not use adaptive pool")
                if d["trace_sha256"] != training["trace_sha256"]:
                    raise ValueError("different route tape")
                if (
                    d["layout"] != [134] * 6 + [133] * 42
                    or d["final_layout"] != training["layouts"][algorithm]["layout"]
                ):
                    raise ValueError("wrong initial or final allocation")
                if (
                    d["budget"] != 6390
                    or sum(d["final_layout"]) != 6390
                    or d["warmup_sweeps"] != 512
                ):
                    raise ValueError("budget/prefix mismatch")
                prefix = d["prefix_stats"]
                prefix_counters.add(
                    (
                        prefix["hits"],
                        prefix["misses"],
                        prefix["decode_device_gate_copy_bytes"],
                    )
                )
                if d["mode"] == "correctness":
                    if d["correctness_checks"] != 4608 or d["measured_sweeps"] != 16:
                        raise ValueError("incomplete payload parity")
                    parity = dict(file=run["file"], checks=d["correctness_checks"])
                    continue
                delta = d["counter_delta"]
                if (
                    d["measured_sweeps"] != 512
                    or delta["decode_device_gate_invocations"] != 512 * 48
                ):
                    raise ValueError("incomplete measured GPU interval")
                if delta["decode_device_gate_copy_bytes"] != delta["misses"] * 2764800:
                    raise ValueError("copy byte/miss mismatch")
                expected_rebalances = int(
                    algorithm != "uniform" and d["final_layout"] != d["layout"]
                )
                if (
                    d["before"]["adaptive_rebalances"] != expected_rebalances
                    or d["after"]["adaptive_rebalances"] != expected_rebalances
                ):
                    raise ValueError("missing or unexpected asynchronous migration")
                samples.append(
                    dict(
                        repeat=run["repeat"],
                        file=run["file"],
                        wall_s=d["wall_s"],
                        migration_wall_s=d["migration_wall_s"],
                        inclusive_wall_s=d["inclusive_wall_s"],
                        misses=delta["misses"],
                        hits=delta["hits"],
                        copy_bytes=delta["decode_device_gate_copy_bytes"],
                        migrated_experts=d["before"]["adaptive_migrated_experts"],
                        evicted_experts=d["before"]["adaptive_evicted_experts"],
                    )
                )
            samples.sort(key=lambda r: r["repeat"])
            counters = [(s["misses"], s["hits"], s["copy_bytes"]) for s in samples]
            if (
                len(set(counters)) != 1
                or len({s["migrated_experts"] for s in samples}) != 1
                or len({s["evicted_experts"] for s in samples}) != 1
            ):
                raise ValueError("nondeterministic routing/migration counters")
            times = [s["inclusive_wall_s"] for s in samples]
            rows[algorithm] = dict(
                samples=samples,
                correctness=parity,
                final_layout=training["layouts"][algorithm]["layout"],
                median_wall_s=statistics.median(s["wall_s"] for s in samples),
                median_migration_s=statistics.median(
                    s["migration_wall_s"] for s in samples
                ),
                median_inclusive_s=statistics.median(times),
                min_inclusive_s=min(times),
                max_inclusive_s=max(times),
                inclusive_cv=statistics.stdev(times) / statistics.mean(times),
                misses=samples[0]["misses"],
                copy_bytes=samples[0]["copy_bytes"],
                migrated_experts=samples[0]["migrated_experts"],
                evicted_experts=samples[0]["evicted_experts"],
            )
        if len(prefix_counters) != 1:
            raise ValueError("candidates did not start from identical uniform prefix")
        baseline = rows["uniform"]
        if (
            baseline["misses"]
            != training["layouts"]["uniform"]["static_held_out"]["misses"]
        ):
            raise ValueError("uniform GPU misses differ from CPU reference")
        for row in rows.values():
            ratios = [
                s["inclusive_wall_s"] / b["inclusive_wall_s"]
                for s, b in zip(row["samples"], baseline["samples"])
            ]
            logs = [math.log(r) for r in ratios]
            center = statistics.mean(logs)
            half = (
                2.776445 * statistics.stdev(logs) / math.sqrt(rounds)
                if rounds == 5
                else None
            )
            row["paired_inclusive_ratios_vs_uniform"] = ratios
            row["geometric_paired_ratio_vs_uniform"] = math.exp(center)
            row["descriptive_95pct_ratio_interval"] = (
                None
                if half is None
                else [math.exp(center - half), math.exp(center + half)]
            )
            row["miss_change_pct"] = (row["misses"] / baseline["misses"] - 1) * 100
            saved_per_sweep = (baseline["median_wall_s"] - row["median_wall_s"]) / 512
            row["approx_payback_sweeps_at_observed_average"] = (
                row["median_migration_s"] / saved_per_sweep
                if saved_per_sweep > 0
                else None
            )
            production_ratios = [
                s["inclusive_wall_s"] / b["inclusive_wall_s"]
                for s, b in zip(row["samples"], rows["production"]["samples"])
            ]
            production_logs = [math.log(r) for r in production_ratios]
            production_center = statistics.mean(production_logs)
            production_half = (
                2.776445 * statistics.stdev(production_logs) / math.sqrt(rounds)
                if rounds == 5
                else None
            )
            row["geometric_paired_ratio_vs_production"] = math.exp(production_center)
            row["descriptive_95pct_ratio_interval_vs_production"] = (
                None
                if production_half is None
                else [
                    math.exp(production_center - production_half),
                    math.exp(production_center + production_half),
                ]
            )
        report["workloads"][case] = dict(
            prefix_counters=list(prefix_counters)[0], algorithms=rows
        )
    return report


def archive(source, destination):
    if (
        not (source / "completed.json").exists()
        or not (source / "restored.json").exists()
    ):
        raise ValueError("experiment must finish and restore before archiving")
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file():
            if path.name.endswith("production-request.json"):
                (destination / (path.name + ".gz")).write_bytes(
                    gzip.compress(path.read_bytes(), mtime=0)
                )
            else:
                shutil.copyfile(path, destination / path.name)
    # Reuse the previous committed compressed tape; validate decoded content.
    old_archive = Path(__file__).parent / "live_fp4_results"
    for case in ("coding", "raft"):
        tape = old_archive / f"{case}.jsonl.gz"
        training = json.loads((source / f"{case}-training.json").read_text())
        if (
            hashlib.sha256(gzip.decompress(tape.read_bytes())).hexdigest()
            != training["trace_sha256"]
        ):
            raise ValueError("archived route tape mismatch")
        # Tapes remain in the adjacent tracked prior archive to avoid duplication.
    files = [
        ROOT / "tools/qwen38_cache_tests/cache_transport_replay.py",
        Path(__file__),
        Path(__file__).with_name("capacity_experiment.py"),
        Path(__file__).with_name("capacity_grid_diagnostic.py"),
        ROOT / "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
        ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py",
        ROOT / "python/sglang/srt/layers/moe/qwen38_cache_adaptive.py",
    ]
    save(
        destination / "source-hashes.json",
        {str(p.relative_to(ROOT)): sha(p) for p in files},
    )
    save(
        destination / "SHA256.json",
        {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = summarize(args.folder)
    save(args.folder / "summary.json", report)
    if args.archive:
        archive(args.folder, args.archive)
    for case, workload in report["workloads"].items():
        for algorithm, row in workload["algorithms"].items():
            print(
                case,
                algorithm,
                f"{row['median_inclusive_s']:.4f}s",
                f"{(row['geometric_paired_ratio_vs_uniform'] - 1) * 100:+.2f}%",
                "misses",
                row["misses"],
                "migration",
                row["median_migration_s"],
            )


if __name__ == "__main__":
    main()
