"""Summarize the paired streaming benchmark without averaging idle-time logs."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    raw = json.loads((directory / "performance.json").read_text())
    if raw["status"] != "passed":
        raise ValueError("benchmark is incomplete")
    summary = {"status": raw["status"], "modes": {}}
    for mode, result in raw["modes"].items():
        cases = []
        for case in result["cases"]:
            reduced = {
                k: v
                for k, v in case.items()
                if k not in ("stream_events", "response_text", "meta_info")
            }
            reduced["cached_tokens"] = case["meta_info"].get("cached_tokens")
            reduced["spec_accept_length"] = case["meta_info"].get("spec_accept_length")
            if reduced["cached_tokens"] != 0:
                raise ValueError(f"{mode}/{case['name']} has a radix prefix hit")
            cases.append(reduced)
        short = [case for case in cases if case["name"].startswith("short_")]
        aggregate = {}
        for metric in (
            "ttft_seconds",
            "e2e_seconds",
            "decode_tokens_per_second",
            "after_64_tokens_per_second",
        ):
            values = [case[metric] for case in short]
            aggregate[metric] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        snapshots = []
        initialization = []
        for line in (
            (directory / f"{mode}.log").read_text(errors="replace").splitlines()
        ):
            if "Qwen38 CPU hybrid enabled:" in line:
                initialization.append(line)
            if "Qwen38ExpertCacheStats[" in line:
                snapshots.append(json.loads(line[line.index("{") :]))
        transitions = []
        for snapshot in snapshots:
            stage = snapshot.get("kv_stage")
            if not transitions or transitions[-1]["stage"] != stage:
                transitions.append(
                    {
                        "stage": stage,
                        "kv_backed_tokens": snapshot.get("kv_backed_tokens"),
                        "global_cache_slots": snapshot.get("global_cache_slots"),
                    }
                )
        final = snapshots[-1] if snapshots else {}
        summary["modes"][mode] = {
            "overrides": result["overrides"],
            "startup_seconds": result["startup_seconds"],
            "short_aggregate": aggregate,
            "cases": cases,
            "transitions": transitions,
            "cpu_initialization": initialization,
            "final_cpu_stats": final.get("global_cpu_hybrid"),
            "final_kv_stage": final.get("kv_stage"),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.directory)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                mode: result["short_aggregate"]
                for mode, result in summary["modes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
