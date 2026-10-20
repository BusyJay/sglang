"""Compare saved runtime counters; no inference or server launch required."""

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path


def analyze(directory):
    results = {}
    for mode in ("default", "global_gpu", "legacy_gpu"):
        rows, chunks = [], []
        for number, line in enumerate(
            (directory / f"{mode}.log").read_text().splitlines(), 1
        ):
            if "Qwen38ExpertCacheStats[" in line:
                rows.append((number, line[1:20], json.loads(line[line.index("{") :])))
            if "Prefill batch," in line:
                match = re.search(r"#pending-token: (\d+)", line)
                if match and int(match[1]) > 0:
                    chunks.append(
                        datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
                    )
        # Approximate the long-prefill interval from counter deltas. Short
        # fixtures have 1088*10 route entries/run; long chunks have 8192*10.
        # Snapshot granularity is 128 layer calls, so endpoints include a small
        # amount of adjacent work and must not be treated as exact request spans.
        active = []
        for i in range(1, len(rows)):
            a, b = rows[i - 1][2], rows[i][2]
            calls = b["run_invocations"] - a["run_invocations"]
            routes = b["route_token_count"] - a["route_token_count"]
            if calls > 0 and routes / calls > 30000:
                active.append(i)
        first, last = min(active) - 1, max(active)
        before, after, final = rows[first][2], rows[last][2], rows[-1][2]
        interval = {
            "start_line": rows[first][0],
            "end_line": rows[last][0],
            "start_timestamp_utc": rows[first][1],
            "end_timestamp_utc": rows[last][1],
            "layer_calls": after["run_invocations"] - before["run_invocations"],
            "foreground_transfer_bytes": after["transferred_bytes"]
            - before["transferred_bytes"],
        }
        if "global_cpu_hybrid" in after:
            interval["global_deltas"] = {
                key: after["global_cpu_hybrid"].get(key, 0)
                - before["global_cpu_hybrid"].get(key, 0)
                for key in (
                    "fill_bytes",
                    "insertions",
                    "misses",
                    "prefill_gpu_misses",
                    "prefill_global_hits",
                    "resize_events",
                    "result_join_waits",
                )
            }
        gaps = [(b - a).total_seconds() for a, b in zip(chunks, chunks[1:])]
        results[mode] = {
            "long_prefill_snapshot_interval_approximate": interval,
            "long_prefill_log_gap_median_seconds": statistics.median(gaps),
            "final_fast_path_counters": {
                key: final[key]
                for key in (
                    "fused_decode_plan_invocations",
                    "decode_device_gate_invocations",
                    "decode_device_gate_hits",
                    "decode_shared_fill_overlap_invocations",
                )
            },
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
