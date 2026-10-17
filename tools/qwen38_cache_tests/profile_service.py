#!/usr/bin/env python3
"""Capture a bounded service profile or summarize a Chrome trace (stdlib only)."""

import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import time
import urllib.request


def request(url, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + endpoint, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read().decode()


def union_us(intervals):
    end, total = float("-inf"), 0.0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def summarize(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        trace = json.load(stream)
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    # Restrict to complete decode/verify spans. Exclude idle and prefill from
    # the denominator, and clip GPU activity at span boundaries.
    steps = [e for e in events if e.get("cat") == "user_annotation"
             and e.get("name", "").startswith(("step[VERIFY", "step[DECODE"))]
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in steps]
    if not windows:
        raise ValueError("No complete DECODE/VERIFY step spans; capture decode with CPU activity enabled")
    windows.sort()
    # Device work can outlive the CPU annotation after asynchronous submission.
    # Match Kineto GPU spans by External id instead of cutting off queued kernels
    # at the CPU return (especially important for device-side expert fills).
    gpu_spans = collections.defaultdict(list)
    external_ids = {e.get("args", {}).get("External id") for e in steps}
    external_ids.discard(None)
    for e in events:
        if (e.get("cat") == "gpu_user_annotation"
                and e.get("args", {}).get("External id") in external_ids):
            gpu_spans[e["args"]["External id"]].append((e["ts"], e["ts"] + e["dur"]))
    gpu_windows = sorted(
        (min(a for a, b in spans), max(b for a, b in spans))
        for spans in gpu_spans.values()
    ) if len(gpu_spans) == len(steps) else windows
    gpu_selection = []
    for lo, hi in gpu_windows:
        if gpu_selection and lo <= gpu_selection[-1][1]:
            gpu_selection[-1] = (gpu_selection[-1][0], max(hi, gpu_selection[-1][1]))
        else:
            gpu_selection.append((lo, hi))
    selected = []
    for e in events:
        begin, end = e["ts"], e["ts"] + e["dur"]
        # One event can cross a step boundary: clip without double-counting.
        chosen_windows = gpu_selection if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") else windows
        for lo, hi in chosen_windows:
            if hi <= begin:
                continue
            if lo >= end:
                break
            overlap = min(end, hi) - max(begin, lo)
            if overlap > 0:
                selected.append((e, overlap, max(begin, lo)))
    groups = {}
    busy = collections.defaultdict(list)
    copy_bytes = collections.Counter()
    for e, duration, begin in selected:
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            busy[e.get("args", {}).get("device", e.get("pid"))].append((begin, begin + duration))
        if cat == "gpu_memcpy":
            copy_bytes[e["name"]] += e.get("args", {}).get("bytes", 0) * duration / e["dur"]
        if cat not in ("kernel", "gpu_memcpy", "cuda_runtime", "cpu_op"):
            continue
        counts = groups.setdefault(cat, collections.defaultdict(lambda: [0, 0.0]))
        counts[e["name"]][0] += 1
        counts[e["name"]][1] += duration / 1000
    wall_ms = union_us(windows) / 1000
    return {
        "trace": str(path), "decode_steps": len(steps), "decode_wall_ms": wall_ms,
        "mean_step_ms": wall_ms / len(steps),
        "gpu_step_window_ms": union_us(gpu_windows) / 1000,
        "gpu_mean_step_ms": sum(b - a for a, b in gpu_windows) / 1000 / len(steps),
        "gpu_window_source": "matched GPU annotations" if gpu_windows is not windows else "CPU annotation fallback",
        "gpu_busy_ms_by_device": {str(k): union_us(v) / 1000 for k, v in busy.items()},
        "copy_bytes_per_step": {k: v / len(steps) for k, v in copy_bytes.items()},
        "top": {cat: sorted(
            ({"name": name, "calls": count, "total_ms": ms, "ms_per_step": ms / len(steps)}
             for name, (count, ms) in rows.items()), key=lambda row: row["total_ms"], reverse=True,
        )[:20] for cat, rows in groups.items()},
        "notes": ["Profiler adds overhead; measure throughput separately without profiling.",
                  "Category/operator totals overlap and are not additive wall time.",
                  "CPU step spans measure host submission; queued GPU work uses matched GPU spans when available.",
                  "GPU graph nodes are included; GPU busy time is an interval union per device.",
                  "Python worker time may be absent from torch CPU events; use NVTX + Nsight Systems for worker staging."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "stop"):
        p = sub.add_parser(action)
        p.add_argument("--url", default=os.getenv("PROFILE_URL", "http://127.0.0.1:8329"))
        if action == "start":
            p.add_argument("--steps", type=int, default=20)
            p.add_argument("--output-dir", default="/prod/homelab-llm/models/profiles")
            p.add_argument("--prefix", default="fp4")
            p.add_argument("--with-stack", action="store_true")
            p.add_argument("--record-shapes", action="store_true")
            p.add_argument("--metadata", type=Path, help="Local server configuration output path")
    p = sub.add_parser("summary")
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "summary":
        result = json.dumps(summarize(args.trace), indent=2)
        if args.output:
            args.output.write_text(result + "\n")
        print(result)
    elif args.action == "start":
        if args.steps <= 0:
            parser.error("--steps must be positive")
        # Fetch metadata before starting; output-dir is a server-side path.
        info = request(args.url, "/get_server_info")
        output = Path(args.output_dir)
        metadata = args.metadata or (
            output if output.is_dir() else Path.cwd() / "profiles"
        ) / (args.prefix + f"-server-{time.time_ns()}.json")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(info + "\n")
        print(request(args.url, "/start_profile", {
            "output_dir": args.output_dir, "num_steps": args.steps,
            "activities": ["CPU", "GPU"], "with_stack": args.with_stack,
            "record_shapes": args.record_shapes, "profile_prefix": args.prefix,
        }))
        print(f"Capturing {args.steps} forward steps; stops automatically as requests run. "
              f"Traces are written on the server under {args.output_dir}. Use stop to end early.")
        print(f"Server configuration: {metadata}")
    else:
        print(request(args.url, "/stop_profile", {}))


if __name__ == "__main__":
    main()
