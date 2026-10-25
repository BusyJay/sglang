"""Stream a real-workload Kineto trace without retaining its CPU op graph."""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

import ijson
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--sample-prefill-ms", type=float, default=500)
parser.add_argument("--sample-decode-cycles", type=int, default=5)
parser.add_argument("--overlap-only", action="store_true")
options = parser.parse_args()
ROOT = options.output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
TRACE = options.trace.resolve()


def events():
    with gzip.open(TRACE, "rb") as f:
        yield from ijson.items(f, "traceEvents.item", use_float=True)


def union(intervals):
    end = float("-inf")
    total = 0.0
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def intersection_us(a, b):
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def measure_fill_overlap_only():
    kernel_streams = collections.defaultdict(list)
    fill_intervals = []
    fill_streams = set()
    for e in events():
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        interval = (e["ts"], e["ts"] + e["dur"])
        stream = str(e.get("args", {}).get("stream"))
        kernel_streams[stream].append(interval)
        if "mailbox_fill_kernel" in e.get("name", ""):
            fill_intervals.append(interval)
            fill_streams.add(stream)
    fill = merge_intervals(fill_intervals)
    other = merge_intervals(
        [
            v
            for stream, vv in kernel_streams.items()
            if stream not in fill_streams
            for v in vv
        ]
    )
    total = sum(b - a for a, b in fill)
    overlap = intersection_us(fill, other)
    result = {
        "fill_streams": sorted(fill_streams),
        "fill_union_seconds": total / 1e6,
        "overlap_any_other_stream_kernel_seconds": overlap / 1e6,
        "overlap_fraction": overlap / total if total else None,
        "scope": "full trace; simultaneous execution, not causal savings",
    }
    (ROOT / "fill-overlap-only.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    cats = collections.Counter()
    steps = []
    gpu_steps = []
    top = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for e in events():
        cats[e.get("cat", "")] += 1
        if e.get("ph") != "X":
            continue
        c, n = e.get("cat", ""), e.get("name", "")
        if n.startswith("step["):
            (gpu_steps if c == "gpu_user_annotation" else steps).append(e)
        if c in ("kernel", "gpu_memcpy", "cuda_runtime", "cpu_op"):
            s = top[c][n]
            s[0] += 1
            s[1] += e.get("dur", 0)
    steps.sort(key=lambda e: e["ts"])
    (ROOT / "steps.json").write_text(
        json.dumps({"cpu": steps, "gpu": gpu_steps}, indent=2)
    )
    print(
        "categories",
        dict(cats),
        "CPU steps",
        len(steps),
        "GPU steps",
        len(gpu_steps),
        flush=True,
    )
    grouped_gpu = collections.defaultdict(list)
    for e in gpu_steps:
        grouped_gpu[e.get("args", {}).get("External id")].append(e)
    # One GPU span per stream may exist for the same CPU forward.
    spans = []
    for e in steps:
        g = grouped_gpu.get(e.get("args", {}).get("External id"), [])
        name = e["name"]
        mode = name.split("[", 1)[1].split(" ", 1)[0]
        stage = "prefill" if mode in ("EXTEND", "MIXED", "DRAFT") else "decode"
        a = min(x["ts"] for x in g) if g else e["ts"]
        b = max(x["ts"] + x["dur"] for x in g) if g else e["ts"] + e["dur"]
        spans.append((a, b, stage, name, bool(g)))
    spans.sort()
    starts = [s[0] for s in spans]
    gpu_top = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    cpu_top = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    gpu_events = collections.defaultdict(list)
    streams = collections.defaultdict(lambda: collections.defaultdict(list))
    copy_bytes = collections.Counter()
    kernels = []
    kernel_streams = collections.defaultdict(list)
    fill_intervals = []
    fill_streams = set()
    samples = collections.defaultdict(list)
    # Representative prefill (second long chunk) and complete steady decode
    # cycle; full-trace metrics below are calculated independently.
    ext = [s for s in spans if s[3].startswith("step[EXTEND ")]
    selected = {}
    if len(ext) > 1:
        selected["prefill"] = (
            ext[1][0],
            min(ext[1][1], ext[1][0] + options.sample_prefill_ms * 1000),
        )
    verify = [
        s for s in spans if s[3].startswith(("step[TARGET_VERIFY", "step[VERIFY"))
    ]
    if len(verify) > 300:
        selected["decode"] = (
            verify[200][0],
            verify[200 + options.sample_decode_cycles][0],
        )
    # Macro phases include draft, sampling and graph work outside the target's
    # ModelRunner.forward span. A target-only denominator would hide that work.
    vstarts = sorted(s[0] for s in verify)
    prefill_regions = []
    for a, b, stage, *_ in spans:
        if stage != "prefill":
            continue
        j = bisect.bisect_right(vstarts, a)
        end = vstarts[j] if j < len(vstarts) else b
        if prefill_regions and a <= prefill_regions[-1][1]:
            prefill_regions[-1] = (
                prefill_regions[-1][0],
                max(end, prefill_regions[-1][1]),
            )
        else:
            prefill_regions.append((a, end))
    print("representative windows", selected, flush=True)
    for e in events():
        if e.get("ph") == "M":
            for k in selected:
                samples[k].append(e)
            continue
        if e.get("ph") != "X" or "dur" not in e:
            continue
        a, b = e["ts"], e["ts"] + e["dur"]
        for k, (lo, hi) in selected.items():
            if lo <= a and a < hi:
                samples[k].append(e)
        cat = e.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "cuda_runtime"):
            continue
        i = bisect.bisect_right(starts, a) - 1
        stage = (
            "prefill"
            if any(lo <= a < hi for lo, hi in prefill_regions)
            else "decode"
            if i >= 0
            else "between_steps"
        )
        n = e.get("name", "")
        dest = gpu_top if cat in ("kernel", "gpu_memcpy", "gpu_memset") else cpu_top
        s = dest[stage][(cat, n)]
        s[0] += 1
        s[1] += e["dur"]
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            interval = (a, b)
            stream = str(e.get("args", {}).get("stream"))
            gpu_events[stage].append(interval)
            streams[stage][stream].append(interval)
            if cat == "kernel":
                kernel_streams[stream].append(interval)
                if "mailbox_fill_kernel" in n:
                    fill_intervals.append(interval)
                    fill_streams.add(stream)
            if cat == "gpu_memcpy":
                copy_bytes[stage + ":" + n] += e.get("args", {}).get("bytes", 0)
            kernels.append(interval)
    result = {
        "categories": dict(cats),
        "step_count": len(steps),
        "gpu_step_count": len(gpu_steps),
        "stage_metrics": {},
        "copy_bytes": dict(copy_bytes),
        "global_top": {},
        "representative_windows": selected,
        "prefill_regions": prefill_regions,
    }
    for k, rows in top.items():
        result["global_top"][k] = [
            {"name": n, "count": v[0], "seconds": v[1] / 1e6}
            for n, v in sorted(rows.items(), key=lambda kv: -kv[1][1])[:40]
        ]
    for stage in ("prefill", "decode", "between_steps"):
        stage_spans = [(a, b) for a, b, st, *_ in spans if st == stage]
        durations = [
            e["dur"] / 1000
            for e in steps
            if (
                "prefill"
                if e["name"].startswith(("step[EXTEND ", "step[MIXED ", "step[DRAFT "))
                else "decode"
            )
            == stage
        ]
        result["stage_metrics"][stage] = {
            "forward_window_seconds": union(stage_spans) / 1e6,
            "gpu_busy_seconds": union(gpu_events[stage]) / 1e6,
            "gpu_sum_seconds": sum(v[1] for v in gpu_top[stage].values()) / 1e6,
            "forward_count": len(stage_spans),
            "gpu_matched_step_count": sum(g for a, b, st, n, g in spans if st == stage),
            "cpu_forward_ms_percentiles": (
                np.percentile(durations, [0, 50, 90, 99, 100]).tolist()
                if durations
                else []
            ),
            "gpu_stream_busy_seconds": {
                s: union(v) / 1e6 for s, v in streams[stage].items()
            },
            "gpu_top": [
                {"category": cat, "name": n, "count": v[0], "seconds": v[1] / 1e6}
                for (cat, n), v in sorted(
                    gpu_top[stage].items(), key=lambda kv: -kv[1][1]
                )[:40]
            ],
            "cpu_top": [
                {"category": cat, "name": n, "count": v[0], "seconds": v[1] / 1e6}
                for (cat, n), v in sorted(
                    cpu_top[stage].items(), key=lambda kv: -kv[1][1]
                )[:40]
            ],
        }
    all_gpu = sorted(kernels)
    result["all_gpu_busy_seconds"] = union(all_gpu) / 1e6
    result["gpu_activity_range"] = (
        [all_gpu[0][0], max(b for a, b in all_gpu)] if all_gpu else []
    )
    gaps = []
    end = all_gpu[0][0]
    for a, b in all_gpu:
        if a > end:
            gaps.append((end, a, (a - end) / 1000))
        end = max(end, b)
    result["largest_gpu_gaps_ms"] = sorted(gaps, key=lambda x: -x[2])[:40]
    result["gpu_gap_seconds"] = sum(g[2] for g in gaps) / 1000
    # Simultaneous execution is evidence of overlap, not latency savings.
    # Different streams are required; same-stream sequential work is excluded.
    fill_union = merge_intervals(fill_intervals)
    other = merge_intervals(
        [
            v
            for stream, vv in kernel_streams.items()
            if stream not in fill_streams
            for v in vv
        ]
    )
    fill_us = sum(b - a for a, b in fill_union)
    overlap_us = intersection_us(fill_union, other)
    result["fill_overlap"] = {
        "fill_streams": sorted(fill_streams),
        "fill_union_seconds": fill_us / 1e6,
        "overlap_any_other_stream_kernel_seconds": overlap_us / 1e6,
        "overlap_fraction": overlap_us / fill_us if fill_us else None,
        "scope": "full trace; simultaneous execution, not causal savings",
    }
    (ROOT / "full-metrics.json").write_text(json.dumps(result, indent=2))
    for k, ee in samples.items():
        with gzip.open(ROOT / (k + "-representative.trace.json.gz"), "wt") as f:
            json.dump({"traceEvents": ee, "displayTimeUnit": "ms"}, f)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("global_top", "stage_metrics")
            },
            indent=2,
        ),
        flush=True,
    )
    for stage, m in result["stage_metrics"].items():
        print(
            stage,
            json.dumps(
                {k: v for k, v in m.items() if k not in ("gpu_top", "cpu_top")},
                indent=2,
            ),
            flush=True,
        )
        print("top GPU", json.dumps(m["gpu_top"][:12], indent=2), flush=True)


if __name__ == "__main__":
    if options.overlap_only:
        measure_fill_overlap_only()
    else:
        main()
