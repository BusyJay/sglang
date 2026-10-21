"""Independent per-layer dependency audit of an Nsight Systems SQLite export.

No input from the previous analysis JSON or report. Includes kernels, memcpy
and memset in device activity. Host intervals are clipped before unioning.
"""

import argparse
import bisect
import collections
import json
import sqlite3
import statistics
from pathlib import Path


def union(spans):
    out = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class Intervals:
    def __init__(self, rows):
        self.rows = sorted(rows)
        self.starts = [r[0] for r in self.rows]

    def inside(self, a, b):
        lo = bisect.bisect_left(self.starts, a)
        hi = bisect.bisect_left(self.starts, b)
        return [r for r in self.rows[lo:hi] if r[1] <= b]

    def overlap(self, a, b):
        # Used only on disjoint, already-unioned intervals.
        lo = max(0, bisect.bisect_right(self.starts, a) - 1)
        hi = bisect.bisect_left(self.starts, b)
        return sum(max(0, min(b, y) - max(a, x)) for x, y in self.rows[lo:hi])


def summary(values):
    if not values:
        return {"count": 0, "sum_ms": 0}
    return {
        "count": len(values),
        "sum_ms": sum(values) / 1e6,
        "median_us": statistics.median(values) / 1e3,
        "p95_us": sorted(values)[int(0.95 * (len(values) - 1))] / 1e3,
    }


def analyze(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    nvtx = [
        (a, b, tid, text or strings.get(textid, ""))
        for a, b, tid, text, textid in db.execute(
            "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end>start"
        )
    ]
    named = collections.defaultdict(list)
    for a, b, tid, name in nvtx:
        named[name].append((a, b, tid))

    def scopes(suffix):
        return sorted(
            r for name, rows in named.items() if name.endswith(suffix) for r in rows
        )

    runs = scopes("HybridCache.run")
    prefills = scopes("HybridCache.prefill")
    if not runs or not prefills:
        raise ValueError("hybrid function annotations required")
    # Isolate the sustained decode tail after the last short-request prefill.
    # No layer-count assumption, no client/host clock conversion.
    start = max(b for a, b, tid in prefills)
    runs = [r for r in runs if r[0] >= start]
    start, end = runs[0][0], runs[-1][1]
    main_tid = runs[0][2]
    runtime = list(
        db.execute(
            "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    launches = {
        (tid & ~0xFFFFFF, cid): (a, b, tid, strings[name])
        for a, b, tid, cid, name in runtime
    }
    activities, per_launch = [], collections.defaultdict(list)
    kernel_names = collections.defaultdict(list)
    for table in ("KERNEL", "MEMCPY", "MEMSET"):
        name_column = ",demangledName" if table == "KERNEL" else ""
        for row in db.execute(
            f"SELECT start,end,globalPid,correlationId,streamId{name_column} FROM CUPTI_ACTIVITY_KIND_{table}"
        ):
            a, b, pid, cid, stream = row[:5]
            label = strings[row[5]] if table == "KERNEL" else table
            activities.append((a, b, label, stream))
            launch = launches.get((pid, cid))
            if launch:
                per_launch[(launch[2], launch[0])].append((a, b, label, stream))
            if table == "KERNEL" and a >= start and b <= end:
                kernel_names[label].append(b - a)
    gpu = Intervals(union((a, b) for a, b, *_ in activities))
    natives = Intervals(scopes("hybrid.cpu.native"))
    cpu = Intervals(scopes("CpuNvfp4.run"))
    joins = Intervals(scopes("hybrid.result_join.pending"))
    remaps = Intervals(scopes("HybridCache._remap"))
    dispatches = Intervals(scopes("HybridCache._gpu"))
    waits = Intervals(scopes("hybrid.cpu.input_wait"))
    costs = Intervals(
        sorted(
            (a, b, name) for a, b, _, name in nvtx if name.startswith("hybrid.cpu/M=")
        )
    )
    main_apis = Intervals(
        [
            (a, b, cid, strings[name])
            for a, b, tid, cid, name in runtime
            if tid == main_tid
        ]
    )
    all_osrt = collections.defaultdict(list)
    for a, b, tid, name in db.execute(
        "SELECT start,end,globalTid,nameId FROM OSRT_API"
    ):
        all_osrt[tid].append((a, b, strings[name]))
    osrt = {tid: Intervals(rows) for tid, rows in all_osrt.items()}
    rows = []
    metrics = collections.defaultdict(list)
    for index, (a, b, tid) in enumerate(runs):
        native = natives.inside(a, b)
        wrapper = cpu.inside(a, b)
        join = joins.inside(a, b)
        apis = main_apis.inside(a, b)
        gpu_work = [
            r
            for x, y, _, name in apis
            for r in per_launch.get((tid, x), [])
            if "Qwen38SmallMDirectFp32" in r[2]
        ]
        cost = costs.inside(a, b)
        row = {
            "index": index,
            "start_ns": a,
            "end_ns": b,
            "host_us": (b - a) / 1e3,
            "gpu_idle_us": ((b - a) - gpu.overlap(a, b)) / 1e3,
            "native": [list(r) for r in native],
            "cpu_wrapper": [list(r) for r in wrapper],
            "join": [list(r) for r in join],
            "resident_kernel": [list(r) for r in gpu_work],
            "miss_scope": cost[0][2] if cost else None,
            "remap": remaps.inside(a, b),
            "gpu_dispatch": dispatches.inside(a, b),
        }
        rows.append(row)
        metrics["hybrid_run"].append(b - a)
        metrics["gpu_idle_in_run"].append((b - a) - gpu.overlap(a, b))
        for key, spans in (
            ("native", native),
            ("join", join),
            ("remap", remaps.inside(a, b)),
            ("gpu_dispatch", dispatches.inside(a, b)),
            ("input_wait", waits.inside(a, b)),
        ):
            metrics[key].extend(y - x for x, y, *_ in spans)
            metrics[key + "_gpu_idle"].extend(
                (y - x) - gpu.overlap(x, y) for x, y, *_ in spans
            )
        for label, spans in (
            ("remap", remaps.inside(a, b)),
            ("gpu_dispatch", dispatches.inside(a, b)),
        ):
            for lo, hi, _ in spans:
                scoped_apis = [r for r in apis if lo <= r[0] and r[1] <= hi]
                api_union = union((r[0], r[1]) for r in scoped_apis)
                metrics[label + "_cuda_api"].append(sum(y - x for x, y in api_union))
                metrics[label + "_outside_cuda_api"].append(
                    hi - lo - sum(y - x for x, y in api_union)
                )
                if label == "gpu_dispatch" and gpu_work:
                    launch_calls = [
                        r
                        for r in scoped_apis
                        if any(
                            "Qwen38SmallMDirectFp32" in v[2]
                            for v in per_launch.get((tid, r[0]), [])
                        )
                    ]
                    if len(launch_calls) != 1:
                        raise ValueError(
                            "resident kernel launch is not uniquely correlated"
                        )
                    launch = launch_calls[0]
                    predecessors = [r for r in scoped_apis if r[1] <= launch[0]]
                    if predecessors:
                        # Ignore cuKernelGetName immediately before launch;
                        # retain the preceding actual GPU submission boundary.
                        submissions = [
                            r
                            for r in predecessors
                            if "LaunchKernel" in r[3] or "Memcpy" in r[3]
                        ]
                        if submissions:
                            gap = (submissions[-1][1], launch[0])
                            row["before_resident_launch_gap"] = gap
                            metrics["before_resident_launch_gap"].append(
                                gap[1] - gap[0]
                            )
                    metrics["resident_launch_to_device_start"].append(
                        gpu_work[0][0] - launch[0]
                    )
        row["gpu_activities"] = [
            v for x, y, _, name in apis for v in per_launch.get((tid, x), [])
        ]
        if native:
            nstart, nend, ntid = native[0]
            row["native_cond_waits"] = [
                (x, y)
                for x, y, name in osrt.get(ntid, Intervals([])).inside(nstart, nend)
                if name == "pthread_cond_wait"
            ]
            metrics["native_cond_wait"].extend(
                y - x for x, y in row["native_cond_waits"]
            )
            for jstart, jend, _ in join:
                lo, hi = max(nstart, jstart), min(nend, jend)
                metrics["join_overlapping_native"].append(max(0, hi - lo))
                if hi > lo:
                    metrics["join_native_gpu_idle"].append(
                        hi - lo - gpu.overlap(lo, hi)
                    )
                metrics["join_after_native_return"].append(
                    max(0, jend - max(jstart, nend))
                )
            if gpu_work:
                gend = max(r[1] for r in gpu_work)
                row["cpu_finishes_after_resident_gpu_us"] = (nend - gend) / 1e3
                metrics["native_tail_after_resident_gpu"].append(max(0, nend - gend))
                metrics["cpu_gpu_compute_overlap"].append(
                    sum(max(0, min(nend, y) - max(nstart, x)) for x, y, *_ in gpu_work)
                )
        row["main_osrt"] = [
            (x, y, name)
            for x, y, name in osrt.get(tid, Intervals([])).inside(a, b)
            if y - x > 100000
        ]
    if not all(len(r["resident_kernel"]) == 1 for r in rows):
        raise ValueError(
            "expected one correlated resident kernel per recorded hybrid run"
        )
    if any(a[1] > b[0] for a, b in zip(runs, runs[1:])):
        raise ValueError("overlapping main-thread forwards")
    if any(r[2] != main_tid for r in runs):
        raise ValueError("multiple serving threads require separate analysis")
    busy = gpu.overlap(start, end)
    gaps = [
        (runs[i - 1][1], r[0])
        for i, r in enumerate(runs)
        if i and r[0] > runs[i - 1][1]
    ]
    api_stats = collections.defaultdict(list)
    for a, b, cid, name in main_apis.inside(start, end):
        api_stats[name].append(b - a)
    result = {
        "sqlite": str(path),
        "decode_window_ns": [start, end],
        "main_tid": main_tid,
        "decode_elapsed_ms": (end - start) / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_idle_ms": (end - start - busy) / 1e6,
        "hybrid_layers": len(runs),
        "between_hybrid_runs": summary([b - a for a, b in gaps]),
        "between_hybrid_runs_gpu_idle_ms": sum(
            b - a - gpu.overlap(a, b) for a, b in gaps
        )
        / 1e6,
        "metrics": {k: summary(v) for k, v in metrics.items()},
        "main_cuda_api": {k: summary(v) for k, v in api_stats.items()},
        "kernels": dict(
            sorted(
                ((k, summary(v)) for k, v in kernel_names.items()),
                key=lambda kv: -kv[1]["sum_ms"],
            )
        ),
        "layers": rows,
    }
    idle_parts = {
        name: result["metrics"][name + "_gpu_idle"]["sum_ms"]
        for name in ("remap", "gpu_dispatch", "join")
    }
    idle_parts["other_inside_run"] = result["metrics"]["gpu_idle_in_run"][
        "sum_ms"
    ] - sum(idle_parts.values())
    idle_parts["between_runs"] = result["between_hybrid_runs_gpu_idle_ms"]
    if abs(sum(idle_parts.values()) - result["gpu_idle_ms"]) > 1e-6:
        raise ValueError("idle partition does not reconcile")
    result["disjoint_gpu_idle_ms"] = idle_parts
    if scopes("hybrid.cpu.native"):
        groups = collections.defaultdict(list)
        for row in rows:
            experts = int(row["miss_scope"].split("E=")[1]) if row["miss_scope"] else 0
            groups[experts].append(row)
        result["by_miss_experts"] = [
            {
                "experts": e,
                "count": len(group),
                "host_median_us": statistics.median(r["host_us"] for r in group),
                "native_median_us": statistics.median(
                    (r["native"][0][1] - r["native"][0][0]) / 1000 if r["native"] else 0
                    for r in group
                ),
                "cpu_finishes_later_count": sum(
                    r.get("cpu_finishes_after_resident_gpu_us", -1) > 0 for r in group
                ),
            }
            for e, group in sorted(groups.items())
        ]
        result["paired_calls"] = sum(
            bool(r["native"] and r["resident_kernel"]) for r in rows
        )
        result["cpu_finishes_later_count"] = sum(
            r.get("cpu_finishes_after_resident_gpu_us", -1) > 0 for r in rows
        )
        result["expected_two_dispatches_per_expert"] = sum(
            2 * int(r["miss_scope"].split("E=")[1]) for r in rows if r["native"]
        )
        result["observed_cond_waits"] = sum(
            len(r.get("native_cond_waits", [])) for r in rows
        )
        all_hits = Intervals([(r["start_ns"], r["end_ns"]) for r in groups[0]])
        copy_counts = collections.Counter()
        for a, b, pid, cid, size, kind in db.execute(
            "SELECT start,end,globalPid,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            launch = launches.get((pid, cid))
            if launch and launch[2] == main_tid:
                lo, hi = launch[:2]
                index = bisect.bisect_right(all_hits.starts, lo) - 1
                if index >= 0 and hi <= all_hits.rows[index][1]:
                    copy_counts[kind, size] += 1
        result["all_hit_copy_counts"] = [
            {"kind": kind, "bytes": size, "count": count}
            for (kind, size), count in sorted(copy_counts.items())
        ]
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true", help="Omit per-layer records")
    args = parser.parse_args()
    result = analyze(args.sqlite)
    if args.compact:
        result.pop("layers")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("layers", "kernels", "main_cuda_api")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
