"""Summarize the committed capture without conflating mapped loads with DMA."""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "nsight_optimization_20261021")
)
from summarize_nsight import intersection_ns, merge_spans, union_ns


def family(name, short):
    if "DeviceGemmMxfp8" in name:
        return "Dense MXFP8 GEMM"
    if "GroupProblemShape" in name:
        return "Grouped FP4 GEMM"
    if "internal::gemvx" in name:
        return "GEMV"
    if short == "Kernel2":
        return "BF16 WMMA GEMM"
    if "Qwen38SmallMDirectFp32" in name:
        return "B12X FP32 routed MoE"
    return short


def analyze(path):
    db = sqlite3.connect(path)
    kernels = db.execute(
        "SELECT k.start,k.end,s.value,ss.value,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName JOIN StringIds ss ON ss.id=k.shortName ORDER BY k.start"
    ).fetchall()
    copies = db.execute(
        "SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ).fetchall()
    plans = [(a, b) for a, b, _, short, _ in kernels if short == "hybrid_plan_kernel"]
    scatter_end = max(
        b for _, b, _, short, _ in kernels if short == "packed_scatter_kernel"
    )
    decode_start = next(a for a, _ in plans if a > scatter_end)
    tail_end = max(b for a, b in plans if b < scatter_end)
    first = min(a for a, *_ in kernels + copies)
    last = max(b for _, b, *_ in kernels + copies)
    windows = {
        "all": (first, last),
        "long_prefill_proxy": (first, plans[0][0]),
        "short_prefill_proxy": (tail_end, decode_start),
        "sustained_decode": (decode_start, last),
    }
    results = {
        "stage_method": "Prefill proxies end at the next hybrid planner. Sustained decode starts at the first planner after the final packed_scatter; excludes decode request prefill.",
        "windows": {},
    }
    for label, (lo, hi) in windows.items():
        ks = [
            (max(lo, a), min(hi, b), name, short, stream)
            for a, b, name, short, stream in kernels
            if a < hi and b > lo
        ]
        cs = [
            (max(lo, a), min(hi, b), kind, size)
            for a, b, kind, size in copies
            if a < hi and b > lo
        ]
        spans = [(a, b) for a, b, *_ in ks]
        copyspans = [(a, b) for a, b, *_ in cs]
        groups = defaultdict(list)
        for a, b, name, short, _ in ks:
            groups[family(name, short)].append((a, b))
        total = sum(b - a for a, b in spans)
        rows = []
        for name, intervals in groups.items():
            durations = sorted(b - a for a, b in intervals)
            t = sum(durations)
            other = [(a, b) for a, b, n, s, _ in ks if family(n, s) != name]
            rows.append(
                {
                    "name": name,
                    "calls": len(intervals),
                    "ms": t / 1e6,
                    "share_pct": 100 * t / total,
                    "mean_us": t / len(intervals) / 1e3,
                    "p50_us": durations[len(durations) // 2] / 1e3,
                    "p95_us": durations[
                        min(len(durations) - 1, int(len(durations) * 0.95))
                    ]
                    / 1e3,
                    "overlap_other_kernels_ms": intersection_ns(intervals, other) / 1e6,
                }
            )
        busy = union_ns(spans + copyspans)
        merged = merge_spans(spans + copyspans)
        gaps = sorted(
            [
                ((b - a) / 1e6, a / 1e9, b / 1e9)
                for (_, a), (b, _) in zip(merged, merged[1:])
            ],
            reverse=True,
        )
        apis = db.execute(
            "SELECT s.value,count(*),sum(min(r.end,?)-max(r.start,?))/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE r.start<? AND r.end>? GROUP BY r.nameId ORDER BY 3 DESC LIMIT 8",
            (hi, lo, hi, lo),
        ).fetchall()
        results["windows"][label] = {
            "start_s": lo / 1e9,
            "end_s": hi / 1e9,
            "span_ms": (hi - lo) / 1e6,
            "kernel_sum_ms": total / 1e6,
            "kernel_union_ms": union_ns(spans) / 1e6,
            "gpu_busy_pct": 100 * busy / (hi - lo),
            "gpu_idle_ms": ((hi - lo) - busy) / 1e6,
            "copy_kernel_overlap_ms": intersection_ns(copyspans, spans) / 1e6,
            "kernel_concurrency_ms": (total - union_ns(spans)) / 1e6,
            "kernels": sorted(rows, key=lambda x: -x["ms"]),
            "copies": [
                {
                    "kind": kind,
                    "calls": sum(k == kind for _, _, k, _ in cs),
                    "bytes": sum(size for _, _, k, size in cs if k == kind),
                    "ms": sum(b - a for a, b, k, _ in cs if k == kind) / 1e6,
                }
                for kind in sorted({k for _, _, k, _ in cs})
            ],
            "top_gpu_idle_gaps_ms_start_s_end_s": gaps[:8],
            "cuda_api_inclusive_ms": [
                {"name": n, "calls": c, "ms": t} for n, c, t in apis
            ],
        }
    rows = db.execute(
        "SELECT m.bytes,count(*),sum(r.end-r.start)/1e6,sum(m.end-m.start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId "
        "JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=r.correlationId "
        "AND m.globalPid=r.globalTid/16777216*16777216 "
        "WHERE r.start>=? AND s.value LIKE 'cudaMemcpyAsync%' AND m.copyKind=2 "
        "GROUP BY m.bytes ORDER BY 3 DESC",
        (decode_start,),
    ).fetchall()
    results["decode_dtoh_api_by_size"] = [
        {
            "bytes_per_copy": n,
            "calls": count,
            "host_api_ms": host,
            "device_copy_ms": device,
        }
        for n, count, host, device in rows
    ]
    results["cpu_native_overlap"] = (
        "unavailable: capture has no CpuNvfp4.run duration ranges"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.sqlite), indent=2) + "\n")
