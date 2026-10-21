"""Summarize Nsight SQLite GPU intervals and traced host function ranges."""

import argparse
import json
import sqlite3
from pathlib import Path


def merge_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def union_ns(spans):
    return sum(end - start for start, end in merge_spans(spans))


def intersection_ns(first, second):
    first, second = merge_spans(first), merge_spans(second)
    i, j, total = 0, 0, 0
    while i < len(first) and j < len(second):
        a, b = first[i]
        c, d = second[j]
        total += max(0, min(b, d) - max(a, c))
        if b < d:
            i += 1
        else:
            j += 1
    return total


def summarize(path):
    with sqlite3.connect(path) as db:
        kernels = db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchall()
        copies = db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()
        spans = kernels + copies
        elapsed = max(b for _, b in spans) - min(a for a, _ in spans)
        busy = union_ns(spans)
        apis = db.execute(
            "SELECT s.value,count(*),sum(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId GROUP BY r.nameId ORDER BY sum(r.end-r.start) DESC LIMIT 12"
        ).fetchall()
        functions = db.execute(
            "SELECT coalesce(n.text,s.value),count(*),sum(n.end-n.start) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL GROUP BY coalesce(n.text,s.value) ORDER BY sum(n.end-n.start) DESC LIMIT 16"
        ).fetchall()
        transfers = db.execute(
            "SELECT copyKind,count(*),sum(end-start),sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
        ).fetchall()
        top_kernels = db.execute(
            "SELECT s.value,count(*),sum(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY k.demangledName ORDER BY sum(k.end-k.start) DESC LIMIT 12"
        ).fetchall()

        def ranges(suffix):
            return db.execute(
                "SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s "
                "ON n.textId=s.id WHERE n.end IS NOT NULL "
                "AND coalesce(n.text,s.value) LIKE ?",
                ("%" + suffix,),
            ).fetchall()

        cpu = ranges("CpuNvfp4.run")
        prefill = ranges("HybridCache.prefill")
        return {
            "gpu_activity_span_seconds": elapsed / 1e9,
            "gpu_busy_union_seconds": busy / 1e9,
            "gpu_busy_fraction": busy / elapsed,
            "gpu_kernel_union_seconds": union_ns(kernels) / 1e9,
            "gpu_copy_union_seconds": union_ns(copies) / 1e9,
            "cpu_wrapper_union_seconds": union_ns(cpu) / 1e9,
            "cpu_wrapper_overlap_gpu_kernel_seconds": intersection_ns(cpu, kernels)
            / 1e9,
            "cpu_wrapper_overlap_prefill_host_seconds": intersection_ns(cpu, prefill)
            / 1e9,
            "cuda_api_totals": [
                {"name": n, "calls": c, "sum_seconds": t / 1e9} for n, c, t in apis
            ],
            "nvtx_inclusive_totals": [
                {"name": n, "calls": c, "sum_seconds": t / 1e9} for n, c, t in functions
            ],
            "copies": [
                {"kind": k, "calls": c, "sum_seconds": t / 1e9, "bytes": b}
                for k, c, t, b in transfers
            ],
            "top_kernels": [
                {"name": n, "calls": c, "sum_seconds": t / 1e9}
                for n, c, t in top_kernels
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="label=/path/to/nsight.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {
        label: summarize(path)
        for label, path in (arg.split("=", 1) for arg in args.inputs)
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
