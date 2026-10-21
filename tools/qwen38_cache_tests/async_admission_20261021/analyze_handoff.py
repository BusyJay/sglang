"""Reproduce the handoff's host, GPU, wait and miss-count timing tables.

Consumes an Nsight Systems SQLite export; no model, GPU or Python packages needed.
Intervals are half-open. Inclusive host times and cumulative GPU times must not
be added across threads/streams. Only explicit same-thread children are removed
from exclusive host time. Correlation IDs attribute GPU work to its launch scope.
"""

import argparse
import bisect
import collections
import json
import re
import sqlite3
from pathlib import Path


def merge(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(spans):
    return sum(end - start for start, end in merge(spans))


def intersect(first, second):
    first, second = merge(first), merge(second)
    i = j = total = 0
    while i < len(first) and j < len(second):
        a, b = first[i]
        c, d = second[j]
        total += max(0, min(b, d) - max(a, c))
        if b < d:
            i += 1
        else:
            j += 1
    return total


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def stats(spans):
    values = [end - start for start, end in spans]
    return {
        "calls": len(values),
        "sum_seconds": sum(values) / 1e9,
        "union_seconds": length(spans) / 1e9,
        **{
            key: percentile(values, fraction) / 1e3 if values else None
            for key, fraction in (("p50_us", 0.5), ("p95_us", 0.95), ("p99_us", 0.99))
        },
        "max_us": max(values) / 1e3 if values else None,
    }


class DisjointRanges:
    def __init__(self, spans):
        self.spans = sorted(spans)
        self.starts = [start for start, _, _ in self.spans]
        if any(a[1] > b[0] for a, b in zip(self.spans, self.spans[1:])):
            raise ValueError("expected disjoint outer scopes")

    def find(self, start, end):
        index = bisect.bisect_right(self.starts, start) - 1
        if index >= 0 and self.spans[index][1] >= end:
            return self.spans[index][2]
        return None


def analyze(path, layers=48):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = [
        (start, end, tid, text or strings.get(text_id, ""))
        for start, end, tid, text, text_id in db.execute(
            "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
        if end > start
    ]
    ranges = [r for r in ranges if r[3].startswith(("sglang.", "hybrid."))]
    by_name, by_tid = collections.defaultdict(list), collections.defaultdict(list)
    for start, end, tid, name in ranges:
        by_name[name].append((start, end))
        by_tid[tid].append((start, end, name))
    roles = collections.defaultdict(set)
    for name, role in (
        ("HybridCache.run", "main"),
        ("HybridCache.prefill", "main"),
        ("HybridCache._submit_ready", "manager"),
        ("HybridCache._copy", "manager"),
        ("GlobalLfu.plan", "manager"),
        ("CpuNvfp4.run", "cpu"),
        ("hybrid.cpu.native", "cpu"),
    ):
        for _, _, tid, label in ranges:
            if label.endswith(name):
                roles[tid].add(role)
    role = {tid: "+".join(sorted(names)) for tid, names in roles.items()}
    outer = {}
    specific = {}
    for tid, rows in by_tid.items():
        outer[tid] = DisjointRanges(
            [
                (a, b, "decode" if label.endswith(".run") else "prefill")
                for a, b, label in rows
                if label.endswith(("HybridCache.run", "HybridCache.prefill"))
            ]
        )
        specific[tid] = []
        for suffix, category in (
            ("hybrid.prefill.ids_readback", "prefill.ids_readback"),
            ("HybridCache._remap", "decode.remap"),
            ("HybridCache._gpu", "decode.gpu_dispatch"),
            ("hybrid.cpu.native", "native"),
            ("hybrid.cpu.input_wait", "input_wait"),
            ("hybrid.mailbox.flush_stats", "stats.mailbox"),
            ("hybrid.stats_snapshot", "stats.snapshot"),
        ):
            matches = [
                (a, b, category) for a, b, label in rows if label.endswith(suffix)
            ]
            if matches:
                specific[tid].append(DisjointRanges(matches))

    def owner(tid, start, end):
        for index in specific.get(tid, ()):
            found = index.find(start, end)
            if found:
                return f"{role.get(tid, 'other')}:{found}"
        phase = outer[tid].find(start, end) if tid in outer else None
        return f"{role.get(tid, 'other')}:{phase or 'outside_hybrid'}"

    runtime = list(
        db.execute(
            "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    launch = {}
    apis, api_by_owner = collections.defaultdict(list), collections.defaultdict(list)
    for start, end, tid, correlation, name_id in runtime:
        name = strings[name_id]
        category = owner(tid, start, end)
        # globalPid is globalTid with the per-process thread field cleared.
        launch[(tid & ~0xFFFFFF, correlation)] = category
        apis[name].append((start, end))
        api_by_owner[(category, name)].append((start, end))

    kernels = list(
        db.execute(
            "SELECT start,end,globalPid,correlationId,shortName,demangledName,streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    copies = list(
        db.execute(
            "SELECT start,end,globalPid,correlationId,copyKind,bytes,streamId "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    )
    gpu = merge([(a, b) for a, b, *_ in kernels + copies])
    kernel_spans = merge([(a, b) for a, b, *_ in kernels])
    copy_spans = merge([(a, b) for a, b, *_ in copies])
    first, last = gpu[0][0], gpu[-1][1]
    elapsed = last - first
    busy = length(gpu)
    kernel_names, gpu_by_owner = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    for start, end, pid, correlation, short, full, stream in kernels:
        kernel_names[(strings[short], strings[full])].append((start, end))
        gpu_by_owner[(launch.get((pid, correlation), "uncorrelated"), "kernel")].append(
            (start, end)
        )
    copy_kinds = collections.defaultdict(list)
    copy_bytes = collections.Counter()
    for start, end, pid, correlation, kind, size, stream in copies:
        gpu_by_owner[(launch.get((pid, correlation), "uncorrelated"), "copy")].append(
            (start, end)
        )
        copy_kinds[kind].append((start, end))
        copy_bytes[kind] += size

    host = {}
    for name, spans in by_name.items():
        row = stats(spans)
        union = length(spans)
        overlap = intersect(spans, gpu)
        row.update(
            gpu_busy_overlap_seconds=overlap / 1e9,
            gpu_kernel_overlap_seconds=intersect(spans, kernel_spans) / 1e9,
            gpu_idle_overlap_seconds=(union - overlap) / 1e9,
        )
        host[name] = row
    # Same-thread nested scopes only. Thread/stream overlap is never subtracted
    # from a host function's wall duration to pretend it is CPU execution time.
    exclusive = collections.Counter()
    for tid, rows in by_tid.items():
        rows.sort()
        starts = [a for a, _, _ in rows]
        for index, (start, end, name) in enumerate(rows):
            bound = bisect.bisect_left(starts, end)
            children = [(a, b) for a, b, _ in rows[index + 1 : bound] if b <= end]
            exclusive[name] += end - start - length(children)
    for name, value in exclusive.items():
        host[name]["exclusive_seconds"] = value / 1e9

    native = [(a, b) for a, b, _, name in ranges if name == "hybrid.cpu.native"]
    native.sort()
    native_starts = [a for a, _ in native]
    forwards = sorted(
        (a, b) for a, b, _, name in ranges if name.endswith("HybridCache.run")
    )
    if len(forwards) % layers:
        raise ValueError("capture does not contain complete model forwards")
    forward_index = DisjointRanges(
        [(a, b, i % layers) for i, (a, b) in enumerate(forwards)]
    )
    layer_cpu = collections.defaultdict(list)
    layer_native = collections.defaultdict(list)
    layer_experts = collections.Counter()
    layer_join = collections.defaultdict(list)
    for start, end in by_name.get("hybrid.result_join.pending", ()):
        layer = forward_index.find(start, end)
        if layer is None:
            raise ValueError("result join outside a target forward")
        layer_join[layer].append((start, end))
    cpu_bins = []
    for name, spans in by_name.items():
        match = re.fullmatch(r"hybrid.cpu/M=(\d+)/E=(\d+)", name)
        if match:
            row = dict(tokens=int(match[1]), miss_experts=int(match[2]), **stats(spans))
            row["native_seconds"] = intersect(spans, native) / 1e9
            cpu_bins.append(row)
            for start, end in spans:
                layer = forward_index.find(start, end)
                if layer is None:
                    raise ValueError("CPU miss call outside its main-thread forward")
                layer_cpu[layer].append((start, end))
                layer_experts[layer] += int(match[2])
                index = bisect.bisect_left(native_starts, start)
                if index >= len(native) or native[index][1] > end:
                    raise ValueError("missing native call inside CPU miss probe")
                layer_native[layer].append(native[index])
    osrt = collections.defaultdict(list)
    for start, end, tid, name_id in db.execute(
        "SELECT start,end,globalTid,nameId FROM OSRT_API"
    ):
        osrt[(owner(tid, start, end), strings[name_id])].append((start, end))
    db.close()
    gpu_sum = sum(b - a for a, b, *_ in kernels)
    result = {
        "sqlite": str(path),
        "gpu_window": {
            "start_ns": first,
            "end_ns": last,
            "seconds": elapsed / 1e9,
            "kernel_copy_union_seconds": busy / 1e9,
            "idle_seconds": (elapsed - busy) / 1e9,
            "active_fraction": busy / elapsed,
            "kernel_union_seconds": length(kernel_spans) / 1e9,
            "copy_union_seconds": length(copy_spans) / 1e9,
            "kernel_cumulative_seconds": gpu_sum / 1e9,
        },
        "host_functions": host,
        "cpu_bins": sorted(cpu_bins, key=lambda r: (r["tokens"], r["miss_experts"])),
        "cpu_by_layer": [
            dict(
                layer=layer,
                mean_miss_experts=layer_experts[layer] / len(spans),
                native=stats(layer_native[layer]),
                pending_join=stats(layer_join[layer]),
                **stats(spans),
            )
            for layer, spans in sorted(layer_cpu.items())
        ],
        "layer_assignment": "main-thread HybridCache.run ordinal modulo layers; capture starts between requests",
        "layers": layers,
        "cuda_apis": {name: stats(spans) for name, spans in apis.items()},
        "cuda_apis_by_owner": [
            dict(
                owner=category,
                name=name,
                gpu_busy_overlap_seconds=(
                    intersect(spans, gpu) / 1e9
                    if "Synchronize" in name or "EventDestroy" in name
                    else None
                ),
                gpu_idle_overlap_seconds=(
                    (length(spans) - intersect(spans, gpu)) / 1e9
                    if "Synchronize" in name or "EventDestroy" in name
                    else None
                ),
                **stats(spans),
            )
            for (category, name), spans in sorted(api_by_owner.items())
        ],
        "gpu_by_launch_owner": [
            dict(owner=category, kind=kind, **stats(spans))
            for (category, kind), spans in sorted(gpu_by_owner.items())
        ],
        "copies": [
            dict(kind=k, bytes=copy_bytes[k], **stats(v)) for k, v in copy_kinds.items()
        ],
        "top_kernels": sorted(
            [
                dict(
                    short_name=short,
                    full_name=full,
                    share=sum(b - a for a, b in spans) / gpu_sum,
                    **stats(spans),
                )
                for (short, full), spans in kernel_names.items()
            ],
            key=lambda r: -r["sum_seconds"],
        )[:30],
        "osrt_by_owner": sorted(
            [
                dict(owner=category, name=name, **stats(spans))
                for (category, name), spans in osrt.items()
            ],
            key=lambda r: -r["sum_seconds"],
        )[:40],
    }
    assert busy <= elapsed
    assert (
        abs(
            result["gpu_window"]["kernel_copy_union_seconds"]
            + result["gpu_window"]["idle_seconds"]
            - elapsed / 1e9
        )
        < 1e-8
    )
    assert all(v["exclusive_seconds"] >= -1e-8 for v in host.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="label=/path/to/export.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    args = parser.parse_args()
    output = {
        label: analyze(Path(path), args.layers)
        for label, path in (s.split("=", 1) for s in args.inputs)
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
