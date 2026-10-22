"""Current LFU n=0/1 decode: classify GPU work, dependencies and counters."""

import collections, gzip, json, statistics
from pathlib import Path

r = Path("/results")
out = {}


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def intersect(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def kind(name):
    if "decode_plan_kernel" in name:
        return "planner"
    if "mailbox_fill_kernel" in name:
        return "fill"
    if "hybrid_merge_kernel" in name:
        return "merge"
    if "MoEMicroKernelW4A16SmallMDirect" in name or "Qwen38SmallMDirectFp32" in name:
        return "routed"
    if "DeviceGemmMxfp8Gemm" in name:
        return "mxfp8_gemm"
    if "gemvx::kernel" in name:
        return "bf16_gemv"
    if "cutlass_80_wmma_tensorop_bf16" in name:
        return "bf16_gemm"
    if "_hc_mix_persistent_kernel" in name:
        return "hc_mix"
    return "other"


for mode in ["n0", "n1"]:
    with gzip.open(r / f"profile-{mode}/decode-only.json.gz", "rt") as f:
        trace = json.load(f)
    gpu = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
    ]
    groups = collections.defaultdict(list)
    for e in gpu:
        groups[kind(e["name"])].append(e)
    total = sum(e.get("dur", 0) for e in gpu)
    summary = {}
    for key, es in groups.items():
        values = [e["dur"] for e in es]
        summary[key] = dict(
            count=len(values),
            total_ms=sum(values) / 1000,
            share_pct=100 * sum(values) / total,
            median_us=statistics.median(values),
            p95_us=sorted(values)[int(len(values) * 0.95)],
            max_us=max(values),
        )
    fills = union([(e["ts"], e["ts"] + e["dur"]) for e in groups["fill"]])
    dense = union([(e["ts"], e["ts"] + e["dur"]) for e in groups["mxfp8_gemm"]])
    overlap = intersect(fills, dense)
    fillactive = [e["dur"] for e in groups["fill"] if e["dur"] > 5]
    stages = {
        k: sorted(groups[k], key=lambda e: e["ts"])
        for k in ["planner", "fill", "routed", "merge"]
    }
    waits = []
    if len(stages["fill"]) == len(stages["routed"]):
        waits = [
            g["ts"] - (f["ts"] + f["dur"])
            for f, g in zip(stages["fill"], stages["routed"])
        ]
    logs = (r / f"profile-{mode}/server.log").read_text().splitlines()
    periodic = []
    inside = False
    for line in logs:
        if "Profiling starts." in line:
            inside = True
        if "Stop profiling..." in line:
            inside = False
        if "Qwen38ExpertCacheStats" in line:
            data = json.loads(line[line.index("{") :])
            periodic.append(
                dict(
                    inside_profile=inside,
                    calls=data["run_invocations"],
                    hybrid=data.get("layered_cpu_hybrid"),
                    copy_bytes=data.get("decode_device_gate_copy_bytes"),
                    hit_rate=data.get("cache_hit_rate"),
                )
            )
    inprofile = [p for p in periodic if p["inside_profile"]]
    delta = None
    if len(inprofile) >= 2:
        b, a = inprofile[0], inprofile[-1]
        delta = {
            "runtime_calls": a["calls"] - b["calls"],
            "hybrid": {k: a["hybrid"][k] - b["hybrid"][k] for k in a["hybrid"]},
            "copy_bytes": a["copy_bytes"] - b["copy_bytes"],
        }
    out[mode] = dict(
        groups=summary,
        total_gpu_activity_ms=total / 1000,
        fill_mxfp8_overlap_ms=overlap / 1000,
        fill_mxfp8_overlap_pct=100 * overlap / (sum(b - a for a, b in fills)),
        slow_fill_count=len(fillactive),
        slow_fill_median_us=statistics.median(fillactive),
        fill_to_routed_gap_median_us=statistics.median(waits) if waits else None,
        fill_to_routed_gap_p95_us=sorted(waits)[int(len(waits) * 0.95)]
        if waits
        else None,
        profile_counter_window=delta,
        periodic=periodic,
    )
    (r / "current-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        mode,
        json.dumps({k: v for k, v in out[mode].items() if k not in ["periodic"]}),
        flush=True,
    )
