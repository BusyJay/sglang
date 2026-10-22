"""Analyze full steady decode and export small middle-window traces for triage."""

import collections, gzip, json, statistics
from pathlib import Path

r = Path("/results")
result = {}


def kind(name):
    for key, needle in [
        ("planner", "decode_plan_kernel"),
        ("fill", "mailbox_fill_kernel"),
        ("merge", "hybrid_merge_kernel"),
        ("mxfp8", "DeviceGemmMxfp8Gemm"),
        ("gemv", "gemvx::kernel"),
        ("bf16_gemm", "cutlass_80_wmma_tensorop_bf16"),
        ("hc_mix", "_hc_mix_persistent_kernel"),
    ]:
        if needle in name:
            return key
    if "MoEMicroKernelW4A16SmallMDirect" in name or "Qwen38SmallMDirectFp32" in name:
        return "routed"
    return "other"


def united(es):
    out = []
    for a, b in sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in es):
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def overlap(a, b):
    i = j = 0
    t = 0
    while i < len(a) and j < len(b):
        t += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


for mode in ["n0", "n1"]:
    log = (r / f"active-{mode}/server.log").read_text().splitlines()
    sections = []
    inside = False
    stats = []
    for line in log:
        if "Profiling starts." in line:
            inside = True
            stats = []
        if inside and "Qwen38ExpertCacheStats" in line:
            d = json.loads(line[line.index("{") :])
            stats.append(d)
        if "Stop profiling..." in line:
            inside = False
            if len(stats) >= 2:
                a, b = stats[0], stats[-1]
                sections.append(
                    dict(
                        runtime_calls=b["run_invocations"] - a["run_invocations"],
                        copy_bytes=b["decode_device_gate_copy_bytes"]
                        - a["decode_device_gate_copy_bytes"],
                        **{
                            k: b["layered_cpu_hybrid"][k] - a["layered_cpu_hybrid"][k]
                            for k in b["layered_cpu_hybrid"]
                        },
                    )
                )
    result[mode] = {}
    for case, index in [("short", 0), ("natural", 1)]:
        root = r / f"active-{mode}" / case
        p = next(p for p in root.glob("*.json.gz") if p.name != "window.json.gz")
        with gzip.open(p, "rt") as f:
            trace = json.load(f)
        events = trace["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        plans = sorted(
            [e for e in kernels if "decode_plan_kernel" in e["name"]],
            key=lambda e: e["ts"],
        )
        start = plans[48]["ts"]
        end = max(e["ts"] + e["dur"] for e in kernels)
        gpu = [
            e
            for e in events
            if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
            and start <= e.get("ts", -1) < end
        ]
        groups = collections.defaultdict(list)
        for e in gpu:
            groups[kind(e["name"])].append(e)
        total = sum(e["dur"] for e in gpu)
        counts = len(groups["planner"])
        summary = {}
        for key, es in groups.items():
            dur = [e["dur"] for e in es]
            summary[key] = dict(
                count=len(es),
                total_ms=sum(dur) / 1000,
                us_per_layer=sum(dur) / counts,
                share_pct=100 * sum(dur) / total,
                median_us=statistics.median(dur),
                p95_us=sorted(dur)[int(len(dur) * 0.95)],
                max_us=max(dur),
            )
        fs = united(groups["fill"])
        ds = united(groups["mxfp8"])
        ov = overlap(fs, ds)
        slowmerge = [e["dur"] for e in groups["merge"] if e["dur"] > 10]
        activefill = [e["dur"] for e in groups["fill"] if e["dur"] > 5]
        begin_step = min(64, len(plans) // 48 // 3)
        last_step = min(begin_step + 24, len(plans) // 48 - 1)
        lo = plans[48 * begin_step]["ts"]
        hi = plans[48 * last_step]["ts"]
        slim = [
            e
            for e in events
            if e.get("ph") == "M"
            or (lo <= e.get("ts", -1) < hi and e.get("cat") != "python_function")
        ]
        trace["traceEvents"] = slim
        with gzip.open(root / "window.json.gz", "wt") as f:
            json.dump(trace, f)
        result[mode][case] = dict(
            trace=str(p),
            retained_decode_layers=counts,
            groups=summary,
            total_gpu_activity_ms=total / 1000,
            fill_mxfp8_overlap_pct=100 * ov / sum(b - a for a, b in fs),
            slow_merge_count=len(slowmerge),
            slow_merge_ms=sum(slowmerge) / 1000,
            slow_fill_count=len(activefill),
            slow_fill_median_us=statistics.median(activefill),
            counter_window=sections[index] if len(sections) > index else None,
        )
        (r / "active-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(mode, case, "analyzed", flush=True)
        del trace, events, kernels, plans, gpu, groups, slim
