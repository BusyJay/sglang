"""Remove initial prefill and first decode step; summarize and retain Chrome traces."""

import collections, gzip, json, statistics
from pathlib import Path

R = Path("/results")
report = {}
for mode in ["n0", "n1"]:
    path = next((R / f"profile-{mode}/trace").rglob("*.json.gz"))
    with gzip.open(path, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    plans = sorted(
        [e for e in kernels if "decode_plan_kernel" in e["name"]], key=lambda e: e["ts"]
    )
    start = plans[48]["ts"]
    end = max(e["ts"] + e.get("dur", 0) for e in kernels)
    selected = [
        e
        for e in events
        if e.get("ph") == "M" or (e.get("ts", -1) >= start and e.get("ts", -1) < end)
    ]
    selected = [
        e
        for e in selected
        if e.get("cat") != "python_function"
        or (
            (
                "python/sglang/" in e.get("name", "")
                or "torch/cuda/graphs.py" in e.get("name", "")
            )
            and "qwen38_cache_stats_format.py" not in e.get("name", "")
        )
    ]
    trace["traceEvents"] = selected
    output = R / f"profile-{mode}/decode-only.json.gz"
    with gzip.open(output, "wt") as f:
        json.dump(trace, f)
    gpu = [
        e for e in selected if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
    ]
    groups = collections.defaultdict(list)
    for e in gpu:
        groups[e["name"]].append(e.get("dur", 0))
    total = sum(sum(v) for v in groups.values())
    summary = [
        dict(
            name=k,
            count=len(v),
            total_us=sum(v),
            share=sum(v) / total,
            median_us=statistics.median(v),
            p95_us=sorted(v)[int(len(v) * 0.95)],
            max_us=max(v),
        )
        for k, v in groups.items()
    ]
    summary.sort(key=lambda x: -x["total_us"])
    intervals = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu)
    busy = 0
    last = start
    for lo, hi in intervals:
        busy += max(0, hi - max(lo, last))
        last = max(last, hi)
    report[mode] = dict(
        source=str(path),
        crop_start_us=start,
        crop_end_us=end,
        planner_count=len(plans),
        retained_planners=sum("decode_plan_kernel" in e["name"] for e in gpu),
        span_us=end - start,
        busy_us=busy,
        kernels=summary,
    )
    (R / "decode-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(mode, "done", flush=True)
