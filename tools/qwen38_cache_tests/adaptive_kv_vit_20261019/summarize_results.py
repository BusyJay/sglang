"""Recompute measured medians and the prefill transport regression evidence."""

import collections
import gzip
import json
import statistics
from pathlib import Path


def prefill_trace(path):
    events = json.load(gzip.open(path, "rt"))["traceEvents"]
    cpu = next(
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("step[EXTEND")
    )
    gpu = [
        e
        for e in events
        if e.get("cat") == "gpu_user_annotation"
        and e.get("args", {}).get("External id") == cpu["args"]["External id"]
    ]
    lo = min(e["ts"] for e in gpu)
    hi = max(e["ts"] + e["dur"] for e in gpu)
    copies = collections.defaultdict(lambda: {"calls": 0, "ms": 0, "bytes": 0})
    for e in events:
        if e.get("cat") == "gpu_memcpy" and lo <= e["ts"] < hi:
            row = copies[e["name"]]
            row["calls"] += 1
            row["ms"] += e["dur"] / 1000
            row["bytes"] += e.get("args", {}).get("bytes", 0)
    return {
        "cpu_ms": cpu["dur"] / 1000,
        "gpu_window_ms": (hi - lo) / 1000,
        "copies": dict(copies),
    }


def main():
    root = Path(__file__).resolve().parent
    result = {"arms": {}, "prefill_diagnostics": {}}
    fields = ("decode_tokens_per_second", "e2e_tokens_per_second", "ttft_s")
    for arm in ("original", "pre-transport-fix", "final"):
        rows = [
            row
            for p in sorted(root.glob(arm + "-round-*.json"))
            for row in json.loads(p.read_text())["cases"]
        ]
        if not rows:
            continue
        result["arms"][arm] = {
            "n": len(rows),
            "median": {k: statistics.median(r[k] for r in rows) for k in fields},
            "by_case": {
                str(case): {
                    k: statistics.median(r[k] for r in rows if r["case"] == case)
                    for k in fields
                }
                for case in range(3)
            },
        }
    for name in ("original-prefill-case2", "prefill-case2", "fixed-prefill-case2"):
        path = root / (name + ".trace.json.gz")
        if path.exists():
            result["prefill_diagnostics"][name] = prefill_trace(path)
    (root / "results-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
