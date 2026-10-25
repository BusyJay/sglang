"""Extract the largest GPU idle transition, including adjacent large gaps."""

import argparse
import collections
import gzip
import json
from pathlib import Path

import ijson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    gaps = metrics["largest_gpu_gaps_ms"]
    lo, hi, _ = gaps[0]
    for a, b, ms in sorted(gaps):
        if ms >= 5 and b >= lo - 1000 and a <= hi + 1000:
            lo, hi = min(lo, a), max(hi, b)
    lo, hi = lo - 10000, hi + 10000
    kept = []
    top = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    overhead = []
    with gzip.open(args.trace, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") == "M":
                kept.append(event)
                continue
            if event.get("ph") != "X" or "dur" not in event:
                continue
            a, b = event["ts"], event["ts"] + event["dur"]
            overlap = min(b, hi) - max(a, lo)
            if overlap <= 0 or event.get("cat") == "Trace":
                continue
            kept.append(event)
            category = event.get("cat", "")
            value = top[category][event.get("name", "")]
            value[0] += 1
            value[1] += overlap / 1000
            if category == "overhead":
                overhead.append(event)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir / "rebalance-transition.trace.json.gz", "wt") as f:
        json.dump({"traceEvents": kept, "displayTimeUnit": "ms"}, f)
    summary = {
        "window_us": [lo, hi],
        "events": len(kept),
        "overhead_events": overhead,
        "category_top": {
            category: [
                {"name": n, "count": v[0], "clipped_duration_ms": v[1]}
                for n, v in sorted(rows.items(), key=lambda kv: -kv[1][1])[:20]
            ]
            for category, rows in top.items()
        },
    }
    (args.output_dir / "transition-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("transition saved", len(kept), "events", flush=True)


if __name__ == "__main__":
    main()
