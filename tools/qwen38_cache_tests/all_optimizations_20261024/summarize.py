"""Summarize the retained microbenchmarks and completed serving A/B runs."""

import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
serving_root = root.parent / "csf_defaults_20261023/all_optimizations_20261024"
report = {
    "baseline_commit": "5e8b2e3d13",
    "hc": [],
    "mxfp8": [],
    "fill": [],
    "serving": {},
}
for line in (root / "hc-benchmark.jsonl").read_text().splitlines():
    case = json.loads(line)
    before = statistics.median(case["persistent_us"])
    after = statistics.median(case["split_k_us"])
    report["hc"].append(
        dict(
            rows=case["rows"],
            dtype=case["dtype"],
            before_us=before,
            after_us=after,
            latency_change_percent=100 * (after / before - 1),
        )
    )
for line in (root / "mxfp8-benchmark.txt").read_text().splitlines():
    if "|" not in line or not line.split()[0].isdigit():
        continue
    labels, times, _ = line.split("|")
    _, role, rows = labels.split()
    before, tile16, tile32, tile64, tile256 = map(float, times.split())
    report["mxfp8"].append(
        dict(
            role=role,
            rows=int(rows),
            cutlass_us=before,
            default_32x128_us=tile32,
            default_speedup=before / tile32,
            candidate_tiles_us={
                "16x128": tile16,
                "32x128": tile32,
                "64x128": tile64,
                "32x256": tile256,
            },
        )
    )
fill = json.loads((root / "fill-benchmark.json").read_text())
assert fill["status"] == "passed"
for count in (0, 1, 2, 3, 4, 6, 8, 10, 20, 30):
    times = {
        variant: statistics.median(
            item["median_gpu_ms"]
            for item in fill["reports"]
            if item["misses"] == count and item["variant"] == variant
        )
        for variant in ("mapped_adaptive", "mapped_candidate")
    }
    before, after = times["mapped_adaptive"], times["mapped_candidate"]
    report["fill"].append(
        dict(
            misses=count,
            before_ms=before,
            after_ms=after,
            latency_change_percent=100 * (after / before - 1),
        )
    )
for variant in ("baseline", "hc_cta", "all", "baseline_repeat"):
    result = json.loads((serving_root / variant / "performance.json").read_text())
    assert result["status"] == "passed", (variant, result["status"])
    cases = result["modes"]["default"]["cases"]
    chat = json.loads((serving_root / variant / "chat-correctness.json").read_text())
    assert chat["status"] == "passed", (variant, chat)
    report["serving"][variant] = dict(
        short_median_tokens_per_second=statistics.median(
            c["decode_tokens_per_second"]
            for c in cases
            if c["name"].startswith("short_")
        ),
        exact_answer_tests_passed=len(chat["cases"]),
        cases=[
            {
                key: c[key]
                for key in (
                    "name",
                    "prompt_tokens",
                    "output_tokens",
                    "ttft_seconds",
                    "e2e_seconds",
                    "decode_tokens_per_second",
                )
            }
            | {
                key: c["meta_info"].get(key)
                for key in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct")
            }
            for c in cases
        ],
    )
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["serving"], indent=2))
