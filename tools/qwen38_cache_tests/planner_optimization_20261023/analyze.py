"""Summarize paired throughput, exact outputs, state hashes and kernel timings."""

import json, statistics
from pathlib import Path

r = Path(__file__).resolve().parent
report = {"serving": {}, "planner": {}}
for before, after in [
    ("before-n1-full", "after-n1-full"),
    ("before-n0-short", "after-n0-short"),
    ("before-gpu-short", "after-gpu-short"),
    ("before-n1-repeat", "after-n1-repeat"),
    ("before-n1-no-spec", "after-n1-no-spec"),
]:
    paths = [r / x / "performance.json" for x in [before, after]]
    if not all(p.exists() for p in paths):
        continue
    arms = [json.loads(p.read_text()) for p in paths]
    if not all(a["status"] == "passed" for a in arms):
        continue
    cases = [{x["name"]: x for x in a["modes"]["default"]["cases"]} for a in arms]
    summary = {}
    for label, values in zip(["before", "after"], cases):
        summary[label] = {
            "short_median_tok_s": statistics.median(
                values[f"short_{i}"]["decode_tokens_per_second"] for i in range(3)
            ),
            "cases": {
                name: {
                    k: v[k]
                    for k in [
                        "decode_tokens_per_second",
                        "after_64_tokens_per_second",
                        "ttft_seconds",
                        "meta_info",
                    ]
                }
                for name, v in values.items()
            },
        }
    summary["short_change_pct"] = 100 * (
        summary["after"]["short_median_tok_s"] / summary["before"]["short_median_tok_s"]
        - 1
    )
    summary["response_text_equal"] = {
        k: cases[0][k]["response_text"] == cases[1][k]["response_text"]
        for k in cases[0]
    }
    report["serving"][after] = summary
for first, second in [
    ("before-n1-full", "before-n1-repeat"),
    ("after-n1-full", "after-n1-repeat"),
]:
    ps = [r / x / "performance.json" for x in [first, second]]
    if all(p.exists() for p in ps):
        arms = [json.loads(p.read_text()) for p in ps]
        if all(a["status"] == "passed" for a in arms):
            cs = [{c["name"]: c for c in a["modes"]["default"]["cases"]} for a in arms]
            report.setdefault("same_revision_text_equal", {})[
                first + " / " + second
            ] = {
                k: cs[0][k]["response_text"] == cs[1][k]["response_text"] for k in cs[1]
            }
paths = [r / f"{x}-planner.json" for x in ["before", "after"]]
if all(p.exists() for p in paths):
    arms = [json.loads(p.read_text()) for p in paths]
    if all(x.get("status") == "passed" for x in arms):
        pairs = []
        for b, a in zip(arms[0]["cases"], arms[1]["cases"]):
            key = ["slots", "tokens", "kind", "n"]
            assert all(b[k] == a[k] for k in key)
            pairs.append(
                {
                    **{k: a[k] for k in key},
                    "state_equal": b["state_sha256"] == a["state_sha256"],
                    "before_graph_us": b["graph_median_us"],
                    "after_graph_us": a["graph_median_us"],
                    "change_pct": 100
                    * (a["graph_median_us"] / b["graph_median_us"] - 1),
                }
            )
        report["planner"]["pairs"] = pairs
        report["planner"]["all_states_equal"] = all(x["state_equal"] for x in pairs)
        report["planner"]["chains"] = [
            dict(
                slots=b["slots"],
                n=b["n"],
                cpu_total=b["cpu_total"],
                state_equal=b["state_sha256"] == a["state_sha256"],
            )
            for b, a in zip(arms[0].get("chains", []), arms[1].get("chains", []))
        ]
        report["planner"]["all_chains_equal"] = bool(
            report["planner"]["chains"]
        ) and all(x["state_equal"] for x in report["planner"]["chains"])
for variant in ["before", "after"]:
    profiles = {}
    for p in r.glob(f"{variant}-*.trace.json"):
        events = json.loads(p.read_text())["traceEvents"]
        dur = [
            e["dur"]
            for e in events
            if e.get("cat") == "kernel" and "decode_plan_kernel" in e.get("name", "")
        ]
        if dur:
            profiles[p.name] = {
                "median_us": statistics.median(dur),
                "p95_us": sorted(dur)[int(len(dur) * 0.95)],
                "samples": len(dur),
            }
    report["planner"][variant + "_kernel"] = profiles
(r / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
for k, v in report["serving"].items():
    print(
        k,
        round(v["before"]["short_median_tok_s"], 3),
        "->",
        round(v["after"]["short_median_tok_s"], 3),
        round(v["short_change_pct"], 2),
        "%",
        v["response_text_equal"],
    )
print("planner states:", report["planner"].get("all_states_equal"))
print(
    "kernels",
    report["planner"].get("before_kernel"),
    report["planner"].get("after_kernel"),
)
