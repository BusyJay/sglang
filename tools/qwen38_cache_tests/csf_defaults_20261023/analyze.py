"""Condense complete runs, retaining MTP acceptance and output identities."""

import hashlib
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
report = {"runs": {}, "groups": {}}
for path in sorted(root.glob("*/performance.json")):
    data = json.loads(path.read_text())
    if data["status"] != "passed":
        continue
    name = path.parent.name
    cases = {}
    for case in data["modes"]["default"]["cases"]:
        meta = case["meta_info"]
        events = case["stream_events"]
        intervals = [
            (b[0] - a[0]) * 1000 / (b[1] - a[1]) for a, b in zip(events, events[1:])
        ]
        verify = meta.get("spec_verify_ct", 0)
        cases[case["name"]] = {
            k: case[k]
            for k in (
                "prompt_tokens",
                "output_tokens",
                "decode_tokens_per_second",
                "after_64_tokens_per_second",
                "ttft_seconds",
                "e2e_seconds",
            )
        }
        cases[case["name"]].update(
            output_sanity_passed=not bool(
                re.search(r"([^\s])\1{63,}", case["response_text"] or "")
            ),
            output_excerpt=case["response_text"][-300:],
            output_sha256=hashlib.sha256(case["response_text"].encode()).hexdigest(),
            stream_interval_per_token_p95_ms=sorted(intervals)[
                int(0.95 * len(intervals))
            ],
            cached_tokens=meta.get("cached_tokens"),
            spec_accept_length=meta.get("spec_accept_length"),
            spec_accept_rate=meta.get("spec_accept_rate"),
            spec_verify_ct=verify,
            ms_per_verify_round=(events[-1][0] - events[0][0]) * 1000 / (verify - 1)
            if verify > 1
            else None,
        )
    cfg = json.loads(
        data["command"][data["command"].index("--model-loader-extra-config") + 1]
    )
    logs = (path.parent / "default.log").read_text(errors="replace").splitlines()
    stats = [
        json.loads(line[line.index("{") :])
        for line in logs
        if "Qwen38ExpertCacheStats" in line
    ]
    run = dict(
        output_sanity_passed=all(c["output_sanity_passed"] for c in cases.values()),
        cases=cases,
        config=cfg,
        startup_seconds=data["modes"]["default"]["startup_seconds"],
    )
    if stats:
        last = stats[-1]
        run["last_stats"] = {
            k: v
            for k, v in last.items()
            if not k.startswith("layer_")
            and k not in ("layers", "layer_stats", "prefill_copy_stats")
        }
    run["capacity_messages"] = [
        line
        for line in logs
        if "adaptive KV stages=" in line
        or "expert source ready" in line
        or "startup cache plan:" in line
    ]
    report["runs"][name] = run
    if run["output_sanity_passed"] and name.startswith(("r0-", "r1-", "fixed-")):
        group = name[6:] if name.startswith("fixed-") else name[3:]
        report["groups"].setdefault(group, []).append(name)
for group, names in list(report["groups"].items()):
    runs = [report["runs"][name] for name in names]
    short = [
        c["decode_tokens_per_second"]
        for r in runs
        for n, c in r["cases"].items()
        if n.startswith("short_")
    ]
    natural = [r["cases"]["natural_chat"]["decode_tokens_per_second"] for r in runs]
    report["groups"][group] = dict(
        runs=names,
        short_median_tok_s=statistics.median(short),
        short_min_tok_s=min(short),
        short_max_tok_s=max(short),
        natural_median_tok_s=statistics.median(natural),
        natural_rounds_tok_s=natural,
    )
(root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
for name, values in report["groups"].items():
    print(
        name,
        "short",
        round(values["short_median_tok_s"], 3),
        "natural",
        round(values["natural_median_tok_s"], 3),
        "rounds",
        len(values["runs"]),
    )
