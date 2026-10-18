"""Summarize completed service experiments; retain paired and per-round results."""

import hashlib
import gzip
import json
from pathlib import Path
import statistics


def read_json(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def metrics(rows):
    return {
        "samples": len(rows),
        "median_decode_tokens_per_second": statistics.median(
            v["decode_tokens_per_second"] for v in rows
        ),
        "median_e2e_tokens_per_second": statistics.median(
            v["e2e_tokens_per_second"] for v in rows
        ),
        "median_ttft_s": statistics.median(v["ttft_s"] for v in rows),
        "median_accept_length": statistics.median(
            v["meta_info"]["spec_accept_length"] for v in rows
        ),
        "median_approx_verify_ms": statistics.median(
            1000 * v["meta_info"]["spec_accept_length"] / v["decode_tokens_per_second"]
            for v in rows
        ),
    }


def block(root, expected_runs):
    manifest = read_json(root / "manifest.json")
    assert len(manifest["runs"]) == expected_runs
    variants = {}
    canaries = []
    for run in manifest["runs"]:
        folder = root / run["folder"]
        record = variants.setdefault(
            run["name"], {"short": [], "long": [], "rounds": []}
        )
        per_round = {"repeat": run["repeat"]}
        for stage, tokens in [("short", 1024), ("long", 512)]:
            data = read_json(folder / f"{stage}.json")
            rows = data["cases"]
            assert len(rows) == (3 if stage == "short" else 1)
            assert all(v["meta_info"]["completion_tokens"] == tokens for v in rows)
            if stage == "long":
                assert all(v["meta_info"]["cached_tokens"] == 0 for v in rows)
                assert all(v["meta_info"]["prompt_tokens"] == 40017 for v in rows)
            per_round[stage] = metrics(rows)
            record[stage].extend(rows)
        record["rounds"].append(per_round)
        canary = read_json(folder / "canary.json")
        canaries.append(
            {
                "run": run["folder"],
                "text_sha256": hashlib.sha256(canary["text"].encode()).hexdigest(),
            }
        )
    for record in variants.values():
        for stage in ["short", "long"]:
            record[stage] = metrics(record[stage])
    baseline = variants["baseline"]
    for name, record in variants.items():
        for stage in ["short", "long"]:
            m = "median_decode_tokens_per_second"
            record[stage]["delta_vs_baseline_pct"] = 100 * (
                record[stage][m] / baseline[stage][m] - 1
            )
        for v in record["rounds"]:
            b = next(x for x in baseline["rounds"] if x["repeat"] == v["repeat"])
            for stage in ["short", "long"]:
                v[stage]["delta_vs_same_round_baseline_pct"] = 100 * (
                    v[stage][m] / b[stage][m] - 1
                )
    return {"variants": variants, "canaries": canaries, "runs": manifest["runs"]}


def main():
    root = Path(__file__).resolve().parent / "results"
    result = {
        "launch_and_fill": block(root / "service", 9),
        "cta_only": block(root / "service-cta-only", 6),
        "limits": [
            "Three fresh workers per candidate; short has three prompts per worker, long one.",
            "Approx verify ms = accept length / sustained tokens per second; streaming boundary effects remain.",
            "MTP output and routing vary even for baseline repeats; no bit-identical full-model claim.",
            "Single request, synthetic repeated-text long context, not every production workload.",
        ],
    }
    (root / "service-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, group in result.items():
        if isinstance(group, dict):
            for variant, v in group["variants"].items():
                print(
                    name,
                    variant,
                    {
                        s: round(v[s]["median_decode_tokens_per_second"], 2)
                        for s in ["short", "long"]
                    },
                )


if __name__ == "__main__":
    main()
