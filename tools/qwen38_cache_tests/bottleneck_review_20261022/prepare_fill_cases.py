# SPDX-License-Identifier: Apache-2.0
"""Prepare fill-aware branch fixtures without dropping layer/expert representatives.

The source is an earlier replay, not a replay of the modified serving cache.
Outputs retain its provenance separately and fingerprint the current adapters;
all branch timings must be measured again on the target GPU/CPU.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def prepare(replay):
    variants = {}
    for frame, original in enumerate(replay["cases"]):
        for tokens in (1, 2, 3):
            rows = original["rows"][:tokens]
            if len(rows) != tokens:
                continue
            counts = Counter(e for row in rows for e in row if e >= 0)
            rejected = sorted(
                set(original["rejected_experts"]) & counts.keys(),
                key=lambda e: (counts[e], e),
            )
            admitted = sorted(set(original["admitted_experts"]) & counts.keys())
            for k in range(1, min(3, len(rejected)) + 1):
                cpu = tuple(counts[e] for e in rejected[:k])
                key = (
                    tokens,
                    sum(counts.values()) - sum(cpu),
                    sum(cpu),
                    k,
                    cpu,
                    tuple(counts[e] for e in rejected),
                    len(admitted) + len(rejected) - k,
                )
                variant = (
                    key,
                    original["layer"],
                    tuple(map(tuple, rows)),
                    tuple(admitted),
                    tuple(rejected),
                )
                record = copy.deepcopy(original)
                record.update(
                    key=list(key),
                    rows=rows,
                    admitted_experts=admitted,
                    rejected_experts=rejected,
                    cpu_experts=rejected[:k],
                    paired_frame=[frame, tokens],
                )
                if (
                    variant not in variants
                    or record["count"] > variants[variant]["count"]
                ):
                    variants[variant] = record
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=HERE.parent / "plan_20261022/route-cases.json"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "fill_cases")
    parser.add_argument("--m3-budget", type=int, default=180)
    parser.add_argument("--m12-budget", type=int, default=72)
    args = parser.parse_args()
    replay = json.loads(args.source.read_text())
    variants = prepare(replay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, tokens, budget in (
        ("m3", {3}, args.m3_budget),
        ("m12", {1, 2}, args.m12_budget),
    ):
        candidates = {v: r for v, r in variants.items() if v[0][0] in tokens}
        sizes = Counter(v[0] for v in candidates)
        popularity = Counter()
        for v, r in candidates.items():
            popularity[v[0]] += r["count"]
        chosen = set()
        for key, _ in popularity.most_common():
            if sizes[key] <= budget:
                chosen.add(key)
                budget -= sizes[key]
        if not chosen:
            raise ValueError(f"{label}: budget cannot cover a complete fill bucket")
        provenance = {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in replay["provenance"]
        }
        for path in (
            args.source.resolve(),
            Path(__file__).resolve(),
            ROOT / "python/sglang/srt/layers/moe/qwen38_hybrid_policy.py",
            ROOT / "python/sglang/srt/layers/moe/qwen38_hybrid.py",
            ROOT / "test/manual/kernels/moe/bench_qwen38_hybrid_gate.py",
        ):
            provenance[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        result = {k: replay[k] for k in ("geometry", "budget")}
        result.update(
            provenance=provenance,
            source_replay_provenance=replay["provenance"],
            cases=[r for v, r in candidates.items() if v[0] in chosen],
            complete_keys=[list(k) for k in sorted(chosen)],
            note=__doc__,
        )
        output = args.output_dir / f"gate-cases-{label}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            label,
            len(result["cases"]),
            "representatives",
            len(chosen),
            "complete buckets",
        )


if __name__ == "__main__":
    main()
