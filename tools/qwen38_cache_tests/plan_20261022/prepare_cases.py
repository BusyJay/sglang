"""Derive paired 1/2/3-expert measurements from a fresh route replay.

All prefixes share the same routes and admitted/rejected experts. One/two-token
cases truncate those recorded routes and retain their initial residency mix;
these are branch fixtures, not claims about a separate cache trajectory.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="route-cases.json")
parser.add_argument("--prefix", default="")
args = parser.parse_args()
source = HERE / args.source
replay = json.loads(source.read_text())
variants = {}
for frame, original in enumerate(replay["cases"]):
    for tokens in (1, 2, 3):
        rows = original["rows"][:tokens]
        counts = Counter(e for row in rows for e in row if e >= 0)
        rejected = sorted(
            set(original["rejected_experts"]) & counts.keys(),
            key=lambda e: (counts[e], e),
        )
        admitted = sorted(set(original["admitted_experts"]) & counts.keys())
        for k in range(1, min(3, len(rejected)) + 1):
            cpu_profile = tuple(counts[e] for e in rejected[:k])
            rejected_profile = tuple(counts[e] for e in rejected)
            key = (
                tokens,
                sum(counts.values()) - sum(cpu_profile),
                sum(cpu_profile),
                k,
                cpu_profile,
                rejected_profile,
            )
            variant = (key, len(admitted), len(rejected) - k)
            record = copy.deepcopy(original)
            record.update(
                key=list(key),
                rows=rows,
                admitted_experts=admitted,
                rejected_experts=rejected,
                cpu_experts=rejected[:k],
                paired_frame=[frame, tokens],
            )
            if variant not in variants or record["count"] > variants[variant]["count"]:
                variants[variant] = record

for label, token_set, budget in (("m3", {3}, 180), ("m12", {1, 2}, 72)):
    candidates = {v: r for v, r in variants.items() if v[0][0] in token_set}
    sizes = Counter(v[0] for v in candidates)
    popularity = Counter()
    for variant, record in candidates.items():
        popularity[variant[0]] += record["count"]
    chosen = set()
    for key, _ in popularity.most_common():
        if sizes[key] <= budget:
            chosen.add(key)
            budget -= sizes[key]
    result = {k: replay[k] for k in ("geometry", "budget", "provenance")}
    result["provenance"] = dict(result["provenance"])
    for p in (source, Path(__file__).resolve()):
        result["provenance"][str(p.relative_to(ROOT))] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    result["cases"] = [r for v, r in candidates.items() if v[0] in chosen]
    result["complete_keys"] = [list(k) for k in sorted(chosen)]
    result["note"] = __doc__
    (HERE / f"gate-cases-{args.prefix}{label}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(label, len(result["cases"]), len(chosen))
