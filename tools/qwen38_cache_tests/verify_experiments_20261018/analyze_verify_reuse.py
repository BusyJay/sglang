#!/usr/bin/env python3
"""Measure cross-token reuse from raw per-token verify routes, without torch."""

import argparse
import collections
import gzip
import json
from pathlib import Path


def summarize(records):
    rounds = len(records)
    valid = sum(r["valid_routes"] for r in records)
    token_unique = sum(r["token_unique_routes"] for r in records)
    unique = sum(r["unique_experts"] for r in records)
    work = sum(r["padded_token_rows"] for r in records)
    return dict(
        rounds=rounds,
        valid_routes=valid,
        token_unique_routes=token_unique,
        unique_experts=unique,
        cross_token_reuse_fraction=(token_unique - unique) / token_unique,
        total_route_reuse_fraction=(valid - unique) / valid,
        within_token_duplicates=valid - token_unique,
        no_cross_token_reuse_fraction=sum(
            r["unique_experts"] == r["token_unique_routes"] for r in records
        )
        / rounds,
        padded_arithmetic_factor=work / valid,
        expert_token_multiplicity=dict(
            sorted(
                collections.Counter(
                    m for r in records for m in r["multiplicities"]
                ).items()
            )
        ),
        unique_expert_histogram=dict(
            sorted(collections.Counter(r["unique_experts"] for r in records).items())
        ),
    )


def analyze(path, num_experts=512):
    opener = gzip.open if path.suffix == ".gz" else open
    groups = collections.defaultdict(list)
    case = None
    with opener(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if "event" in row:
                case = row["case"] if row["event"] == "start" else None
                continue
            if case is None or case == "warmup":
                continue
            routes = row["routes"]
            if len(routes) not in (2, 3):
                continue
            filtered = [[e for e in ids if 0 <= e < num_experts] for ids in routes]
            counts = collections.Counter(e for ids in filtered for e in set(ids))
            valid = sum(map(len, filtered))
            if not valid:
                continue
            record = dict(
                valid_routes=valid,
                token_unique_routes=sum(len(set(ids)) for ids in filtered),
                unique_experts=len(counts),
                multiplicities=list(counts.values()),
                padded_token_rows=len(counts) * (1 << (len(routes) - 1).bit_length()),
            )
            groups[case, row["layer"], len(routes)].append(record)
    if not groups:
        raise ValueError("No marked verify samples")
    by_case, by_layer, overall = (
        collections.defaultdict(list),
        collections.defaultdict(list),
        [],
    )
    for (case, layer, tokens), rows in groups.items():
        by_case[case, tokens].extend(rows)
        by_layer[layer, tokens].extend(rows)
        overall.extend(rows)
    return dict(
        scope="Actual target verify routes including proposed positions, not only accepted tokens; no timing from instrumented worker",
        overall=summarize(overall),
        cases=[
            dict(case=c, tokens=t, **summarize(rows))
            for (c, t), rows in sorted(by_case.items())
        ],
        layers=[
            dict(layer=l, tokens=t, **summarize(rows))
            for (l, t), rows in sorted(by_layer.items())
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
    for row in result["cases"]:
        print(
            row["case"],
            row["tokens"],
            row["rounds"],
            f"reuse={row['cross_token_reuse_fraction']:.2%}",
            f"arithmetic={row['padded_arithmetic_factor']:.3f}x",
        )


if __name__ == "__main__":
    main()
