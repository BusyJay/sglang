"""CPU-only comparison of independently loaded full-model probe artifacts."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--hash-only",
    action="store_true",
    help="Verify exact recorded arrays without local NPZ files",
)
args = p.parse_args()


def read(path):
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


a = read(args.baseline)
b = read(args.candidate)
assert a["prompt_ids"] == b["prompt_ids"]
records = []
for u, v in zip(a["forced"], b["forced"], strict=True):
    assert (u["prefix"], u["rows"]) == (v["prefix"], v["rows"])
    suffix = f"_p{u['prefix']}_r{u['rows']}.npz"
    row = {k: u[k] for k in ("prefix", "rows")}
    if args.hash_only:
        for key in ("logits", "hidden"):
            if u[key] != v[key]:
                raise ValueError(
                    "Recorded arrays differ; regenerate local NPZ files to measure errors"
                )
            row[key + "_equal"] = True
            row[key + "_max_abs"] = 0.0
        row["top1_equal"], row["kl"] = True, 0.0
        row["comparison"] = "recorded_fp32_sha256"
    else:
        arrays_a = np.load(
            args.baseline.with_name(
                args.baseline.name.removesuffix(".gz").removesuffix(".json") + suffix
            )
        )
        arrays_b = np.load(
            args.candidate.with_name(
                args.candidate.name.removesuffix(".gz").removesuffix(".json") + suffix
            )
        )
        for key in ("logits", "hidden"):
            row[key + "_equal"] = np.array_equal(arrays_a[key], arrays_b[key])
            row[key + "_max_abs"] = float(np.max(np.abs(arrays_a[key] - arrays_b[key])))
        logits_a, logits_b = arrays_a["logits"], arrays_b["logits"]
        row["top1_equal"] = np.array_equal(
            np.argmax(logits_a, -1), np.argmax(logits_b, -1)
        )

        def logsoftmax(x):
            x = x.astype(np.float64) - x.max(-1, keepdims=True)
            return x - np.log(np.exp(x).sum(-1, keepdims=True))

        la, lb = logsoftmax(logits_a), logsoftmax(logits_b)
        row["kl"] = float(np.mean(np.sum(np.exp(la) * (la - lb), axis=-1)))
        row["comparison"] = "local_fp32_arrays"
    row["routes_equal"] = u["routes"] == v["routes"]
    row["cache_equal"] = u["caches"] == v["caches"]
    row["baseline_ms"], row["candidate_ms"] = u["median_ms"], v["median_ms"]
    records.append(row)
generation = []
for u, v in zip(a["generation"], b["generation"], strict=True):
    assert u.get("prefix") == v.get("prefix")
    generation.append(
        dict(
            repeat=u["repeat"],
            prefix=u.get("prefix"),
            tokens_equal=u["tokens"] == v["tokens"],
            acceptance_equal=all(
                u[k] == v[k]
                for k in (
                    "verify_ct",
                    "num_proposed_drafts",
                    "num_reuse_tokens",
                    "eligible_by_depth",
                    "correct_by_depth",
                )
            ),
            baseline_tokens_per_s=u["tokens_per_s"],
            candidate_tokens_per_s=v["tokens_per_s"],
        )
    )
result = dict(
    forced=records,
    generation=generation,
    baseline_peak_bytes=a["peak_bytes"],
    candidate_peak_bytes=b["peak_bytes"],
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
