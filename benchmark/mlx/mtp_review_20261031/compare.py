# SPDX-License-Identifier: Apache-2.0
"""Compare full HTTP outputs/contracts across independent server processes."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference, contracts = {}, {}
    result = {"configurations": [], "timing_comparisons": 0, "contract_comparisons": 0}
    for path in args.reports:
        report = json.loads(path.read_text())
        for config in report["configurations"]:
            if not config["complete"]:
                raise RuntimeError(f"Incomplete panel: {path}, step={config['steps']}")
            for row in config["timings"]:
                key = row["prompt"], row["repeat"]
                reference.setdefault(key, row["sha256"])
                if reference[key] != row["sha256"]:
                    raise RuntimeError(f"Timing output mismatch: {path}, {key}")
                result["timing_comparisons"] += 1
            for name, row in config["contracts"].items():
                contracts.setdefault(name, row["sha256"])
                if contracts[name] != row["sha256"]:
                    raise RuntimeError(f"Contract mismatch: {path}, {name}")
                result["contract_comparisons"] += 1
            result["configurations"].append(
                dict(
                    report=path.name,
                    steps=config["steps"],
                    mean_decode_tokens_per_s=statistics.mean(
                        x["decode_tokens_per_s"] for x in config["timings"]
                    ),
                    by_prompt={
                        prompt: statistics.mean(
                            x["decode_tokens_per_s"]
                            for x in config["timings"]
                            if x["prompt"] == prompt
                        )
                        for prompt in sorted({x["prompt"] for x in config["timings"]})
                    },
                )
            )
    result["all_outputs_equal"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
