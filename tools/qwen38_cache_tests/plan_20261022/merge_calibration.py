"""Merge disjoint token-shape tables, preserving paired measurement records."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prefix", default="gate-final")
parser.add_argument("--output", default="calibration.json")
args = parser.parse_args()
tables = [
    json.loads((HERE / f"{args.prefix}-{shape}.calibration.json").read_text())
    for shape in ("m3", "m12")
]
metadata = [{k: v for k, v in table.items() if k != "samples"} for table in tables]
if metadata[0] != metadata[1]:
    raise ValueError("calibration metadata differs")
records = {}
for table in tables:
    for row in table["samples"]:
        key = tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in row.items()
            if k not in ("cpu_us", "gpu_us")
        )
        if (
            key not in records
            or row["cpu_us"] / row["gpu_us"]
            > records[key]["cpu_us"] / records[key]["gpu_us"]
        ):
            records[key] = row
merged = dict(metadata[0], samples=list(records.values()))
(HERE / args.output).write_text(json.dumps(merged, indent=2) + "\n")
print("accepted measured keys", len(records), "runtime", merged["runtime_version"])
