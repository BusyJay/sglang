#!/usr/bin/env python3
"""Summarize trusted startup diagnostic JSON, including unreleasable split segments.

Never loads pickle: retain snapshots for optional PyTorch memory_viz inspection.
"""

import argparse
import json
from pathlib import Path


def summarize(directory):
    path = Path(directory)
    events = [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]
    result = []
    for event in events:
        values = event["values"]
        segment_path = path / (Path(event["snapshot"]).stem + ".segments.json")
        segments = json.loads(segment_path.read_text())
        splits = []
        for segment in segments:
            device = segment.get("device")
            if (
                event.get("device_index") is not None
                and device != event["device_index"]
            ):
                continue
            # A segment containing active allocations cannot be released whole.
            blocks = segment.get("blocks", [])
            if any(b.get("state", "").startswith("active") for b in blocks):
                inactive = sum(
                    b["size"] for b in blocks if b.get("state") == "inactive"
                )
                if inactive:
                    splits.append(
                        {
                            "device": device,
                            "address": segment.get("address"),
                            "total_size": segment.get("total_size"),
                            "inactive_bytes": inactive,
                            "active_blocks": [
                                {
                                    "size": b["size"],
                                    "requested_size": b.get("requested_size"),
                                    "state": b.get("state"),
                                    "frames": b.get("frames", []),
                                }
                                for b in blocks
                                if b.get("state", "").startswith("active")
                            ],
                        }
                    )
        result.append(
            {
                "stage": event["stage"],
                "sequence": event["sequence"],
                "values": values,
                "delta": event["delta"],
                "metadata": event["metadata"],
                "largest_mixed_segments": sorted(
                    splits, key=lambda s: s["inactive_bytes"], reverse=True
                )[:10],
                "inventory": event.get("inventory"),
            }
        )
    return {
        "scope": "startup CUDA diagnosis; not performance evidence",
        "directory": str(path.resolve()),
        "stages": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = summarize(args.directory)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("stage | driver free MiB | allocated MiB | reserved MiB | inactive split MiB")
    for stage in result["stages"]:
        v = stage["values"]
        print(
            stage["stage"]
            + " | "
            + " | ".join(
                f"{v[k]/2**20:.2f}"
                for k in (
                    "driver_free_bytes",
                    "allocated_bytes.all.current",
                    "reserved_bytes.all.current",
                    "inactive_split_bytes.all.current",
                )
            )
        )


if __name__ == "__main__":
    main()
