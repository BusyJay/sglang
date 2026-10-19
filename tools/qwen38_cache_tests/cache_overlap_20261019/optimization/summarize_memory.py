"""Read only trusted local allocator snapshots from this experiment."""

import collections
import json
import pickle
from pathlib import Path

root = Path(__file__).resolve().parent
reports = []
for path in sorted(root.glob("memory-*/*.pickle")):
    data = pickle.load(path.open("rb"))
    workspaces = collections.Counter()
    for event in data["device_traces"][0]:
        if event["action"] == "alloc" and any(
            f.get("filename", "").endswith("core.py")
            and f.get("name") == "cutlass_fused_moe"
            for f in event.get("frames", [])
        ):
            workspaces[event["size"]] += 1
    reports.append(
        dict(
            case=path.parent.name,
            allocator=data.get("allocator_settings"),
            oom_events=sum(e["action"] == "oom" for e in data["device_traces"][0]),
            max_cutlass_allocation_bytes=max(workspaces, default=0),
            allocations=[
                dict(bytes=size, count=count)
                for size, count in workspaces.most_common()
            ],
        )
    )
(root / "memory-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
for r in reports:
    print(r["case"], r["oom_events"], r["max_cutlass_allocation_bytes"] / 2**20)
