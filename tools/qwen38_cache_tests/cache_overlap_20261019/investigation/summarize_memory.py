"""Summarize a trusted, locally captured allocator OOM snapshot."""

import collections
import json
import pickle
from pathlib import Path

root = Path(__file__).resolve().parent
path = sorted((root / "oom-snapshots").glob("*.pickle"))[-1]
snapshot = pickle.load(path.open("rb"))
metadata = json.loads(path.with_suffix(".json").read_text())
normal = [
    s for s in snapshot["segments"] if tuple(s.get("segment_pool_id", [])) == (0, 0)
]
states = collections.Counter()
for segment in normal:
    for block in segment["blocks"]:
        states[block["state"]] += block["size"]
fragment = next(
    s
    for s in normal
    if s["total_size"] == 840 * 2**20
    and any(b["state"] == "inactive" and b["size"] == 680 * 2**20 for b in s["blocks"])
)
transitions = []
workspace_sizes = collections.Counter()
for event in snapshot["device_traces"][0]:
    if event["action"] != "alloc":
        continue
    python = [f for f in event.get("frames", []) if f["filename"].endswith(".py")]
    origin = next(
        (
            f
            for f in python
            if f["filename"].endswith("core.py") and f["name"] == "cutlass_fused_moe"
        ),
        None,
    )
    if origin:
        workspace_sizes[event["size"]] += 1
    if event.get("addr") == fragment["address"]:
        frames = [(Path(f["filename"]).name, f["line"], f["name"]) for f in python[:3]]
        transitions.append(dict(bytes=event["size"], frames=frames))
result = dict(
    snapshot=path.name,
    oom=metadata,
    normal_pool_bytes=dict(states),
    largest_normal_inactive_bytes=max(
        b["size"] for s in normal for b in s["blocks"] if b["state"] == "inactive"
    ),
    workspace_allocations=[
        dict(bytes=size, count=count) for size, count in workspace_sizes.most_common()
    ],
    fragmented_segment=dict(
        bytes=fragment["total_size"],
        blocks=[
            dict(
                bytes=b["size"],
                state=b["state"],
                python_frames=[
                    (Path(f["filename"]).name, f["line"], f["name"])
                    for f in b.get("frames", [])
                    if f["filename"].endswith(".py")
                ][:3],
            )
            for b in fragment["blocks"]
        ],
    ),
    segment_history_first=transitions[:3],
    segment_history_last=transitions[-3:],
)
(root / "memory-root-cause.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in (
                "normal_pool_bytes",
                "largest_normal_inactive_bytes",
                "fragmented_segment",
            )
        },
        indent=2,
    )
)
