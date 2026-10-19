"""Count actual GPU kernels and copy directions, excluding GPU annotations."""

import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent
reports = []
for path in sorted((root / "profiles").glob("*.trace.json")):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    copies = collections.Counter(
        e["name"] for e in events if e.get("cat") == "gpu_memcpy"
    )
    reports.append(
        dict(
            tag=path.name,
            kernels=len(kernels),
            kernel_us=sum(e.get("dur", 0) for e in kernels),
            copies=dict(copies),
            kernel_names=dict(collections.Counter(e["name"] for e in kernels)),
        )
    )
(root / "profile-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
for row in reports:
    print(row["tag"], row["kernels"], round(row["kernel_us"], 1), row["copies"])
