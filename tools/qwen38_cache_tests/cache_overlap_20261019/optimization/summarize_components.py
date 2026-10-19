import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent
reports = []
for path in sorted((root / "profiles").glob("*.trace.json")):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    calls = sum(
        e.get("name") == "component.layer_call" and e.get("cat") == "user_annotation"
        for e in events
    )
    runtime = collections.Counter(
        e.get("name") for e in events if e.get("cat") == "cuda_runtime"
    )
    copies = collections.Counter(
        e.get("name") for e in events if e.get("cat") == "gpu_memcpy"
    )
    reports.append(
        dict(
            tag=path.name,
            calls=calls,
            kernels_per_call=len(kernels) / calls,
            kernel_us_per_call=sum(e.get("dur", 0) for e in kernels) / calls,
            runtime_per_call={k: v / calls for k, v in runtime.items()},
            copies_per_call={k: v / calls for k, v in copies.items()},
            kernels=[
                dict(
                    name=n,
                    count=sum(e["name"] == n for e in kernels) / calls,
                    us=sum(e.get("dur", 0) for e in kernels if e["name"] == n) / calls,
                )
                for n in dict.fromkeys(e["name"] for e in kernels)
            ],
        )
    )
(root / "component-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
for r in reports:
    print(
        r["tag"],
        r["kernels_per_call"],
        round(r["kernel_us_per_call"], 2),
        r["copies_per_call"],
        r["runtime_per_call"].get("cudaStreamSynchronize", 0),
    )
