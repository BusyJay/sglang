"""Sequential GPU experiments in independent processes, ABBA order."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

out = Path(__file__).resolve().parent
for dtype in ("fp32", "bf16"):
    counts = {False: 0, True: 0}
    for warm in (False, True, True, False):
        cmd = [sys.executable, str(out / "chain_probe.py"), "--dtype", dtype,
               "--sample", str(counts[warm])]
        counts[warm] += 1
        if warm:
            cmd.append("--warmup")
        subprocess.run(cmd, check=True)
summary = {}
for dtype in ("fp32", "bf16"):
    summary[dtype] = {}
    for mode in ("lazy", "warm"):
        rows = [json.loads(p.read_text()) for p in sorted(out.glob(f"chain_{dtype}_{mode}_*.json"))]
        summary[dtype][mode] = {
            key: [row[key] for row in rows]
            for key in ("startup_warmup_ms", "first_request_step_ms", "steady_mean_ms")
        }
(out / "chain_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
