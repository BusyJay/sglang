"""Serial GPU validation. Start after before-n1-full has completed successfully."""

import json, subprocess, time
from pathlib import Path

r = Path("/results")
while True:
    p = r / "before-n1-full/performance.json"
    if p.exists() and json.loads(p.read_text()).get("status") == "passed":
        break
    time.sleep(5)
for name, args in [
    ("after-n1-full", []),
    ("before-n0-short", ["--before", "--n", "0", "--short"]),
    ("after-n0-short", ["--n", "0", "--short"]),
    ("before-gpu-short", ["--before", "--gpu", "--short"]),
    ("after-gpu-short", ["--gpu", "--short"]),
    ("before-n1-repeat", ["--before", "--short"]),
    ("after-n1-repeat", ["--short"]),
]:
    with (r / f"{name}-run.log").open("w") as log:
        subprocess.run(
            ["python", "/results/run_serving.py", name, *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(name, "passed", flush=True)
# Microbenchmarks and tests run after serving, avoiding CPU/GPU contention.
for name, repo in [("before", "/before"), ("after", "/workspace/sglang")]:
    with (r / f"{name}-planner-run.log").open("w") as log:
        subprocess.run(
            [
                "env",
                f"PYTHONPATH={repo}/python",
                "python",
                "/results/bench_planner.py",
                name,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(name, "planner passed", flush=True)
