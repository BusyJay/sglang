"""Remove speculative-acceptance confounding, then run cache regressions serially."""

import glob, json, subprocess, time
from pathlib import Path

r = Path("/results")
while True:
    p = r / "after-planner.json"
    if p.exists() and json.loads(p.read_text()).get("status") == "passed":
        break
    time.sleep(5)
for name, args in [("before-n1-no-spec", ["--before"]), ("after-n1-no-spec", [])]:
    with (r / f"{name}-run.log").open("w") as log:
        subprocess.run(
            ["python", "/results/run_serving.py", name, "--short", "--no-spec", *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(name, "passed", flush=True)
files = []
for pattern in [
    "test/registered/kernels/ops/moe/test_qwen38_*.py",
    "test/registered/unit/layers/moe/test_qwen38_hybrid*.py",
    "test/registered/unit/layers/moe/test_qwen38_layered*.py",
    "test/registered/unit/layers/moe/test_qwen38_cache_stats_log.py",
    "test/registered/unit/layers/moe/test_qwen38_cache_overlap_policy.py",
    "test/registered/unit/layers/quantization/test_qwen38_b12x_hybrid.py",
    "test/registered/unit/layers/quantization/test_qwen38_cache_cost_replay.py",
    "test/registered/unit/model_loader/test_qwen38_nvfp4_cache_loader_config.py",
    "test/registered/unit/kernels/test_cpu_nvfp4_build.py",
]:
    files.extend(glob.glob("/workspace/sglang/" + pattern))
(r / "test-files.json").write_text(json.dumps(files, indent=2) + "\n")
with (r / "tests-final.log").open("w") as log:
    subprocess.run(
        ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *sorted(set(files))],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
print("regressions passed", flush=True)
