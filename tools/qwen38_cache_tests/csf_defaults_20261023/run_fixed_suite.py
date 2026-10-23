"""One pass after the scatter fix; reuse valid native baselines, no reverse repeat."""

from run_matrix import run
import subprocess

subprocess.run(
    [
        "python3",
        "-m",
        "unittest",
        "test.registered.kernels.ops.moe.test_qwen38_csf_gpu",
        "-v",
    ],
    cwd="/workspace/sglang",
    check=True,
)
for route in ("off", "n0", "n1"):
    run(f"fixed-csf-{route}", "csf", route)
print("single-pass corrected suite passed", flush=True)
