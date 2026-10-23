"""Run the live parity diagnostic with no benchmark and no service port exposure."""

import json, os, subprocess
from pathlib import Path

root = Path("/results")
env = dict(
    os.environ, **json.loads((root / "fixed-csf-off/environment.json").read_text())
)
env["PYTHONPATH"] = "/results/diagnostic:/workspace/sglang/python"
try:
    subprocess.run(
        [
            "python3",
            "/results/benchmark_checked.py",
            "--reference",
            "/results/fixed-csf-off/command.json",
            "--output",
            f"/results/diag-before-{env.get('QWEN38_DIAG_SYNC', 'prefill')}",
            "--modes",
            "default",
            "--short-only",
            "--warmup-only",
        ],
        env=env,
        timeout=240,
        check=True,
    )
except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
    pass
