"""Run the affected cache, compression, loader and lifecycle tests serially."""

import glob
import json
import subprocess
from pathlib import Path

files = []
for pattern in (
    "test/registered/kernels/ops/moe/test_qwen38_*.py",
    "test/registered/unit/layers/moe/test_qwen38_*.py",
    "test/registered/unit/layers/quantization/test_qwen38_*.py",
    "test/registered/unit/model_loader/test_qwen38_*.py",
    "test/registered/unit/mem_cache/test_qwen38_*.py",
    "test/registered/unit/kernels/test_cpu_nvfp4_build.py",
):
    files.extend(glob.glob("/workspace/sglang/" + pattern))
files = sorted(set(files))
Path("/results/test-files.json").write_text(json.dumps(files, indent=2) + "\n")
subprocess.run(
    ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files], check=True
)
