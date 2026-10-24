"""Run one checked Docker case with a bounded GPU memory sampler."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
reference, name = sys.argv[1:]
command = ["docker", "exec", "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "-e", "PYTHONPATH=/workspace/sglang/python", "qwen38-csf-defaults-20261023", "/opt/sglang/bin/python3", "/results/capacity_checked.py", "--reference", "/results/" + reference, "--output", "/results/" + name, "--modes", "default"]
with (root / (name + "-console.log")).open("w") as log, (root / (name + "-memory.csv")).open("w") as memory:
    worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    sampler = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=timestamp,memory.used,memory.free", "--format=csv,noheader,nounits", "-lms", "100"], stdout=memory, stderr=subprocess.DEVNULL)
    try:
        code = worker.wait()
    finally:
        sampler.terminate()
        sampler.wait()
raise SystemExit(code)
