# SPDX-License-Identifier: Apache-2.0
"""Build on one physical core; this command never runs a benchmark."""

import argparse
import os
import subprocess
from pathlib import Path


def check_temperature():
    values = []
    for path in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
        try:
            values.append(int(path.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    if values and max(values) >= 75:
        raise RuntimeError(f"temperature guard: {max(values):.1f} C")
    return max(values, default=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=int, default=0)
    args = parser.parse_args()
    if args.core not in os.sched_getaffinity(0):
        raise ValueError("build core outside affinity")
    print(f"temperature before build: {check_temperature():.1f} C", flush=True)
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/kernels/ops/moe/cpu_nvfp4/kernel.cpp"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"}
    subprocess.run(
        [
            "nice",
            "-n",
            "15",
            "taskset",
            "-c",
            str(args.core),
            "g++",
            "-O3",
            "-std=c++17",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512vl",
            "-mavx512bf16",
            "-fPIC",
            "-shared",
            "-pthread",
            str(source),
            "-o",
            str(args.output.resolve()),
        ],
        env=env,
        check=True,
        timeout=60,
    )
    print(f"temperature after build: {check_temperature():.1f} C", flush=True)


if __name__ == "__main__":
    main()
