"""Capture only warmed serving work in a disposable full-model server."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "global_cpu_resize_v2_20261020")
)
from benchmark_performance import generate
from transformers import AutoTokenizer


def request(endpoint, data=None):
    req = urllib.request.Request(
        "http://127.0.0.1:30938/" + endpoint,
        None if data is None else json.dumps(data).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["default", "legacy_gpu"])
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="NVTX CPU native, input and result waits",
    )
    parser.add_argument("--cpu-sampling", action="store_true")
    parser.add_argument(
        "--nsys",
        default=shutil.which("nsys") or "/usr/lib/nsight-systems/target-linux-x64/nsys",
        help="Nsight CLI executable inside the capture environment",
    )
    parser.add_argument(
        "--functions-trace",
        type=Path,
        default=Path(
            "tools/qwen38_cache_tests/nsight_optimization_20261021/functions.json"
        ),
        help="Function annotations used with --annotations",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = Path("tools/qwen38_cache_tests/global_cpu_resize_v2_20261020")
    command = json.loads((base / "performance-command.json").read_text())["command"] + [
        "--random-seed",
        "42",
    ]
    if args.detailed:
        command = [
            command[0],
            "tools/qwen38_cache_tests/async_admission_20261021/launch_profiled.py",
            *command[3:],
        ]
    tok = AutoTokenizer.from_pretrained(
        command[command.index("--model-path") + 1], local_files_only=True
    )
    phrase = tok.encode(
        "Explain how an expert cache handles a miss. ", add_special_tokens=False
    )

    def ids(length, index):
        return [
            1000 + ((index + 1) * 2654435761 + s * 40503) % 100000 for s in range(64)
        ] + (phrase * ((length + len(phrase) - 1) // len(phrase)))[:length]

    results = {}
    for mode in args.modes:
        env = dict(os.environ)
        if mode == "legacy_gpu":
            env["SGLANG_ENABLE_QWEN38_CPU_HYBRID"] = "0"
        elif mode != "default":
            raise ValueError(mode)
        profiled = [
            args.nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--python-sampling=true",
            "--python-sampling-frequency=100",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--wait=all",
            "--force-overwrite=true",
            "--output",
            str(args.output / mode),
        ] + command
        if args.annotations:
            profiled[2:2] = [f"--python-functions-trace={args.functions_trace}"]
        if args.cpu_sampling:
            profiled[profiled.index("--sample=none")] = "--sample=process-tree"
            profiled[profiled.index("--cpuctxsw=none")] = "--cpuctxsw=process-tree"
        process = None
        try:
            with (args.output / f"{mode}.log").open("w") as log:
                process = subprocess.Popen(
                    profiled,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                started = time.monotonic()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{mode} exited {process.returncode}")
                    if time.monotonic() - started > 600:
                        raise TimeoutError("startup")
                    try:
                        with urllib.request.urlopen(
                            "http://127.0.0.1:30938/ready", timeout=3
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(2)
                print(mode, "warmup", flush=True)
                generate(ids(1024, 0), 128)
                # Sanity probe before arming CUDA tracing.
                answer = request(
                    "generate",
                    {
                        "text": "What is 17 plus 25? Reply with just the number.",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 128},
                    },
                )
                if "42" not in answer.get("text", ""):
                    raise AssertionError(answer)
                print(mode, "capture", flush=True)
                request("start_profile", {"activities": ["CUDA_PROFILER"]})
                cases = []
                for name, length, tokens, index in [
                    ("prefill", 16384, 8, 1),
                    ("decode", 1024, 256, 2),
                ]:
                    case = generate(ids(length, index), tokens)
                    case["name"] = name
                    cases.append(case)
                    print(mode, name, case["e2e_seconds"], flush=True)
                request("stop_profile", {})
                results[mode] = {"cases": cases, "command": profiled, "sanity": answer}
                (args.output / "requests.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=50)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        time.sleep(3)


if __name__ == "__main__":
    main()
