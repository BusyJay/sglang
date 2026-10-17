#!/usr/bin/env python3
"""Reproduce production geometry in an isolated A16/A4/A4/A16 experiment."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
URL = "http://127.0.0.1:18329"
NAME = "qwen38-mtp-w4a4-ab"
IMAGE = "swallow-sglang:service-flash-next-f712e3118c75"
PROMPTS = [
    "Write a detailed technical explanation of how a CPU cache works, with examples.",
    "用中文详细解释数据库索引的原理以及 B 树和哈希索引的区别。",
    "Implement a Python LRU cache and explain its design step by step.",
]


def api(path, payload=None, timeout=600):
    req = urllib.request.Request(
        URL + path,
        None if payload is None else json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    try:
        return json.loads(body)
    except ValueError:
        return body.decode()


def memory():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.free,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def generate(prompt, tokens):
    api("/flush_cache?timeout=30", {})
    payload = {
        "text": prompt,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": tokens,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        URL + "/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    points = []
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            final = json.loads(line[6:])
            count = final.get("meta_info", {}).get("completion_tokens", 0)
            if count and (not points or count > points[-1][1]):
                points.append([time.monotonic() - start, count])
    elapsed = time.monotonic() - start
    assert final["meta_info"]["completion_tokens"] == tokens, final
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "text": final["text"],
        "meta_info": final["meta_info"],
        "wall_s": elapsed,
        "ttft_s": points[0][0],
        "e2e_tokens_per_second": tokens / elapsed,
        "decode_tokens_per_second": (points[-1][1] - points[0][1])
        / (points[-1][0] - points[0][0]),
        "stream_points": points,
        "gpu_memory": memory(),
    }


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("w4a16", "w4a4"),
        default=["w4a16", "w4a4", "w4a4", "w4a16"],
    )
    args = parser.parse_args()
    ROOT = args.results.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "provenance.json").write_text(
        json.dumps(
            {
                "image": subprocess.check_output(
                    ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
                    text=True,
                ).strip(),
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                "runtime_sha256": hashlib.sha256(
                    (
                        REPO / "python/sglang/srt/layers/quantization/qwen38_fp4_mtp.py"
                    ).read_bytes()
                ).hexdigest(),
                "order": args.modes,
                "note": "All cases flush prefix cache; same persistent source/PLE cache and production launcher. No profiler.",
            },
            indent=2,
        )
        + "\n"
    )
    for index, mode in enumerate(args.modes):
        folder = ROOT / f"round-{index}-{mode}"
        folder.mkdir(exist_ok=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--name",
            NAME,
            "--network",
            "sglang-net",
            "--gpus",
            "device=0",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--cap-add",
            "SYS_PTRACE",
            "--security-opt",
            "seccomp:unconfined",
            "-p",
            "127.0.0.1:18329:8000",
            "-v",
            f"{REPO}:/workspace/sglang:ro",
            "-v",
            "/home/user/workspace/prod:/prod",
            "-v",
            "/home/user/workspace/prod/cache:/root/.cache",
            "-v",
            "/dev/shm/sglang-qwen38-startup:/dev/shm/sglang-qwen38-startup",
            "-v",
            "/tmp/hicache:/hicache",
            "-v",
            f"{ROOT}:/workspace/results",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-e",
            "SGLANG_QWEN38_STARTUP_CACHE=auto",
            "-e",
            "SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup",
            "-e",
            "SGLANG_QWEN38_HOST_TOKEN_EMBEDDING=1",
            "-e",
            "SGLANG_QWEN38_BORROW_PREFILL_ARENA=1",
            "-e",
            "SGLANG_QWEN38_PLE_PACKED_CACHE=1",
            "-e",
            f"SGLANG_QWEN38_MTP_ACTIVATION={mode}",
            "--entrypoint",
            "bash",
            IMAGE,
            "/workspace/results/launch/sglang-fp4-local.sh",
        ]
        (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (folder / "service.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            memory_log = (folder / "gpu-memory.csv").open("w")
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
                    "--format=csv",
                    "-lms",
                    "100",
                ],
                stdout=memory_log,
            )
            try:
                started = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited: {folder}")
                    try:
                        api("/health", timeout=2)
                        break
                    except Exception:
                        if time.monotonic() - started > 600:
                            raise RuntimeError("Server startup timed out")
                        time.sleep(2)
                print(
                    f"READY {index} {mode} {time.monotonic()-started:.1f}s", flush=True
                )
                result = {
                    "mode": mode,
                    "startup_s": time.monotonic() - started,
                    "server_info": api("/get_server_info"),
                    "memory_after_start": memory(),
                    "cases": [],
                }

                def save():
                    (folder / "result.json").write_text(
                        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
                    )

                result["warmup"] = generate(PROMPTS[0], 128)
                for case, prompt in enumerate(PROMPTS):
                    row = generate(prompt, 1024)
                    row.update(kind="short", case=case)
                    result["cases"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "kind",
                                    "case",
                                    "decode_tokens_per_second",
                                    "ttft_s",
                                    "meta_info",
                                )
                            }
                        ),
                        flush=True,
                    )
                for long_case in range(2):
                    row = generate("test context " * 16000 + PROMPTS[long_case], 1024)
                    row.update(kind="long", case=long_case)
                    result["cases"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "kind",
                                    "case",
                                    "decode_tokens_per_second",
                                    "ttft_s",
                                    "meta_info",
                                )
                            }
                        ),
                        flush=True,
                    )
                result["quality"] = []
                for prompt, answer in (
                    ("What is 17 * 23? Respond with only the number.", "391"),
                    (
                        "Reverse the string abcdefg. Respond with only the reversed string.",
                        "gfedcba",
                    ),
                    (
                        "What does sum(i*i for i in range(10)) evaluate to in Python? Respond with only the number.",
                        "285",
                    ),
                    ("Solve 3*x + 7 = 28. Respond with only the value of x.", "7"),
                ):
                    api("/flush_cache?timeout=30", {})
                    response = api(
                        "/v1/chat/completions",
                        {
                            "model": result["server_info"]["served_model_name"],
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": 256,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    content = response["choices"][0]["message"]["content"]
                    result["quality"].append(
                        {
                            "prompt": prompt,
                            "expected": answer,
                            "response": response,
                            "passed": content.strip() == answer,
                        }
                    )
                    save()
                result["server_info_after"] = api("/get_server_info")
                result["memory_after"] = memory()
                save()
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
                memory_log.close()
                subprocess.run(
                    ["docker", "stop", "--time", "20", NAME],
                    check=False,
                    stdout=subprocess.DEVNULL,
                )
                proc.wait(timeout=60)
        print(f"DONE {index} {mode}", flush=True)


if __name__ == "__main__":
    main()
