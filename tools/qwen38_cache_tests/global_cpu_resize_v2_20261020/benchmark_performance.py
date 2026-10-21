"""Manual, paired full-model streaming benchmark; run via run_performance.sh."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def generate(ids, output_tokens):
    payload = {
        "input_ids": ids,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:30938/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    events = []
    final = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data)
            count = data.get("meta_info", {}).get("completion_tokens", 0)
            if count and (not events or count > events[-1][1]):
                events.append([time.perf_counter() - started, count])
            final = data
    elapsed = time.perf_counter() - started
    if not events or events[-1][1] != output_tokens:
        raise AssertionError((output_tokens, events[-3:], final))
    # MTP returns multiple tokens per event. This rate uses token deltas rather
    # than treating each network event as one token; it excludes the first event.
    first, last = events[0], events[-1]
    rate = (last[1] - first[1]) / (last[0] - first[0])
    steady = next((item for item in events if item[1] >= 64), first)
    steady_rate = (last[1] - steady[1]) / (last[0] - steady[0])
    return {
        "prompt_tokens": len(ids),
        "output_tokens": output_tokens,
        "ttft_seconds": first[0],
        "e2e_seconds": elapsed,
        "decode_tokens_per_second": rate,
        "after_64_tokens_per_second": steady_rate,
        "mean_time_per_output_token_ms": 1000 / rate,
        "stream_events": events,
        "meta_info": final.get("meta_info"),
        "response_text": final.get("text"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["default", "legacy_gpu"],
        default=["default", "legacy_gpu"],
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse fully completed modes"
    )
    parser.add_argument(
        "--short-only",
        action="store_true",
        help="omit long-context calibration workloads",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = json.loads(Path(args.reference).read_text())["command"]
    command += ["--random-seed", "42"]
    tokenizer = AutoTokenizer.from_pretrained(
        command[command.index("--model-path") + 1], local_files_only=True
    )
    phrase = tokenizer.encode(
        "Explain how an expert cache handles a miss. ", add_special_tokens=False
    )

    # Fixed paired inputs, distinct prefixes prevent cross-request radix hits.
    def fixture(length, index):
        salt = [
            1000 + ((index + 1) * 2654435761 + step * 40503) % 100000
            for step in range(64)
        ]
        return salt + (phrase * ((length + len(phrase) - 1) // len(phrase)))[:length]

    chat = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explain in detail how a CPU and GPU can cooperate to run a mixture of experts language model when GPU memory is limited. Discuss cache misses, data movement, latency, and dynamic cache resizing.",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(chat, "get"):
        chat = chat["input_ids"]
    if chat and isinstance(chat[0], list):
        chat = chat[0]
    workloads = [("warmup", fixture(1024, 0), 128)]
    workloads += [(f"short_{i}", fixture(1024, i + 1), 512) for i in range(3)]
    workloads += [
        ("natural_chat", chat, 512),
        ("long_170k", fixture(170000, 5), 8),
        ("post_resize_short", fixture(1024, 6), 512),
    ]
    if args.short_only:
        workloads = workloads[:-2]
    report = {"command": command, "modes": {}, "status": "running"}
    destination = output / "performance.json"
    if args.resume and destination.exists():
        previous = json.loads(destination.read_text())
        if previous["command"] != command:
            raise ValueError("resume command differs")
        report = previous
    report["status"] = "running"
    for mode in args.modes:
        completed = report["modes"].get(mode, {}).get("cases", [])
        if args.resume and [c["name"] for c in completed] == [w[0] for w in workloads]:
            print(f"{mode}: reuse completed mode", flush=True)
            continue
        env = dict(os.environ)
        env.pop("SGLANG_ENABLE_QWEN38_CPU_HYBRID", None)
        env.pop("SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD", None)
        if mode == "legacy_gpu":
            env["SGLANG_ENABLE_QWEN38_CPU_HYBRID"] = "0"
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 30938))
        result = {
            "cases": [],
            "overrides": {
                key: env[key]
                for key in (
                    "SGLANG_ENABLE_QWEN38_CPU_HYBRID",
                    "SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD",
                )
                if key in env
            },
        }
        report["modes"][mode] = result
        process = None
        try:
            with (output / f"{mode}.log").open("w") as log:
                begun = time.monotonic()
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{mode} server exited {process.returncode}")
                    if time.monotonic() - begun > 600:
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
                result["startup_seconds"] = time.monotonic() - begun
                for name, ids, tokens in workloads:
                    print(f"{mode}: {name} starting", flush=True)
                    case = generate(ids, tokens)
                    case["name"] = name
                    result["cases"].append(case)
                    destination.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        f"{mode}: {name}: TTFT={case['ttft_seconds']:.3f}s decode={case['decode_tokens_per_second']:.2f}tok/s e2e={case['e2e_seconds']:.3f}s",
                        flush=True,
                    )
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        time.sleep(3)
    report["status"] = "passed"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
