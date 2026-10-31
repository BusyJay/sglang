# SPDX-License-Identifier: Apache-2.0
"""Sweep fixed/adaptive MTP budgets through real streaming HTTP generation.

Starts one local server at a time. Saves timing, token/logprob hashes and
contract checks, without generated text or local checkpoint paths in reports.
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, route, payload=None, *, stream=False):
    req = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        if not stream:
            data = response.read()
            if route in ("/health", "/flush_cache"):
                return data.decode()
            return json.loads(data)
        first = last = None
        first_count = 0
        gaps = []
        event = None
        previous_count = 0
        for line in response:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            count = event["meta_info"]["completion_tokens"]
            if count > previous_count:
                now = time.perf_counter()
                if first is None:
                    first, first_count = now, count
                else:
                    gaps.append(now - last)
                last, previous_count = now, count
        if event is None or first is None:
            raise RuntimeError("Streaming request returned no tokens")
        metrics = dict(
            elapsed_s=time.perf_counter() - started,
            ttft_s=first - started,
            decode_tokens_per_s=(previous_count - first_count) / (last - first)
            if last > first
            else 0,
            max_token_gap_s=max(gaps, default=0),
            completion_tokens=previous_count,
            cached_tokens=event["meta_info"].get("cached_tokens", 0),
        )
        return event, metrics


def panel(event):
    meta = event["meta_info"]
    values = dict(
        output_ids=event["output_ids"],
        logprobs=meta.get("output_token_logprobs"),
        top_logprobs=meta.get("output_top_logprobs"),
        finish_reason=meta.get("finish_reason"),
    )
    return dict(
        sha256=hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
        tokens_sha256=hashlib.sha256(
            json.dumps(values["output_ids"]).encode()
        ).hexdigest(),
        completion_tokens=meta["completion_tokens"],
        cached_tokens=meta.get("cached_tokens", 0),
    )


def contracts(base, prompts):
    results = {}

    def run(name, inputs, sampling=None, **extras):
        body = dict(
            input_ids=inputs,
            sampling_params=sampling
            or dict(temperature=0, max_new_tokens=32, ignore_eos=True),
            return_logprob=True,
            top_logprobs_num=3,
            **extras,
        )
        event = request(base, "/generate", body)
        results[name] = panel(event)
        return event

    request(base, "/flush_cache")
    prefix = [10, 11, 12, 13] * 80
    run("chunked_cold", prefix)
    run("prefix_exact", prefix)
    run("prefix_suffix", prefix + [19, 23, 29, 31])
    request(base, "/flush_cache")
    run(
        "seeded_sample",
        prompts["english"],
        dict(
            temperature=0.7,
            top_p=0.9,
            top_k=32,
            min_p=0.05,
            sampling_seed=37,
            max_new_tokens=32,
            ignore_eos=True,
        ),
    )
    request(base, "/flush_cache")
    run(
        "logit_bias",
        prompts["english"],
        dict(
            temperature=0,
            max_new_tokens=12,
            ignore_eos=True,
            logit_bias={"42": 100.0, "43": 95.0},
        ),
    )
    request(base, "/flush_cache")
    run(
        "regex_grammar",
        prompts["english"],
        dict(
            temperature=0,
            max_new_tokens=8,
            regex="[0-9]{3}",
        ),
    )
    request(base, "/flush_cache")
    control = run("short_control", prompts["english"])
    stop = control["output_ids"][5]
    request(base, "/flush_cache")
    run(
        "stop_token",
        prompts["english"],
        dict(
            temperature=0,
            max_new_tokens=32,
            stop_token_ids=[stop],
        ),
    )
    for limit in (1, 2, 3):
        request(base, "/flush_cache")
        run(
            f"budget_{limit}",
            prompts["chinese"],
            dict(
                temperature=0,
                max_new_tokens=limit,
                ignore_eos=True,
            ),
        )
    # Independent concurrent requests exercise scheduler composition changes.
    request(base, "/flush_cache")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = {
            name: pool.submit(run, "parallel_" + name, prompts[name])
            for name in ("english", "code")
        }
        for future in futures.values():
            future.result()
    # Client disconnect while a private draft window is live; subsequent
    # requests must still reuse only committed prefix state.
    request(base, "/flush_cache")
    body = dict(
        input_ids=prompts["english"],
        stream=True,
        sampling_params=dict(temperature=0, max_new_tokens=256, ignore_eos=True),
    )
    req = urllib.request.Request(
        base + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                if json.loads(line[5:])["meta_info"]["completion_tokens"] >= 3:
                    break
    time.sleep(0.2)
    run("after_disconnect", prompts["code"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 6, 8, -1]
    )
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--check-contracts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.output_length < 2
        or args.repeats < 1
        or any(not -1 <= s <= 16 for s in args.steps)
    ):
        parser.error("Require output-length >= 2, repeats >= 1 and steps in -1..16")
    from transformers import AutoTokenizer

    from sglang.test.test_utils import terminate_and_kill_process_tree

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompts = {
        name: tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        for name, text in (
            (
                "english",
                "Explain how a hash table resolves collisions, with a small example.",
            ),
            ("chinese", "请解释为什么推理时减少权重读取可以提升速度，并举一个例子。"),
            (
                "code",
                "Write a Python function that merges overlapping intervals. "
                "Explain its complexity.",
            ),
        )
    }
    base = f"http://127.0.0.1:{args.port}"
    report = dict(
        scope=(
            "serial streaming HTTP; fixed-length greedy; "
            "cache flushed per timing request"
        ),
        output_length=args.output_length,
        repeats=args.repeats,
        configurations=[],
    )
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous["output_length"], previous["repeats"]) != (
            args.output_length,
            args.repeats,
        ):
            parser.error("Resume settings must match the saved report")
        report["configurations"] = [
            c for c in previous["configurations"] if c.get("complete")
        ]
    reference = {}
    if report["configurations"]:
        for row in report["configurations"][0]["timings"]:
            reference[("timing", row["prompt"], row["repeat"])] = row["sha256"]
        for name, row in report["configurations"][0]["contracts"].items():
            reference[("contract", name)] = row["sha256"]
    completed = {c["steps"] for c in report["configurations"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for steps in args.steps:
        if steps in completed:
            continue
        log = args.output.with_name(args.output.stem + f"_step{steps}.log")
        command = [
            sys.executable,
            "-u",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(args.model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--context-length",
            "4096",
            "--max-total-tokens",
            "4352",
            "--chunked-prefill-size",
            "256",
            "--max-running-requests",
            "2",
            "--mlx-enable-sampling",
            "--mlx-mtp-num-steps",
            str(steps),
            "--random-seed",
            "1234",
            "--enable-deterministic-inference",
            "--disable-cuda-graph",
        ]
        env = dict(os.environ, SGLANG_USE_MLX="1")
        with log.open("w") as handle:
            proc = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env
            )
            try:
                started = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited: {log}")
                    try:
                        request(base, "/health")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() - started > 300:
                            raise RuntimeError(f"Server startup timeout: {log}")
                        time.sleep(0.5)
                print(
                    f"Ready step={steps} after {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                config = dict(steps=steps, timings=[], contracts={}, complete=False)
                report["configurations"].append(config)
                for repeat in range(args.repeats):
                    order = (
                        list(prompts) if repeat % 2 == 0 else list(reversed(prompts))
                    )
                    for name in order:
                        request(base, "/flush_cache")
                        body = dict(
                            input_ids=prompts[name],
                            stream=True,
                            sampling_params=dict(
                                temperature=0,
                                max_new_tokens=args.output_length,
                                ignore_eos=True,
                            ),
                        )
                        event, metrics = request(base, "/generate", body, stream=True)
                        result = panel(event)
                        key = ("timing", name, repeat)
                        reference.setdefault(key, result["sha256"])
                        if reference[key] != result["sha256"]:
                            raise RuntimeError(f"AR/MTP mismatch step={steps}, {key}")
                        if (
                            metrics["completion_tokens"] != args.output_length
                            or metrics["cached_tokens"]
                        ):
                            raise RuntimeError(
                                "Timing request did not use the expected "
                                "cold fixed-length contract"
                            )
                        row = dict(prompt=name, repeat=repeat, **metrics)
                        row.update(result)
                        config["timings"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(dict(steps=steps, **row)), flush=True)
                if args.check_contracts:
                    config["contracts"] = contracts(base, prompts)
                    for name, result in config["contracts"].items():
                        key = ("contract", name)
                        reference.setdefault(key, result["sha256"])
                        if reference[key] != result["sha256"]:
                            raise RuntimeError(
                                f"Contract mismatch step={steps}, {name}"
                            )
                    print(
                        f"Contracts step={steps}: {len(config['contracts'])} passed",
                        flush=True,
                    )
                text = log.read_text()
                stats = [
                    line.split("MLX MTP stats: ", 1)[1]
                    for line in text.splitlines()
                    if "MLX MTP stats: " in line
                ]
                config["mtp_stats"] = ast.literal_eval(stats[-1]) if stats else None
                config["complete"] = True
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            finally:
                terminate_and_kill_process_tree(proc)
                time.sleep(1)


if __name__ == "__main__":
    main()
