#!/usr/bin/env python3
"""Matched, warmed MTP throughput measurements using a local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from tools.qwen38_cache_tests.benchmark.core import ROOT, subprocess_environment
from tools.qwen38_cache_tests.benchmark.runner import (
    assert_port_available,
    generate,
    log_snapshots,
    request,
    stop_owned,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def decode_metrics(sample):
    events = [event for event in sample["events"] if event["new_tokens"]]
    if len(events) < 2:
        raise ValueError("need at least two token-bearing stream events")
    first, last = events[0], events[-1]
    count = last["cumulative_tokens"] - first["cumulative_tokens"]
    duration = last["seconds"] - first["seconds"]
    if count <= 0 or duration <= 0:
        raise ValueError("invalid decode timing span")
    meta = sample["events"][-1]["raw"].get("meta_info", {})
    verified = meta.get("spec_verify_ct", 0)
    correct = meta.get("spec_num_correct_drafts", 0)
    proposed = meta.get("spec_num_proposed_drafts", 0)
    return {
        "decode_tokens": count,
        "decode_seconds": duration,
        "decode_tok_s": count / duration,
        "tpot_ms": duration / count * 1000,
        "ttft_s": sample["ttft_s"],
        "e2e_tok_s": sample["output_tokens"] / sample["duration_s"],
        "verify_rounds": verified,
        "correct_drafts": correct,
        "proposed_drafts": proposed,
        "accept_rate": correct / proposed if proposed else None,
        "mean_verify_tokens": 1 + correct / verified if verified else None,
        "correct_drafts_histogram": meta.get("spec_correct_drafts_histogram"),
        "cached_tokens": meta.get("cached_tokens"),
        "meta_info": meta,
    }


def flush(url):
    req = urllib.request.Request(url + "/flush_cache?timeout=30", data=b"{}")
    with urllib.request.urlopen(req, timeout=40) as response:
        if response.status != 200:
            raise RuntimeError("KV cache flush failed")
        response.read()


def inputs(checkpoint, directory, lengths):
    if (directory / "inputs.json").exists():
        return json.loads((directory / "inputs.json").read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    marker = "MTP_THROUGHPUT_PADDING_27af"
    question = (
        "请详细解释 Raft 的选举、日志复制、提交规则、网络分区恢复和成员变更，"
        "给出具体例子，并逐步分析安全性。请持续展开，不要只给摘要。"
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker + question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = rendered.split(marker)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    a, b = encode(head), encode(tail)
    fill = encode(
        "Raft 节点保存任期、投票、日志和提交位置；考虑重启、乱序及网络恢复。\n"
    )
    saved = {}
    for length in lengths:
        remaining = length - len(a) - len(b)
        if remaining < 0:
            raise ValueError("input length cannot fit the chat template")
        ids = a + (fill * ((remaining + len(fill) - 1) // len(fill)))[:remaining] + b
        saved[str(length)] = ids
    write(directory / "inputs.json", saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--slots", type=int, default=5040)
    parser.add_argument("--mem-fraction", type=float, default=0.92)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument("--mamba-slots", type=int, default=10)
    parser.add_argument("--packed", choices=["direct", "packed"], default="packed")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--port", type=int, default=30939)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if max(args.lengths) + args.output_tokens > args.kv_tokens:
        parser.error("input and output must fit the common KV/context capacity")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    frozen = inputs(args.checkpoint, directory, args.lengths)
    loader = {
        "manifest_path": args.manifest,
        "checkpoint_revision": args.revision,
        "runtime_mode": "threaded",
        "cache_slots_total": args.slots,
        "prefill_arena": True,
        "packed_decode": args.packed,
        "packed_prefill": args.packed,
        "async_pack_prefetch": True,
        "preallocate_gdn_proj": True,
        "adaptive_cache": args.adaptive,
        "adaptive_cache_window": 128,
        "adaptive_cache_interval": 0,
        "stats_log_interval": 48,
    }
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.checkpoint,
        "--revision",
        args.revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--load-format",
        "qwen38_nvfp4_cache",
        "--model-loader-extra-config",
        json.dumps(loader),
        "--language-model-only",
        "--ple-offload-embedding",
        "--ple-offload-backend",
        "pinned",
        "--cuda-graph-backend-prefill=disabled",
        "--cuda-graph-backend-decode=breakable",
        "--cuda-graph-bs-decode",
        "1",
        "--cuda-graph-max-bs-decode",
        "1",
        "--disable-shared-experts-fusion",
        "--moe-runner-backend",
        "flashinfer_cutlass",
        "--max-running-requests",
        "1",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        str(args.kv_tokens),
        "--max-total-tokens",
        str(args.kv_tokens),
        "--chunked-prefill-size",
        str(args.chunk),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--max-mamba-cache-size",
        str(args.mamba_slots),
        "--mamba-ssm-dtype",
        "bfloat16",
        "--random-seed",
        "42",
        "--stream-interval",
        "1",
        "--decode-log-interval",
        "64",
    ]
    if args.steps:
        argv += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-draft-load-format",
            "qwen38_mtp",
            "--speculative-num-steps",
            str(args.steps),
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(args.steps + 1),
        ]
    report = {
        "status": "failed",
        "settings": vars(args),
        "command": argv,
        "warmups": [],
        "measurements": [],
    }
    report["input_sha256"] = hashlib.sha256(
        (directory / "inputs.json").read_bytes()
    ).hexdigest()
    process = None
    stopped = threading.Event()
    peak = [0]

    def memory_watch():
        while not stopped.wait(1):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                peak[0] = max(peak[0], int(result.stdout.splitlines()[0]))

    watcher = threading.Thread(target=memory_watch, daemon=True)
    watcher.start()
    log_path = directory / "server.log"
    url = f"http://127.0.0.1:{args.port}"
    turn = {
        "length_policy": "fixed",
        "sampling": {
            "temperature": 0,
            "top_k": -1,
            "top_p": 1,
            "sampling_seed": 42,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        },
    }
    try:
        assert_port_available(args.port)
        env = subprocess_environment()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("startup timeout")
                try:
                    with urllib.request.urlopen(url + "/ready", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    time.sleep(1)
            report["server_info"] = request(url + "/server_info")
            info = report["server_info"]
            expected = {
                "mem_fraction_static": args.mem_fraction,
                "max_total_num_tokens": args.kv_tokens,
                "max_running_requests": 1,
                "chunked_prefill_size": args.chunk,
                "mamba_ssm_dtype": "bfloat16",
                "stream_interval": 1,
                "disable_overlap_schedule": False,
            }
            for name, value in expected.items():
                if info.get(name) != value:
                    raise RuntimeError(
                        f"effective {name}={info.get(name)!r}, expected {value!r}"
                    )
            actual_loader = info["model_loader_extra_config"]
            if isinstance(actual_loader, str):
                actual_loader = json.loads(actual_loader)
            if any(actual_loader.get(key) != value for key, value in loader.items()):
                raise RuntimeError("effective expert-cache configuration differs")
            if args.steps and info.get("speculative_num_steps") != args.steps:
                raise RuntimeError("effective MTP width differs")
            if not args.steps and info.get("speculative_algorithm") is not None:
                raise RuntimeError("target-only control unexpectedly uses speculation")
            for length in args.lengths:
                ids = frozen[str(length)]
                for repetition in range(-1, args.repeats):
                    flush(url)
                    before = log_snapshots(log_path)
                    sample = generate(
                        url,
                        ids,
                        turn,
                        args.timeout,
                        directory / f"request-{length}-{repetition}.json",
                    )
                    if sample["status"] != "ok":
                        raise RuntimeError(f"request failed: {sample['errors']}")
                    metrics = decode_metrics(sample)
                    if metrics["cached_tokens"] not in (None, 0):
                        raise RuntimeError("prefix KV was not cleared")
                    after = log_snapshots(log_path)
                    if before and after:
                        old, new = before[-1]["stats"], after[-1]["stats"]
                        metrics["periodic_counter_delta"] = {
                            key: new.get(key, 0) - old.get(key, 0)
                            for key in (
                                "transferred_bytes",
                                "hits",
                                "misses",
                                "h2d_copy_calls",
                                "adaptive_rebalances",
                            )
                        }
                        metrics["counter_alignment"] = (
                            "periodic snapshots; boundaries can differ by one layer invocation"
                        )
                    row = {"input_tokens": length, "repetition": repetition, **metrics}
                    report["warmups" if repetition < 0 else "measurements"].append(row)
                    write(directory / "report.json", report)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "input_tokens",
                                    "repetition",
                                    "decode_tok_s",
                                    "ttft_s",
                                    "mean_verify_tokens",
                                )
                            }
                        ),
                        flush=True,
                    )
            report["summary"] = {}
            for length in args.lengths:
                rows = [
                    row
                    for row in report["measurements"]
                    if row["input_tokens"] == length
                ]
                report["summary"][str(length)] = {
                    key: {
                        "median": statistics.median([row[key] for row in rows]),
                        "min": min(row[key] for row in rows),
                        "max": max(row[key] for row in rows),
                    }
                    for key in ("decode_tok_s", "ttft_s", "tpot_ms", "e2e_tok_s")
                }
            report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop_owned(process)
        stopped.set()
        watcher.join(5)
        report["sampled_peak_gpu_used_mib"] = peak[0]
        write(directory / "report.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
