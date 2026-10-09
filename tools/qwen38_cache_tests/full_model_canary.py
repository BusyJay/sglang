#!/usr/bin/env python3
"""Launch one local model, test requested contexts, and stop only that process.

No HF downloads, no existing server stop, no GPU switching. Run in an already
configured GPU container with local checkpoint and runtime pack mounted.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def request(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ready(url, timeout=5):
    with urllib.request.urlopen(url + "/ready", timeout=timeout) as response:
        return 200 <= response.status < 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ple-backend", choices=["pinned", "file"], default="pinned")
    parser.add_argument("--ple-dir")
    parser.add_argument("--port", type=int, default=30938)
    parser.add_argument(
        "--runtime-mode",
        choices=["sync", "async", "device", "threaded"],
        default="sync",
    )
    parser.add_argument("--slots", type=int, default=56)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--context-length", type=int, default=225280)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=7200)
    args = parser.parse_args()
    if min(args.startup_timeout, args.request_timeout, args.slots, args.chunk) <= 0:
        parser.error("timeouts, slots and chunk must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    if not checkpoint.is_dir() or not manifest.is_file():
        parser.error("local checkpoint and expert-pack manifest are required")
    if any(length < 1 or length + 32 > args.context_length for length in args.contexts):
        parser.error("requested contexts must fit context-length with 32 output tokens")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Config bytes define the requested metadata identity; user pack revision
    # must still correspond to these local checkpoint shards.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False
    )
    phrase_ids = tokenizer.encode(
        "Explain how an expert cache handles a miss. ", add_special_tokens=False
    )
    if not phrase_ids:
        raise ValueError("empty tokenizer fixture")
    extra = {
        "manifest_path": str(manifest),
        "cache_slots_per_layer": args.slots,
        "prefill_arena": True,
        "runtime_mode": args.runtime_mode,
    }
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(checkpoint),
        "--load-format",
        "qwen38_nvfp4_cache",
        "--model-loader-extra-config",
        json.dumps(extra),
        "--language-model-only",
        "--ple-offload-embedding",
        "--disable-cuda-graph",
        "--disable-shared-experts-fusion",
        "--moe-runner-backend",
        "flashinfer_cutlass",
        "--max-running-requests",
        "1",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        str(args.context_length),
        "--chunked-prefill-size",
        str(args.chunk),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.ple_backend == "file" and not args.ple_dir:
        parser.error("file PLE backend requires --ple-dir")
    command += ["--ple-offload-backend", args.ple_backend]
    if args.ple_dir:
        command += ["--ple-offload-dir", args.ple_dir]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    report = {
        "status": "failed",
        "command": command,
        "cases": [],
        "weights_downloaded": False,
    }
    process = None
    try:
        # Avoid attaching to an unrelated process that already occupies the port.
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        with (output / "server.log").open("w") as log:
            print("Starting local SGLang; details in server.log", flush=True)
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            started = time.monotonic()
            url = f"http://127.0.0.1:{args.port}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited with {process.returncode}; inspect server.log"
                    )
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("server startup timed out")
                try:
                    if ready(url):
                        info = request(url + "/get_server_info", timeout=5)
                        report["server_info"] = info
                        break
                except Exception:
                    pass
                time.sleep(2)
            print("Server ready; starting generation cases", flush=True)
            for length in args.contexts:
                print(f"Generating context={length}", flush=True)
                ids = (
                    phrase_ids * ((length + len(phrase_ids) - 1) // len(phrase_ids))
                )[:length]
                begun = time.monotonic()
                response = request(
                    url + "/generate",
                    {
                        "input_ids": ids,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                    timeout=args.request_timeout,
                )
                text = response.get("text")
                if not isinstance(text, str) or not text:
                    raise AssertionError(
                        f"empty generation at context={length}: {response}"
                    )
                case = {
                    "context_tokens": length,
                    "seconds": time.monotonic() - begun,
                    "response": response,
                }
                report["cases"].append(case)
                (output / f"context-{length}.json").write_text(
                    json.dumps(case, indent=2) + "\n"
                )
            report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
        except ProcessLookupError:
            pass
        except Exception as error:
            report["status"] = "failed"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
        (output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
