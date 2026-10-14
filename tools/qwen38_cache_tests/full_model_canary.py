#!/usr/bin/env python3
"""Launch one local model, test requested contexts, and stop only that process.

No HF downloads, no existing server stop, no GPU switching. Run in an already
configured GPU container with local checkpoint and runtime pack mounted.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _visible_generation_text(text):
    """Drop any wrapped hidden-reasoning span so only the visible reply remains."""
    stripped = re.sub(r"<think[^>]*>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<thinking[^>]*>.*?</thinking>", "", stripped, flags=re.DOTALL)
    for closing_tag in ("</think>", "</thinking>"):
        close_pos = stripped.rfind(closing_tag)
        if close_pos != -1:
            stripped = stripped[close_pos + len(closing_tag) :]
    return stripped.strip()


def _encode_probe_prompt(tokenizer, prompt, context_length):
    """Render a user prompt through the chat template when available."""
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt_text = prompt
    encoded = tokenizer.encode(prompt_text, add_special_tokens=False)[:context_length]
    if not encoded:
        raise ValueError(f"probe prompt encoded to zero tokens: {prompt!r}")
    return prompt_text, encoded


def _probe_case(prompt, pattern, tokenizer, context_length):
    """Build a report entry for a semantic correctness probe."""
    prompt_text, encoded = _encode_probe_prompt(tokenizer, prompt, context_length)
    return {
        "prompt": prompt,
        "answer_pattern": pattern,
        "prompt_text": prompt_text,
        "prompt_tokens": len(encoded),
    }, encoded


def _finalize_probe_case(case, response, elapsed_seconds):
    """Attach raw and visible generations plus the regex match result."""
    raw_text = response.get("text") or ""
    visible_text = _visible_generation_text(raw_text)
    matched = re.search(case["answer_pattern"], visible_text) is not None
    case.update(
        seconds=elapsed_seconds,
        raw_response_text=raw_text[:800],
        response_text=visible_text[:400],
        matched=matched,
        meta_info=response.get("meta_info"),
    )
    return matched


def request(url, data=None, timeout=30):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ready(url, timeout=5):
    with urllib.request.urlopen(url + "/ready", timeout=timeout) as response:
        return 200 <= response.status < 300


def _extract_cache_stats(server_log_path):
    marker = "Qwen38ExpertCacheStats["
    latest = None
    if not server_log_path.is_file():
        return None
    for line in server_log_path.read_text(errors="replace").splitlines():
        marker_pos = line.find(marker)
        if marker_pos == -1:
            continue
        payload_pos = line.find("{", marker_pos)
        if payload_pos == -1:
            continue
        payload = line[payload_pos:].strip()
        try:
            latest = json.loads(payload)
        except json.JSONDecodeError:
            continue
    return latest


def _extract_decode_throughput(server_log_path):
    marker = "gen throughput (token/s):"
    samples = []
    if not server_log_path.is_file():
        return None
    for line in server_log_path.read_text(errors="replace").splitlines():
        marker_pos = line.find(marker)
        if marker_pos == -1:
            continue
        payload = line[marker_pos + len(marker) :]
        value_text = payload.split(",", 1)[0].strip()
        try:
            samples.append(float(value_text))
        except ValueError:
            continue
    if not samples:
        return None
    return {
        "samples": len(samples),
        "avg_token_per_second": sum(samples) / len(samples),
        "max_token_per_second": max(samples),
        "last_token_per_second": samples[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", help="HF revision shared by config, tokenizer and weights")
    parser.add_argument("--cache-dir", help="existing HF hub cache directory")
    parser.add_argument("--served-model-name", help="API model ID; defaults to checkpoint argument")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ple-backend", choices=["pinned", "file", "lazy"], default="pinned")
    parser.add_argument("--ple-dir")
    parser.add_argument("--port", type=int, default=30938)
    parser.add_argument(
        "--runtime-mode",
        choices=["sync", "async", "device", "threaded"],
        default="sync",
    )
    slot_options = parser.add_mutually_exclusive_group()
    slot_options.add_argument("--slots", type=int, default=None,
                              help="legacy capacity per layer (default 56)")
    slot_options.add_argument("--total-slots", type=int,
                              help="total expert cache slots across all 48 layers")
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--context-length", type=int, default=225280)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--staging-buffers", type=int, default=4)
    parser.add_argument("--staging-buffer-mib", type=int, default=64)
    parser.add_argument("--warm-decode-cache", action="store_true")
    parser.add_argument(
        "--adaptive-cache",
        action="store_true",
        help="live fixed-budget per-layer expert-cache allocation",
    )
    parser.add_argument(
        "--adaptive-cache-window",
        type=int,
        default=512,
        help="decode samples per layer before evaluating a new allocation",
    )
    parser.add_argument(
        "--adaptive-cache-interval",
        type=int,
        default=4096,
        help="unsampled single-token decode calls per layer between contiguous windows",
    )
    parser.add_argument(
        "--adaptive-cache-mode",
        choices=["rebalance", "observe", "sample"],
        default="rebalance",
    )
    parser.add_argument("--cache-policy", choices=["lfu", "slru"], default="lfu")
    parser.add_argument("--cache-protected-fraction", type=float, default=0.7)
    parser.add_argument("--no-prefill-reuse-cache", action="store_true")
    parser.add_argument("--no-prefill-seed-cache", action="store_true")
    parser.add_argument(
        "--encoder-urls",
        nargs="+",
        help="existing ViT servers on other GPUs; enables language-only EPD",
    )
    parser.add_argument("--encoder-transfer-backend", default="zmq_to_scheduler")
    parser.add_argument("--image", help="local image for remote-ViT OpenAI chat probe")
    parser.add_argument("--image-prompt", default="Describe the image briefly.")
    parser.add_argument("--image-answer-pattern", help="regex checked against the visible image answer")
    parser.add_argument("--image-max-new-tokens", type=int, default=1024)
    parser.add_argument("--decode-graph", action="store_true",
                        help="opt into batch-1 breakable decode graph (threaded runtime only)")
    parser.add_argument("--offload-startup-warmup", action="store_true",
                        help="run representative offload prefill/decode before serving")
    parser.add_argument("--layer-cache-slots", help="JSON file containing a capacity list")
    parser.add_argument("--route-trace-path", help="profiling-only JSONL path inside server environment")
    parser.add_argument("--route-trace-limit", type=int, default=10000)
    parser.add_argument("--context-max-new-tokens", type=int, default=32)
    parser.add_argument("--context-ignore-eos", action="store_true")
    # Memory knobs mirroring a known-good deployment profile; omitted by
    # default so auto-sizing still applies when callers want it.
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--max-mamba-cache-size", type=int)
    parser.add_argument("--mamba-full-memory-ratio", type=float)
    parser.add_argument(
        "--probe",
        action="append",
        default=None,
        metavar='"PROMPT|EXPECTED_ANSER_PATTERN"',
        help="semantic correctness probe checked after the context cases; repeatable",
    )
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=7200)
    parser.add_argument("--cache-stats-log-interval", type=int, default=0)
    parser.add_argument("--mtp-steps", type=int, choices=[1, 2, 3],
                        help="opt-in same-card eager MTP; verify width is steps + 1")
    args = parser.parse_args()
    if args.mtp_steps and (args.runtime_mode != "threaded" or args.encoder_urls):
        parser.error("MTP requires threaded and pure text")
    if args.mtp_steps and (args.decode_graph or args.adaptive_cache):
        parser.error("MTP graph/adaptive require separate compatibility patches")
    if args.total_slots is not None and args.layer_cache_slots:
        parser.error("--total-slots cannot be combined with --layer-cache-slots")
    if args.total_slots is not None and not 480 <= args.total_slots <= 24576:
        parser.error("--total-slots must be in [480, 24576]")
    if args.image and not args.encoder_urls:
        parser.error("--image requires --encoder-urls (ViT is not loaded locally)")
    if args.image_answer_pattern and not args.image:
        parser.error("--image-answer-pattern requires --image")
    if args.image_max_new_tokens <= 0:
        parser.error("--image-max-new-tokens must be positive")
    if args.route_trace_limit < 1:
        parser.error("route trace limit must be positive")
    if args.warm_decode_cache and args.runtime_mode not in ("device", "threaded"):
        parser.error("decode warmup requires device/threaded mode")
    if args.adaptive_cache and args.runtime_mode != "threaded":
        parser.error("adaptive cache requires threaded mode")
    if args.adaptive_cache_window < 16:
        parser.error("adaptive cache window must be at least 16")
    if args.adaptive_cache_interval < 0 or not 0 <= args.cache_protected_fraction < 1:
        parser.error("invalid adaptive interval or protected fraction")
    if args.cache_policy == "slru" and args.runtime_mode != "threaded":
        parser.error("SLRU requires threaded mode")
    if args.decode_graph and args.runtime_mode != "threaded":
        parser.error("--decode-graph requires --runtime-mode threaded")
    if not 2 <= args.staging_buffers <= 64 or args.staging_buffer_mib < 0:
        parser.error("staging buffers must be 2..64 and staging buffer MiB non-negative")
    if min(
        args.startup_timeout,
        args.request_timeout,
        args.slots if args.slots is not None else 56,
        args.chunk,
        args.context_max_new_tokens,
    ) <= 0:
        parser.error("timeouts, slots, chunk and context max_new_tokens must be positive")
    if args.cache_stats_log_interval < 0:
        parser.error("cache stats log interval must be non-negative")
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        parser.error("local expert-pack manifest is required")
    from sglang.srt.model_loader.qwen38_nvfp4_common import (
        resolve_cached_checkpoint, cached_checkpoint_weight_files,
        validate_cached_checkpoint_revision,
    )

    checkpoint = resolve_cached_checkpoint(
        args.checkpoint, revision=args.revision, cache_dir=args.cache_dir
    )
    cached_checkpoint_weight_files(checkpoint)
    pack_revision = json.loads(manifest.read_text())["checkpoint_revision"]
    validate_cached_checkpoint_revision(checkpoint, object(), pack_revision)
    if any(
        length < 1 or length + args.context_max_new_tokens > args.context_length
        for length in args.contexts
    ):
        parser.error("requested contexts must fit context-length with context max_new_tokens")
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
        "prefill_arena": True,
        "runtime_mode": args.runtime_mode,
    }
    if args.total_slots is not None:
        extra["cache_slots_total"] = args.total_slots
    else:
        extra["cache_slots_per_layer"] = args.slots if args.slots is not None else 56
    if args.cache_stats_log_interval:
        extra["stats_log_interval"] = args.cache_stats_log_interval
    if args.runtime_mode != "sync":
        extra["staging_buffers"] = args.staging_buffers
        extra["staging_buffer_bytes"] = args.staging_buffer_mib << 20
    if args.warm_decode_cache:
        extra["warm_decode_cache"] = True
    if args.adaptive_cache:
        extra["adaptive_cache"] = True
        extra["adaptive_cache_window"] = args.adaptive_cache_window
        extra["adaptive_cache_interval"] = args.adaptive_cache_interval
        extra["adaptive_cache_mode"] = args.adaptive_cache_mode
    if args.runtime_mode == "threaded":
        extra["cache_policy"] = args.cache_policy
        extra["cache_protected_fraction"] = args.cache_protected_fraction
        extra["prefill_reuse_cache"] = not args.no_prefill_reuse_cache
        extra["prefill_seed_cache"] = not args.no_prefill_seed_cache
    if args.layer_cache_slots:
        extra["layer_cache_slots"] = json.loads(Path(args.layer_cache_slots).read_text())
    if args.route_trace_path:
        extra["route_trace_path"] = args.route_trace_path
        extra["route_trace_limit"] = args.route_trace_limit
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.checkpoint,
        "--served-model-name",
        args.served_model_name or args.checkpoint,
        "--reasoning-parser",
        "qwen3",
        "--load-format",
        "qwen38_nvfp4_cache",
        "--model-loader-extra-config",
        json.dumps(extra),
        "--ple-offload-embedding",
        "--cuda-graph-backend-prefill=disabled",
        "--cuda-graph-backend-decode=" + ("breakable" if args.decode_graph else "disabled"),
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
    if args.encoder_urls:
        command += ["--language-only", "--encoder-urls", *args.encoder_urls,
                    "--encoder-transfer-backend", args.encoder_transfer_backend]
    else:
        command += ["--language-model-only"]
    if args.ple_backend == "file" and not args.ple_dir:
        parser.error("file PLE backend requires --ple-dir")
    if args.revision:
        command += ["--revision", args.revision]
    if args.mtp_steps:
        command += ["--speculative-algorithm", "NEXTN",
                    "--speculative-draft-load-format", "qwen38_mtp",
                    "--speculative-num-steps", str(args.mtp_steps),
                    "--speculative-eagle-topk", "1",
                    "--speculative-num-draft-tokens", str(args.mtp_steps + 1)]
    if args.offload_startup_warmup:
        command += ["--warmups", "qwen38_offload"]
    if args.cache_dir:
        command += ["--download-dir", args.cache_dir]
    if args.mem_fraction_static is not None:
        command += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.max_mamba_cache_size is not None:
        command += ["--max-mamba-cache-size", str(args.max_mamba_cache_size)]
    if args.mamba_full_memory_ratio is not None:
        command += ["--mamba-full-memory-ratio", str(args.mamba_full_memory_ratio)]
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
    server_log_path = output / "server.log"
    try:
        # Avoid attaching to an unrelated process that already occupies the port.
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        with server_log_path.open("w") as log:
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
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": args.context_max_new_tokens,
                            "ignore_eos": args.context_ignore_eos,
                        },
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
        # Semantic correctness probes: real prompts with expected answers, kept
        # distinct from the repetition fixtures above which only prove serving.
        for probe in args.probe or []:
            prompt, _, pattern = probe.partition("|")
            if not prompt or not pattern:
                raise ValueError(f"probe must be PROMPT|ANSWER_REGEX: {probe!r}")
            case, encoded = _probe_case(prompt, pattern, tokenizer, args.context_length)
            begun = time.monotonic()
            response = request(
                url + "/generate",
                {
                    "input_ids": encoded,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                },
                timeout=args.request_timeout,
            )
            matched = _finalize_probe_case(case, response, time.monotonic() - begun)
            report["cases"].append(case)
            print(f"Probe {prompt!r} -> {'MATCH' if matched else 'NO MATCH'}", flush=True)
            if not matched:
                report["error"] = f"semantic probe failed: {case}"
                raise AssertionError(report["error"])
        if args.image:
            image_path = Path(args.image)
            mime = mimetypes.guess_type(image_path.name)[0]
            if mime not in ("image/png", "image/jpeg", "image/webp"):
                raise ValueError("image probe requires local PNG/JPEG/WebP")
            data_url = "data:" + mime + ";base64," + base64.b64encode(image_path.read_bytes()).decode()
            begun = time.monotonic()
            response = request(url + "/v1/chat/completions", {
                "model": args.served_model_name or args.checkpoint,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": args.image_prompt},
                ]}],
                "temperature": 0, "max_tokens": args.image_max_new_tokens, "stream": False,
            }, timeout=args.request_timeout)
            visible = _visible_generation_text(response["choices"][0]["message"].get("content") or "")
            if not visible:
                raise AssertionError("empty visible answer for remote ViT probe")
            matched = not args.image_answer_pattern or re.search(args.image_answer_pattern, visible) is not None
            report["cases"].append({"image": str(image_path), "response_text": visible,
                "matched": matched, "seconds": time.monotonic() - begun,
                "usage": response.get("usage")})
            if not matched:
                raise AssertionError("remote ViT answer pattern did not match")
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
        cache_stats = _extract_cache_stats(server_log_path)
        if cache_stats is not None:
            report["cache_stats"] = cache_stats
        decode_throughput = _extract_decode_throughput(server_log_path)
        if decode_throughput is not None:
            report["decode_throughput"] = decode_throughput
        (output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
