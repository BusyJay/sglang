"""Small, serial HTTP baseline for a running GLM5-Next MLX server.

Uses exact-length synthetic token inputs and fixed-length greedy generation.
TTFT includes HTTP/scheduler overhead; decode rate excludes the first token.
This is a smoke benchmark, not a quality evaluation or a saturation test.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
import urllib.request
from pathlib import Path


def get_json(base_url, route):
    with urllib.request.urlopen(base_url + route, timeout=30) as response:
        return json.load(response)


def measure(base_url, token_ids, output_length):
    payload = {
        "input_ids": token_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_length,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    completed = 0
    metadata = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            metadata = event.get("meta_info", metadata)
            count = metadata.get("completion_tokens", 0)
            if count > completed:
                now = time.perf_counter()
                if first is None:
                    first = now
                    first_count = count
                completed, last = count, now
    elapsed = time.perf_counter() - start
    if completed != output_length or first is None:
        raise RuntimeError(f"Expected {output_length} output tokens, got {metadata}")
    if metadata.get("prompt_tokens") != len(token_ids):
        raise RuntimeError(f"Unexpected prompt length: {metadata}")
    if metadata.get("cached_tokens", 0):
        raise RuntimeError("Baseline requires uncached prompts")
    decode_tokens = completed - first_count
    decode_time = last - first
    return {
        "input_tokens": len(token_ids),
        "output_tokens": completed,
        "first_event_tokens": first_count,
        "ttft_s": first - start,
        "e2e_s": elapsed,
        "decode_tokens_per_s": decode_tokens / decode_time,
        "tpot_ms": 1000 * decode_time / decode_tokens,
        "cached_tokens": metadata.get("cached_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="mlx-community/GLM-5.3-Flash-4bit")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[32, 256, 1024])
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--server-mlx-enable-tf32",
        type=int,
        choices=(0, 1),
        help="Record the server's precision setting; this is not inferred from the client",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_length < 2 or args.repeats < 1 or min(args.input_lengths) < 1:
        parser.error("Require output-length >= 2, repeats >= 1, and positive inputs")

    import mlx.core as mx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    passage = (
        "The team reviews the inference system, measures latency and throughput, "
        "and records reproducible results before changing the implementation. "
    )
    tokens = tokenizer.encode(passage, add_special_tokens=False)
    inputs = {
        n: (tokens * ((n + len(tokens) - 1) // len(tokens)))[:n]
        for n in args.input_lengths
    }
    info = get_json(args.base_url, "/get_server_info")
    server_args = info.get("server_args", info)
    fields = (
        "model_path",
        "context_length",
        "max_total_tokens",
        "max_running_requests",
        "chunked_prefill_size",
        "disable_radix_cache",
        "disable_overlap_schedule",
        "mlx_enable_sampling",
    )
    public_model = "mlx-community/GLM-5.3-Flash-4bit"
    safe_server_args = {key: server_args.get(key) for key in fields}
    if safe_server_args["model_path"] != public_model:
        safe_server_args["model_path"] = "<custom-model>"
    device = mx.device_info()
    report = {
        # Keep only reproducibility metadata: no hostnames, paths, request IDs,
        # exact timestamps, environment dumps, input text or generated responses.
        "platform": {
            "os": platform.system(),
            "os_version": platform.mac_ver()[0],
            "machine": platform.machine(),
        },
        "device": {
            key: device[key]
            for key in (
                "device_name",
                "memory_size",
                "max_recommended_working_set_size",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "torch", "transformers")
        },
        "server": safe_server_args,
        "server_mlx_enable_tf32": args.server_mlx_enable_tf32,
        "input_source": "repeated synthetic English passage, raw token IDs",
        "warmup": measure(args.base_url, inputs[min(inputs)], 8),
        "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n, token_ids in inputs.items():
        for repeat in range(args.repeats):
            result = measure(args.base_url, token_ids, args.output_length)
            result["repeat"] = repeat + 1
            report["trials"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    report["medians"] = []
    for n in inputs:
        trials = [r for r in report["trials"] if r["input_tokens"] == n]
        summary = {"input_tokens": n, "output_tokens": args.output_length}
        for key in ("ttft_s", "e2e_s", "decode_tokens_per_s", "tpot_ms"):
            summary[key] = statistics.median(r[key] for r in trials)
        report["medians"].append(summary)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
