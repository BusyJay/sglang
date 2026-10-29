"""Measure aggregate output throughput of concurrent synthetic GLM requests.

Use with an already running server. Raw token inputs, greedy decoding and
ignore_eos keep every request's input/output lengths fixed. Report end-to-end
aggregate throughput, including prefill and HTTP; this is not decode-only TPOT.
"""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path


def measure(base_url, concurrency, input_length, output_length):
    def generate(index):
        pattern = list(range(10 + index, 14 + index))
        tokens = (pattern * ((input_length + 3) // 4))[:input_length]
        payload = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_length,
                "ignore_eos": True,
            },
        }
        request = urllib.request.Request(
            base_url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        metadata = data["meta_info"]
        if (
            metadata["completion_tokens"] != output_length
            or metadata["prompt_tokens"] != input_length
        ):
            raise RuntimeError(f"Unexpected token counts: {metadata}")
        return {
            "request": index,
            "output_ids": data["output_ids"],
            "completion_tokens": metadata["completion_tokens"],
            "cached_tokens": metadata.get("cached_tokens", 0),
            "e2e_s": time.perf_counter() - start,
        }

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = list(pool.map(generate, range(concurrency)))
    elapsed = time.perf_counter() - start
    total_tokens = sum(r["completion_tokens"] for r in requests)
    return {
        "concurrency": concurrency,
        "total_output_tokens": total_tokens,
        "elapsed_s": elapsed,
        "output_tokens_per_s": total_tokens / elapsed,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=32)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency, args.input_length, args.output_length, args.repeats) < 1:
        parser.error("concurrency, lengths and repeats must be positive")
    report = {
        "method": __doc__,
        "input_tokens_per_request": args.input_length,
        "output_tokens_per_request": args.output_length,
        "warmup": measure(args.base_url, args.concurrency, args.input_length, 4),
        "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(args.repeats):
        result = measure(
            args.base_url, args.concurrency, args.input_length, args.output_length
        )
        result["trial"] = trial + 1
        report["trials"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True
        )
    report["median_output_tokens_per_s"] = statistics.median(
        r["output_tokens_per_s"] for r in report["trials"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
