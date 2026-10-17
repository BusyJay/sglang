#!/usr/bin/env python3
"""Queue a long prefill during sustained decode on a real service."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8329")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=10240)
    parser.add_argument("--prefix-tag", default="", help="Avoid prefix reuse on repeat runs")
    args = parser.parse_args()
    decoding = threading.Event()
    report = {"started_at": time.time(), "decode_tokens": args.decode_tokens}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def generate(text, tokens, signal=False):
        payload = {"text": text, "stream": True, "sampling_params": {
            "temperature": 0, "max_new_tokens": tokens, "ignore_eos": True,
        }}
        started = time.monotonic()
        request = urllib.request.Request(
            args.url + "/generate", json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        first = None
        result = None
        with urllib.request.urlopen(request, timeout=600) as stream:
            for line in stream:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                result = json.loads(line[6:])
                if "error" in result:
                    raise RuntimeError(result["error"])
                count = result.get("meta_info", {}).get("completion_tokens", 0)
                if count and first is None:
                    first = time.monotonic() - started
                if signal and count >= 256:
                    decoding.set()
        if result is None or result.get("meta_info", {}).get("completion_tokens") != tokens:
            raise RuntimeError("Incomplete generation")
        return {"wall_seconds": time.monotonic() - started,
                "first_token_seconds": first, "meta_info": result["meta_info"]}

    first_text = (
        "The system processes database queries, memory pages, and network requests. " * 1800
        + "\nWrite a detailed technical report about this system."
    )
    second_text = (
        "A compiler translates source code into machine instructions and optimizes loops. " * 3600
        + "\nExplain the role of register allocation."
    )
    if args.prefix_tag:
        first_text = args.prefix_tag + "\n" + first_text
        second_text = args.prefix_tag + "\n" + second_text
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(generate, first_text, args.decode_tokens, True)
            if not decoding.wait(180):
                raise RuntimeError("First request did not reach sustained decode")
            if first.done():
                raise RuntimeError("Decode completed before the second request was queued")
            report["long_prefill_queued_at"] = time.time()
            save()
            print("Long prefill queued while first request is decoding", flush=True)
            second = pool.submit(generate, second_text, 64)
            report["decode"] = first.result()
            save()
            print("Sustained decode completed", flush=True)
            report["long_prefill"] = second.result()
        if report["long_prefill"]["meta_info"]["prompt_tokens"] < 40000:
            raise RuntimeError("Long prefill was too short for this regression")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        report["finished_at"] = time.time()
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
