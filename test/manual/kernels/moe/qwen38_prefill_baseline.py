#!/usr/bin/env python3
"""Collect full-model prefill baselines on an isolated existing server.

Requires real local token IDs. Cache flushing is explicit and never implicit.
No model/service restart or optimization is performed.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def request(url, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument(
        "--input-ids",
        type=Path,
        required=True,
        help="JSON array of >= max length real token IDs",
    )
    p.add_argument(
        "--lengths", nargs="+", type=int, default=[4096, 8192, 16384, 32768, 46000]
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush server prefix cache before EACH request; use isolated worker",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = json.loads(args.input_ids.read_text())
    if not isinstance(ids, list) or any(not isinstance(v, int) or v < 0 for v in ids):
        p.error("invalid token IDs")
    if args.rounds < 1 or min(args.lengths) < 1 or max(args.lengths) > len(ids):
        p.error("invalid rounds or insufficient token IDs")
    base = args.url.rstrip("/")
    info = request(base + "/get_server_info")
    result = dict(
        status="running",
        server=info,
        rounds=args.rounds,
        cache_state=(
            "explicit flush before every request"
            if args.flush_cache
            else "prefix cache retained; cached_tokens reported"
        ),
        requests=[],
        limits=[
            "TTFT includes scheduling, prefill and first decode; pair with stage-specific profile",
            "Use same worker argv, cache/KV budget and token IDs across candidates",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        for repeat in range(args.rounds):
            if args.flush_cache:
                flushed = request(base + "/flush_cache", {})
                if flushed is False:
                    raise RuntimeError("server rejected cache flush")
            started = time.perf_counter()
            reply = request(
                base + "/generate",
                dict(
                    input_ids=ids[:length],
                    sampling_params=dict(
                        temperature=0, max_new_tokens=1, ignore_eos=True
                    ),
                    stream=False,
                ),
            )
            wall = time.perf_counter() - started
            meta = reply["meta_info"]
            if (
                meta.get("completion_tokens") != 1
                or meta.get("prompt_tokens") != length
            ):
                raise RuntimeError("request token count mismatch")
            if args.flush_cache and meta.get("cached_tokens", 0) != 0:
                raise RuntimeError("prefix cache was reused despite requested flush")
            result["requests"].append(
                dict(
                    length=length,
                    repeat=repeat,
                    wall_s=wall,
                    meta_info=meta,
                    text=reply.get("text"),
                    uncached_input_tokens=length - meta.get("cached_tokens", 0),
                )
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(length, repeat, wall, meta.get("cached_tokens"), flush=True)
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
