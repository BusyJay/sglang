#!/usr/bin/env python3
"""Benchmark a manually launched isolated service with one exact prompt/config."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def request(url, path, data=None):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Flush prefix cache before each sample; isolated service only",
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.samples < 1:
        parser.error("tokens >= 2 and samples >= 1 required")
    prompt = args.prompt_file.read_text()
    with request(args.url, "/get_server_info") as response:
        info = json.load(response)
    result = dict(
        variant=args.variant,
        server_info=info,
        sampling_seed=args.seed,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prefix_cache_flushed=args.flush,
        samples=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.samples):
        if args.flush:
            with request(args.url, "/flush_cache?timeout=30", {}):
                pass
        payload = dict(
            text=prompt,
            stream=True,
            sampling_params=dict(
                temperature=0,
                max_new_tokens=args.tokens,
                ignore_eos=True,
                sampling_seed=args.seed,
            ),
        )
        started = time.monotonic()
        points, final = [], None
        with request(args.url, "/generate", payload) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                final = json.loads(line[6:])
                count = final.get("meta_info", {}).get("completion_tokens", 0)
                if count and (not points or count > points[-1][1]):
                    points.append([time.monotonic() - started, count])
        wall = time.monotonic() - started
        if (
            final is None
            or final.get("meta_info", {}).get("completion_tokens") != args.tokens
        ):
            raise RuntimeError(f"Incomplete generation: {final}")
        if len(points) < 2:
            raise RuntimeError("Insufficient stream samples for decode throughput")
        result["samples"].append(
            dict(
                index=index,
                text=final.get("text"),
                meta_info=final["meta_info"],
                ttft_s=points[0][0],
                wall_s=wall,
                decode_tokens_per_second=(points[-1][1] - points[0][1])
                / (points[-1][0] - points[0][0]),
                stream_points=points,
            )
        )
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result["samples"][-1].items()
                    if k not in ("text", "stream_points")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
