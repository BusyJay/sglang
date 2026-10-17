#!/usr/bin/env python3
"""Small fixed-output diagnostic; run separately from profiling on an idle service."""

import argparse
import json
from pathlib import Path
import time
import urllib.request

from profile_service import request


PROMPTS = [
    "Write a detailed technical explanation of how a CPU cache works, with examples.",
    "用中文详细解释数据库索引的原理以及 B 树和哈希索引的区别。",
    "Implement a Python LRU cache and explain its design step by step.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8329")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", default=192, type=int)
    parser.add_argument("--stream", action="store_true", help="Also measure sustained decode excluding time to first token")
    parser.add_argument("--context-repeat", type=int, default=0, help="Prepend this many copies of 'test context ' for long-context diagnostics")
    parser.add_argument("--case", type=int, choices=range(len(PROMPTS)), action="append")
    args = parser.parse_args()
    result = {"server_info": json.loads(request(args.url, "/get_server_info")), "cases": []}
    for index, prompt in enumerate(PROMPTS):
        if args.case is not None and index not in args.case:
            continue
        prompt = "test context " * args.context_repeat + prompt
        start = time.monotonic()
        payload = {
            "text": prompt, "sampling_params": {
                "temperature": 0, "max_new_tokens": args.tokens, "ignore_eos": True,
            },
        }
        points = []
        if args.stream:
            response = None
            payload["stream"] = True
            req = urllib.request.Request(
                args.url.rstrip("/") + "/generate", json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as stream:
                for line in stream:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    response = json.loads(line[6:])
                    count = response.get("meta_info", {}).get("completion_tokens", 0)
                    if count and (not points or count > points[-1][1]):
                        points.append((time.monotonic() - start, count))
            if response is None or "meta_info" not in response:
                raise RuntimeError(f"Generation stream ended without a result: {response}")
        else:
            response = json.loads(request(args.url, "/generate", payload))
        wall = time.monotonic() - start
        meta = response["meta_info"]
        row = {"case": index, "wall_s": wall, "meta_info": meta,
               "e2e_tokens_per_second": meta["completion_tokens"] / wall}
        if len(points) >= 2:
            first, last = points[0], points[-1]
            row.update(
                ttft_s=first[0],
                decode_tokens_per_second=(last[1] - first[1]) / (last[0] - first[0]),
                stream_points=points,
            )
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "stream_points"}), flush=True)


if __name__ == "__main__":
    main()
