"""Reproducible remote-image reuse and real 80K/160K boundary probes."""

import argparse
import base64
import json
import os
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def request(endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        os.environ.get("BENCH_URL", "http://127.0.0.1:8329") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except ValueError:
            return body


def image_url(color):
    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data))
        )

    raw = (b"\0" + bytes(color) * 224) * 224
    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["images", "stages"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = []

    def run(name, endpoint, payload):
        start = time.monotonic()
        response = request(endpoint, payload)
        results.append(
            {"name": name, "wall_s": time.monotonic() - start, "response": response}
        )
        a.output.write_text(json.dumps(results, indent=2) + "\n")
        print(name, results[-1]["wall_s"], flush=True)
        return response

    if a.mode == "images":
        messages = []
        for name, color, salt in [
            ("red-first", (255, 0, 0), "a"),
            ("red-history", (255, 0, 0), "a"),
            ("red-salt", (255, 0, 0), "b"),
            ("blue-changed", (0, 0, 255), "a"),
        ]:
            if name != "red-history":
                messages = []
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(color)}},
                        {
                            "type": "text",
                            "text": "Name the dominant color in this image. Reply with one color word.",
                        },
                    ],
                }
                if name != "red-history"
                else {
                    "role": "user",
                    "content": "What was the color? Reply with one word.",
                }
            )
            response = run(
                name,
                "/v1/chat/completions",
                {
                    "model": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
                    "messages": messages,
                    "max_tokens": 16,
                    "temperature": 0,
                    "cache_salt": salt,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            messages.append(response["choices"][0]["message"])
    else:
        for length in (74000, 170000):
            run(
                f"long-{length}",
                "/generate",
                {
                    "input_ids": [1000 + i % 64 for i in range(length)],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                    },
                },
            )
        run("flush", "/flush_cache", {})
        run(
            "short-reborrow",
            "/generate",
            {
                "text": "Say hello.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        run(
            "regrow-74000",
            "/generate",
            {
                "input_ids": [1000 + i % 64 for i in range(74000)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
        )


if __name__ == "__main__":
    main()
