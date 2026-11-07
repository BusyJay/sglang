# SPDX-License-Identifier: Apache-2.0
"""Bounded single-request HTTP parity, prefix reuse and streaming observation."""

import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:39007"


def request(route, payload=None):
    req = urllib.request.Request(
        BASE + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        data = response.read()
    return (
        json.loads(data) if data and route not in ("/health", "/flush_cache") else None
    )


def payload(ids, count=32, **extra):
    return {
        "rid": "kda-" + uuid.uuid4().hex,
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": count,
            "ignore_eos": True,
        },
        **extra,
    }


def signature(value):
    return {
        "tokens": value["output_ids"],
        "logprobs": value["meta_info"].get("output_token_logprobs"),
        "top": value["meta_info"].get("output_top_logprobs"),
        "finish": value["meta_info"].get("finish_reason"),
    }


references = json.loads((OUT / "http_references.json").read_text())
baseline = references["baseline"]
prefix_baseline = references["prefix"]
data = {"server_info": request("/get_server_info"), "cases": [], "streaming": []}
for length in (32, 512):
    request("/flush_cache", {})
    first = None
    for warm in (False, True):
        started = time.perf_counter()
        result = request(
            "/generate",
            payload(
                [10, 11, 12, 13] * (length // 4),
                count=8 if length == 512 else 32,
                return_logprob=True,
                top_logprobs_num=3,
            ),
        )
        current = signature(result)
        if first is None:
            first = current
        if length == 32:
            assert first == current
        else:
            # The original commit has a reproducible cold/warm logprob
            # discrepancy here. Require exact parity with each baseline arm.
            assert current == signature(prefix_baseline[int(warm)])
        if length == 32:
            reference = next(c for c in baseline["cases"] if c["label"] == "cold32")
            assert current == reference["signature"]
        if length == 512 and warm:
            assert result["meta_info"]["cached_tokens"] > 0
        data["cases"].append(
            {
                "length": length,
                "warm": warm,
                "seconds": time.perf_counter() - started,
                "result": result,
            }
        )
        print(
            "CASE", length, warm, result["meta_info"].get("cached_tokens"), flush=True
        )

prompt = references["prompt"]
reference = next(
    c for c in baseline["cases"] if c["label"] == "perf" and c["text_id"] == 1
)
for repeat in range(2):
    request("/flush_cache", {})
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload(prompt, 96, stream=True)).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            chunk = line[5:].strip()
            if chunk == b"[DONE]":
                break
            event = json.loads(chunk)
            count = event["meta_info"]["completion_tokens"]
            if count and (not arrivals or count != arrivals[-1][1]):
                arrivals.append([time.perf_counter() - started, count])
    assert event["output_ids"] == reference["event"]["output_ids"]
    row = {
        "repeat": repeat,
        "arrivals": arrivals,
        "event": event,
        "seconds": time.perf_counter() - started,
    }
    row["decode_tok_s"] = (arrivals[-1][1] - arrivals[0][1]) / (
        arrivals[-1][0] - arrivals[0][0]
    )
    data["streaming"].append(row)
    print("STREAM", repeat, row["decode_tok_s"], flush=True)
data["source_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path("python/sglang/srt/hardware_backend/mlx").rglob("*.py")
}
data["complete"] = True
(OUT / "http_smoke.json").write_text(json.dumps(data, indent=2) + "\n")
