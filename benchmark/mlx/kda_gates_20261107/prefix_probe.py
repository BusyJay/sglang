# SPDX-License-Identifier: Apache-2.0
"""Record exact-chunk cold/warm prefix signatures for an isolated service."""

import argparse
import json
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()


def request(route, body):
    req = urllib.request.Request(
        "http://127.0.0.1:39007" + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read()
    return json.loads(raw) if route != "/flush_cache" else None


request("/flush_cache", {})
rows = []
for _ in range(2):
    result = request(
        "/generate",
        {
            "rid": uuid.uuid4().hex,
            "input_ids": [10, 11, 12, 13] * 128,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 3,
        },
    )
    rows.append(result)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    print(
        result["meta_info"]["cached_tokens"],
        result["output_ids"],
        result["meta_info"]["output_token_logprobs"][0],
        flush=True,
    )
