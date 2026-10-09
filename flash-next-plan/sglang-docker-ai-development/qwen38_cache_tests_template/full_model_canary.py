#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import urllib.request


TESTS = [
    ("add", "不要解释，只输出一个整数：37 + 58 = ?", "95"),
    ("multiply", "不要解释，只输出一个整数：17 × 23 = ?", "391"),
]


def request(base_url: str, prompt: str) -> str:
    payload = {
        "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read())
    return result["choices"][0]["message"]["content"]


def main() -> int:
    base_url = os.environ["SGLANG_TEST_BASE_URL"]
    results = []
    for name, prompt, expected in TESTS:
        text = request(base_url, prompt)
        numbers = re.findall(r"-?\d+", text.replace(",", ""))
        actual = numbers[-1] if numbers else None
        results.append(
            {
                "name": name,
                "passed": actual == expected,
                "expected": expected,
                "actual": actual,
                "raw": text,
            }
        )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

