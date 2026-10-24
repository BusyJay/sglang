"""Deterministic answers through the complete target/draft/acceptance serving path."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--command", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
command = json.loads(Path(args.command).read_text())["command"]
tokenizer = AutoTokenizer.from_pretrained(
    command[command.index("--model-path") + 1], local_files_only=True
)
base = "http://127.0.0.1:30938"
for attempt in range(120):
    try:
        with urllib.request.urlopen(base + "/ready", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(2)
else:
    raise TimeoutError("server readiness")

cases = [
    ("addition", "Only output the number: what is 17 + 25?", "42"),
    (
        "counting",
        "Output the integers from 1 through 30 in order, separated by commas. Output nothing else.",
        list(range(1, 31)),
    ),
    (
        "json",
        'Output exactly this JSON object, without markdown or explanation: {"cache_miss":true,"resident_experts":10}',
        {"cache_miss": True, "resident_experts": 10},
    ),
]
reports = []
for name, prompt, expected in cases:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "get"):
        ids = ids["input_ids"]
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 256},
        "stream": False,
    }
    request = urllib.request.Request(
        base + "/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    text = result["text"].strip()
    if name == "addition":
        actual = text
    elif name == "counting":
        actual = (
            [int(x) for x in text.split(",")]
            if re.fullmatch(r"[\d,\s]+", text)
            else text
        )
    else:
        try:
            actual = json.loads(text)
        except ValueError:
            actual = text
    passed = actual == expected
    reports.append(
        {
            "name": name,
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "response": result,
            "expected": expected,
            "passed": passed,
        }
    )
    Path(args.output).write_text(
        json.dumps({"status": "running", "cases": reports}, indent=2)
    )
    print(name, passed, repr(text), flush=True)
status = "passed" if all(c["passed"] for c in reports) else "failed"
Path(args.output).write_text(json.dumps({"status": status, "cases": reports}, indent=2))
assert status == "passed", status
