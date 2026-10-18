"""Capture sixteen decode steps after at least 64 generated warmup tokens."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8329"
SERVER_DIR = "/prod/homelab-llm/models/profiles/launch-fill-selected-20261018"
ROOT = Path(__file__).resolve().parent / "results/profile"


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        data = response.read().decode()
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            # SGLang's profiler control routes return plain text.
            return data


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "server.json").write_text(
        json.dumps(request("/get_server_info"), indent=2) + "\n"
    )
    warmed = threading.Event()

    def generate():
        payload = {
            "text": "Explain CPU cache locality with detailed examples.",
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1024,
                "ignore_eos": True,
            },
        }
        req = urllib.request.Request(
            BASE + "/generate",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        latest = None
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    latest = json.loads(line[6:])
                    if latest.get("meta_info", {}).get("completion_tokens", 0) >= 64:
                        warmed.set()
        return latest

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(generate)
        if not warmed.wait(60):
            if future.done():
                future.result()
            raise RuntimeError("decode warmup did not complete")
        payload = {
            "output_dir": SERVER_DIR,
            "num_steps": 16,
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": False,
            "profile_prefix": "selected-decode",
        }
        started = time.time()
        response = request("/start_profile", payload)
        reply = future.result(timeout=180)
        assert reply["meta_info"]["completion_tokens"] == 1024
        (ROOT / "capture.json").write_text(
            json.dumps(
                {
                    "started_at": started,
                    "start_payload": payload,
                    "start_response": response,
                    "warmup_min_completion_tokens": 64,
                    "request_meta_info": reply["meta_info"],
                    "note": "Profile request is excluded from all throughput measurements.",
                },
                indent=2,
            )
            + "\n"
        )
    print("Captured selected decode profile under", SERVER_DIR)


if __name__ == "__main__":
    main()
