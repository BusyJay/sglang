"""Verify the deployed configuration with an unchanged private Hermes payload."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def http(endpoint, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        URL + endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def replay(payload):
    request = urllib.request.Request(
        URL + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    events, final = [], None
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            final = json.loads(line[6:])
            if "error" in final:
                raise RuntimeError(final["error"])
            count = final.get("meta_info", {}).get("completion_tokens", 0)
            if count and (not events or count > events[-1][1]):
                events.append([time.perf_counter() - started, count])
    if len(events) < 2:
        raise ValueError("Too few output events for performance analysis")
    first, last = events[0], events[-1]
    steady = next((e for e in events if e[1] >= 128), first)
    if last[1] <= steady[1]:
        raise ValueError("Request too short for steady decode measurement")
    gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
    meta = final.get("meta_info", {})
    return {
        "ttft_seconds": first[0],
        "e2e_seconds": time.perf_counter() - started,
        "decode_tokens_per_second": (last[1] - first[1]) / (last[0] - first[0]),
        "steady_after_128_tokens_per_second": (last[1] - steady[1])
        / (last[0] - steady[0]),
        "p95_stream_gap_seconds": percentile(gaps, 0.95),
        "max_stream_gap_seconds": max(gaps),
        "output_tokens": last[1],
        "response_text": final.get("text", ""),
        "output_sha256": hashlib.sha256(final.get("text", "").encode()).hexdigest(),
        "meta_info": meta,
        "stream_events": events,
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--url", default="http://127.0.0.1:8329")
parser.add_argument("--restart", action="store_true")
parser.add_argument("--container", default="homelab-llm")
parser.add_argument(
    "--log",
    type=Path,
    default=Path.home()
    / "workspace/prod/homelab-llm"
    / ("home-" + Path.home().name)
    / "remote-mapping/llm/sglang-fp4-local.log",
)
args = parser.parse_args()
URL = args.url
args.output_dir.mkdir(parents=True, exist_ok=False)
if args.restart:
    subprocess.run(["docker", "restart", args.container], check=True, timeout=90)
deadline = time.monotonic() + 900
while True:
    try:
        http("/ready", timeout=2)
        break
    except OSError:
        if time.monotonic() > deadline:
            raise TimeoutError("production startup timed out")
        time.sleep(2)
info = json.loads(http("/server_info"))
(args.output_dir / "server-info.json").write_text(json.dumps(info, indent=2))
print("READY; verifying unchanged Hermes payload", flush=True)
payload = json.loads(args.request.read_text())
offset = args.log.stat().st_size
r = replay(payload)
(args.output_dir / "response.txt").write_text(r.pop("response_text"))
with args.log.open("rb") as f:
    f.seek(offset)
    (args.output_dir / "server.log").write_bytes(f.read())
with urllib.request.urlopen(URL + "/ready", timeout=10) as response:
    r["ready_status"] = response.status
(args.output_dir / "result.json").write_text(json.dumps(r, indent=2))
print(
    json.dumps({k: v for k, v in r.items() if k not in ("stream_events", "meta_info")}),
    flush=True,
)
