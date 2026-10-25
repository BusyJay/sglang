import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request
from types import SimpleNamespace

URL = "http://127.0.0.1:8329"


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


ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--tag", required=True)
ap.add_argument("--output-dir", type=pathlib.Path, required=True)
ap.add_argument("--request", type=pathlib.Path, required=True)
a = ap.parse_args()
repo = pathlib.Path(__file__).resolve().parents[3]
root = a.output_dir
root.mkdir(parents=True, exist_ok=False)
launch = (
    pathlib.Path.home()
    / "workspace/prod/homelab-llm"
    / ("home-" + pathlib.Path.home().name)
    / "remote-mapping/llm"
)
lnames = ["sglang-fp4-local.sh", "sglang-flash-next-common.sh"]
backup = root / "snapshot"
backup.mkdir(exist_ok=True)
groups = json.loads(
    (
        repo / "tools/qwen38_cache_tests/offload_r3_ablation_20261025/components.json"
    ).read_text()
)
owned = sorted({n for group in groups.values() for n in group})
for n in owned:
    p = backup / n
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((repo / n).read_bytes())
for n in lnames:
    p = backup / n
    if not p.exists():
        p.write_bytes((launch / n).read_bytes())
reference = {n: (backup / n).read_bytes() for n in owned}
base = {
    n: subprocess.check_output(
        ["git", "show", "1ba4deaff878d8faf8bf3c057fd538a9b3052375:" + n], cwd=repo
    )
    for n in owned
}
original = {n: (backup / n).read_text() for n in lnames}
flags = {
    "a": "FC1_PRECOMPILE",
    "b": "SPLIT_FILL",
    "c1": "HC_GATE_OVERLAP",
    "c2": "SHARED_GATE_OVERLAP",
    "u5": "CUTLASS_PREFILL_TUNE",
}
m = SimpleNamespace(http=http, replay=replay)
frozen = a.request
payload = json.loads(frozen.read_text())
payload["sampling_params"].update(
    temperature=0, top_k=1, top_p=1, max_new_tokens=4096, ignore_eos=False
)
(root / "sampling-plan.json").write_text(
    json.dumps(
        dict(
            prompt_tokens=len(payload["input_ids"]),
            sampling_params=payload["sampling_params"],
            original_request_path=str(frozen),
            server_seed=42,
            profiler=False,
            source_groups=groups,
        ),
        indent=2,
    )
)
rows = []


def restore_sources():
    for n in owned:
        current = (repo / n).read_bytes()
        if current not in (reference[n], base[n]):
            raise RuntimeError("Concurrent source edit: " + n)
    for n, data in reference.items():
        (repo / n).write_bytes(data)


try:
    for i, case in enumerate(a.cases):
        folder = root / (a.tag + f"-{i:02d}-" + case)
        folder.mkdir(exist_ok=False)
        component = case.removesuffix("-off")
        assert case == "ref" or component in {*groups, *flags}
        subprocess.run(
            ["docker", "stop", "-t", "30", "homelab-llm"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        restore_sources()
        if component in groups:
            for n in groups[component]:
                (repo / n).write_bytes(base[n])
        text = original[lnames[0]]
        for flag in flags.values():
            needle = f'export SGLANG_OPT_QWEN38_{flag}="1"'
            assert text.count(needle) == 1
            text = text.replace(
                needle,
                f'export SGLANG_OPT_QWEN38_{flag}="{int(flags.get(component) != flag)}"',
            )
        needle = "export SGLANG_QWEN38_OBSERVE=${SGLANG_QWEN38_OBSERVE:-0}"
        assert text.count(needle) == 1
        text = text.replace(needle, "export SGLANG_QWEN38_OBSERVE=0")
        (launch / lnames[0]).write_text(text)
        needle = '  --port "$PORT" \\\n'
        assert original[lnames[1]].count(needle) == 1
        (launch / lnames[1]).write_text(
            original[lnames[1]].replace(needle, needle + "  --random-seed 42 \\\n")
        )
        for n in lnames:
            subprocess.run(["bash", "-n", str(launch / n)], check=True)
        (folder / "source-hashes.json").write_text(
            json.dumps(
                {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in owned},
                indent=2,
            )
        )
        log = launch / "sglang-fp4-local.log"
        offset = log.stat().st_size
        t = time.monotonic()
        print("START", folder.name, flush=True)
        subprocess.run(
            ["docker", "start", "homelab-llm"], check=True, capture_output=True
        )
        deadline = time.monotonic() + 900
        while True:
            try:
                m.http("/ready", timeout=2)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(case + " startup timeout")
                time.sleep(2)
        startup = time.monotonic() - t
        info = json.loads(m.http("/server_info"))
        assert info["random_seed"] == 42
        (folder / "server-info.json").write_text(json.dumps(info, indent=2))
        print("READY", folder.name, round(startup, 2), flush=True)
        r = m.replay(payload)
        (folder / "response.txt").write_text(r.pop("response_text"))
        r.update(case=case, startup_seconds=startup)
        e = r["stream_events"]
        lo = next(x for x in e if x[1] >= 128)
        hi = e[-1]
        r["fixed_window_tps"] = (hi[1] - lo[1]) / (hi[0] - lo[0])
        r["generation_ms_per_verify"] = (
            (r["e2e_seconds"] - r["ttft_seconds"])
            * 1000
            / r["meta_info"]["spec_verify_ct"]
        )
        r["completed_budget"] = r["output_tokens"] == 4096
        (folder / "result.json").write_text(json.dumps(r, indent=2))
        with log.open("rb") as f:
            f.seek(offset)
            raw = f.read()
            (folder / "server.log").write_bytes(raw)
        if any(x in raw for x in [b"Traceback", b"CUDA error", b"OutOfMemoryError"]):
            raise RuntimeError("Inference failure in " + case)
        row = {k: v for k, v in r.items() if k not in ("meta_info", "stream_events")}
        row.update(
            accept=r["meta_info"]["spec_accept_length"],
            cached_tokens=r["meta_info"]["cached_tokens"],
            retractions=r["meta_info"]["num_retractions"],
        )
        rows.append(row)
        (root / (a.tag + "-results.json")).write_text(json.dumps(rows, indent=2))
        print("RESULT", json.dumps(row), flush=True)
finally:
    subprocess.run(
        ["docker", "stop", "-t", "30", "homelab-llm"], capture_output=True, timeout=60
    )
    restore_sources()
    for n, s in original.items():
        (launch / n).write_text(s)
    subprocess.run(["docker", "start", "homelab-llm"], capture_output=True, timeout=60)
