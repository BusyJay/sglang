"""Compare C2 consumer 16/8 warps with the unchanged private Hermes request."""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--request", type=pathlib.Path, required=True)
ap.add_argument("--output-dir", type=pathlib.Path, required=True)
args = ap.parse_args()
repo = pathlib.Path(__file__).resolve().parents[3]
args.output_dir.mkdir(parents=True, exist_ok=False)
helper = (
    repo / "tools/qwen38_cache_tests/offload_r3_ablation_20261025/verify_production.py"
)
ns = dict(
    json=json, urllib=urllib, time=time, hashlib=hashlib, URL="http://127.0.0.1:8329"
)
for node in ast.parse(helper.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "http",
        "percentile",
        "replay",
    ):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), "exec"), ns)
source_path = repo / "python/sglang/kernels/ops/moe/shared_expert_gate.py"
source = source_path.read_bytes()
old = subprocess.check_output(
    ["git", "show", "e4160f8222:python/sglang/kernels/ops/moe/shared_expert_gate.py"],
    cwd=repo,
)
launch = (
    pathlib.Path.home()
    / "workspace/prod/homelab-llm"
    / ("home-" + pathlib.Path.home().name)
    / "remote-mapping/llm"
)
common = launch / "sglang-flash-next-common.sh"
original = common.read_bytes()
needle = b'  --port "$PORT" \\\n'
assert original.count(needle) == 1
payload = json.loads(args.request.read_text())
rows = []
try:
    for index, arm in enumerate(("old", "new", "new", "old")):
        case = args.output_dir / f"{index:02d}-{arm}"
        case.mkdir()
        subprocess.run(
            ["docker", "stop", "-t", "30", "homelab-llm"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        assert source_path.read_bytes() in (source, old)
        source_path.write_bytes(old if arm == "old" else source)
        common.write_bytes(
            original.replace(needle, needle + b"  --random-seed 42 \\\n")
        )
        log = launch / "sglang-fp4-local.log"
        offset = log.stat().st_size
        started = time.monotonic()
        subprocess.run(
            ["docker", "start", "homelab-llm"], check=True, capture_output=True
        )
        while True:
            try:
                ns["http"]("/ready", timeout=2)
                break
            except OSError:
                if time.monotonic() - started > 900:
                    raise TimeoutError("startup")
                time.sleep(2)
        print("READY", arm, flush=True)
        (case / "server-info.json").write_bytes(ns["http"]("/server_info"))
        result = ns["replay"](payload)
        (case / "response.txt").write_text(result.pop("response_text"))
        result.update(
            arm=arm,
            index=index,
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        )
        with log.open("rb") as f:
            f.seek(offset)
            raw = f.read()
        (case / "server.log").write_bytes(raw)
        assert not any(
            x in raw for x in (b"Traceback", b"CUDA error", b"OutOfMemoryError")
        )
        meta = result["meta_info"]
        (case / "result.json").write_text(json.dumps(result, indent=2))
        print("META", json.dumps(meta), flush=True)
        assert meta["prompt_tokens"] == 25560
        assert meta["cached_tokens"] == 0 and meta["num_retractions"] == 0
        assert result["output_tokens"] <= 5895
        result["generation_ms_per_verify"] = (
            (result["e2e_seconds"] - result["ttft_seconds"])
            * 1000
            / meta["spec_verify_ct"]
        )
        (case / "result.json").write_text(json.dumps(result, indent=2))
        compact = {k: v for k, v in result.items() if k not in ("stream_events",)}
        rows.append(compact)
        (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2))
        print(
            "RESULT",
            json.dumps({k: v for k, v in compact.items() if k != "meta_info"}),
            flush=True,
        )
finally:
    subprocess.run(
        ["docker", "stop", "-t", "30", "homelab-llm"], capture_output=True, timeout=60
    )
    assert source_path.read_bytes() in (source, old)
    source_path.write_bytes(source)
    common.write_bytes(original)
    subprocess.run(["docker", "start", "homelab-llm"], check=True, capture_output=True)
