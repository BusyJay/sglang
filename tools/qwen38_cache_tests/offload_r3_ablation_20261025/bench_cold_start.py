"""Compare only A using fresh phase-cache keys; source AST is unchanged."""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--request", type=pathlib.Path, required=True)
ap.add_argument("--output-dir", type=pathlib.Path, required=True)
args = ap.parse_args()
repo = pathlib.Path(__file__).resolve().parents[3]
folder = repo / "tools/qwen38_cache_tests/offload_r3_ablation_20261025"
root = args.output_dir
root.mkdir(parents=True, exist_ok=False)
phase = repo / "python/sglang/srt/layers/quantization/qwen38_b12x_fc1_overlap.py"
source = phase.read_bytes()
launch = pathlib.Path(
    "/home/user/workspace/prod/homelab-llm/home-jay/remote-mapping/llm"
)
names = ["sglang-fp4-local.sh", "sglang-flash-next-common.sh"]
original = {n: (launch / n).read_text() for n in names}
# Reuse exactly the archived request parser; execute only its helper definitions.
module = ast.parse((folder / "run_ablation.py").read_text())
ns = dict(
    json=json, urllib=urllib, time=time, hashlib=hashlib, URL="http://127.0.0.1:8329"
)
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "http",
        "percentile",
        "replay",
    ):
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                str(folder / "run_ablation.py"),
                "exec",
            ),
            ns,
        )
payload = json.loads(args.request.read_text())
payload["sampling_params"].update(
    temperature=0, top_k=1, top_p=1, max_new_tokens=4096, ignore_eos=False
)
rows = []
try:
    for enabled in (False, True):
        case = root / ("on" if enabled else "off")
        case.mkdir()
        subprocess.run(
            ["docker", "stop", "-t", "30", "homelab-llm"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        assert phase.read_bytes() == source
        changed = (
            source
            + f"\n# Cold-A compiler cache isolation: {time.time_ns()} {enabled}\n".encode()
        )
        assert ast.dump(ast.parse(changed)) == ast.dump(ast.parse(source))
        phase.write_bytes(changed)
        (case / "phase-source-sha256.txt").write_text(
            hashlib.sha256(changed).hexdigest()
        )
        text = (
            original[names[0]]
            .replace(
                'export SGLANG_OPT_QWEN38_FC1_PRECOMPILE="1"',
                f'export SGLANG_OPT_QWEN38_FC1_PRECOMPILE="{int(enabled)}"',
            )
            .replace(
                "export SGLANG_QWEN38_OBSERVE=${SGLANG_QWEN38_OBSERVE:-0}",
                "export SGLANG_QWEN38_OBSERVE=1",
            )
        )
        (launch / names[0]).write_text(text)
        port = '  --port "$PORT" \\\n'
        (launch / names[1]).write_text(
            original[names[1]].replace(port, port + "  --random-seed 42 \\\n")
        )
        log = launch / "sglang-fp4-local.log"
        offset = log.stat().st_size
        t = time.monotonic()
        print("START cold A", enabled, flush=True)
        subprocess.run(
            ["docker", "start", "homelab-llm"], check=True, capture_output=True
        )
        deadline = time.monotonic() + 900
        while True:
            try:
                ns["http"]("/ready", timeout=2)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("cold A startup")
                time.sleep(2)
        startup = time.monotonic() - t
        print("READY cold A", enabled, startup, flush=True)
        r = ns["replay"](payload)
        (case / "response.txt").write_text(r.pop("response_text"))
        r.update(enabled=enabled, startup_seconds=startup)
        (case / "result.json").write_text(json.dumps(r, indent=2))
        with log.open("rb") as f:
            f.seek(offset)
            (case / "server.log").write_bytes(f.read())
        row = {k: v for k, v in r.items() if k not in ("meta_info", "stream_events")}
        rows.append(row)
        (root / "summary.json").write_text(json.dumps(rows, indent=2))
        print("RESULT", row, flush=True)
        subprocess.run(
            ["docker", "stop", "-t", "30", "homelab-llm"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        assert phase.read_bytes() == changed
        phase.write_bytes(source)
finally:
    subprocess.run(
        ["docker", "stop", "-t", "30", "homelab-llm"], capture_output=True, timeout=60
    )
    phase.write_bytes(source)
    for n, s in original.items():
        (launch / n).write_text(s)
    subprocess.run(["docker", "start", "homelab-llm"], capture_output=True, timeout=60)
