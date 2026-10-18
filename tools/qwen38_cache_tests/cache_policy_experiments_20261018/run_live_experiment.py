"""Run isolated recordings and fixed-route GPU experiments; restore serving finally."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = Path(
    "/home/user/workspace/prod/homelab-llm/home-jay/remote-mapping/llm/sglang-fp4-local.sh"
)
MANIFEST = "/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json"
BASE_URL = "http://127.0.0.1:8330"


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def request(path, payload=None, timeout=1800):
    req = urllib.request.Request(
        BASE_URL + path,
        None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(url, container, timeout=360):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        state = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            text=True,
        ).strip()
        if state != "true":
            raise RuntimeError(f"{container} exited before readiness")
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{container} readiness timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    config = json.loads(subprocess.check_output(["docker", "inspect", "homelab-llm"]))[
        0
    ]
    image = config["Image"]
    original = LAUNCHER.read_bytes()
    if args.resume:
        saved = json.loads((out / "experiment.json").read_text())
        if (out / "completed.json").exists():
            raise ValueError("Completed experiments cannot be resumed")
        if (
            saved["image"] != image
            or (out / "launcher-before.sh").read_bytes() != original
        ):
            raise ValueError(
                "Recording image or launcher changed; use fresh recordings"
            )
    (out / "launcher-before.sh").write_bytes(original)
    vit_start = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.State.StartedAt}}", "homelab-vit"],
        text=True,
    ).strip()
    metadata = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        image=image,
        image_tag=config["Config"]["Image"],
        budget=6390,
        warmup=128,
        recording_sweeps=1024,
        rounds=args.rounds,
        vit_started_at=vit_start,
        launcher_sha256=hashlib.sha256(original).hexdigest(),
        scope="fixed-route transport; no model compute timed",
    )
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    gpu_script = """#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/sglang/python:/workspace/sglang
export SGLANG_QWEN38_FILL_CTA=adaptive
export SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1
export SGLANG_QWEN38_SOURCE_MODE=shared_thp
export SGLANG_QWEN38_SOURCE_CACHE_DIR=/dev/shm/sglang-qwen38-startup
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /workspace/sglang
exec python "$@"
"""
    (out / "gpu.sh").write_text(gpu_script)
    common = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        "32g",
        "--ulimit",
        "memlock=-1:-1",
        "--volumes-from",
        "homelab-llm",
        "--mount",
        f"type=bind,src={out},dst=/results",
        "--entrypoint",
        "bash",
    ]
    prompts = {
        "coding": "Implement a robust Python LRU cache, then a segmented LRU cache. Explain invariants, eviction, concurrency, testing and complexity, with complete implementations and detailed examples. Continue explaining all design tradeoffs thoroughly.",
        "raft": "请详细设计一个基于 Raft 的分布式键值数据库，依次说明日志复制、选举、持久化、快照、成员变更、线性一致性读取、故障恢复及测试。给出伪代码和具体故障时序，逐步论证安全性与活性。",
    }
    stopped = False
    try:
        print("STOP serving language worker; ViT unchanged", flush=True)
        stopped = True
        run(["docker", "stop", "--time", "45", "homelab-llm"])
        for case, prompt in prompts.items():
            if args.resume and (out / f"{case}-response.json").exists():
                rows = (out / f"{case}.jsonl").read_text().splitlines()
                if len(rows) != 1024 * 48:
                    raise RuntimeError(f"Incomplete saved tape: {case}")
                print("REUSE_COMPLETE_RECORDING", case, flush=True)
                continue
            name = f"qwen38-policy-record-{case}"
            text = original.decode().replace(
                'rootdir=$(dirname -- "$(readlink -f -- "$0")")',
                f"rootdir={LAUNCHER.parent.as_posix()}".replace(
                    "/home/user/workspace/prod", "/prod"
                ),
            )
            text = text.replace(
                'export SGLANG_LOG_FILE="$rootdir/sglang-fp4-local.log"',
                f'export SGLANG_LOG_FILE="/results/{case}-server.log"',
            )
            controls = f"""export QWEN38_ROUTE_RECORD_PATH=/results/{case}.jsonl
export QWEN38_ROUTE_RECORD_ARM=/results/{case}.armed
export QWEN38_ROUTE_RECORD_SWEEPS=1024
export PYTHONPATH=/workspace/sglang/tools/qwen38_cache_tests/cache_policy_experiments_20261018/record_overlay:/workspace/sglang:/workspace/sglang/python
export QWEN38_WARMUP=none
export CUDA_VISIBLE_DEVICES=0
export PORT=8000
export HOST=0.0.0.0
"""
            text = text.replace(
                'exec bash "$rootdir/sglang-flash-next-common.sh" "$@"',
                controls + '\nexec bash "$rootdir/sglang-flash-next-common.sh" "$@"',
            )
            (out / f"record-{case}.sh").write_text(text)
            print("RECORD_START", case, flush=True)
            run(
                common
                + [
                    "--name",
                    name,
                    "--network",
                    config["HostConfig"]["NetworkMode"],
                    "-p",
                    "127.0.0.1:8330:8000",
                    "-d",
                    image,
                    f"/results/record-{case}.sh",
                ]
            )
            try:
                wait_ready(BASE_URL, name)
                info = request("/get_server_info")
                (out / f"{case}-server-info.json").write_text(
                    json.dumps(info, indent=2) + "\n"
                )
                tokenizer_code = """import sys,json
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('/prod/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300',trust_remote_code=True,local_files_only=True)
prompt=json.load(sys.stdin)
print(json.dumps(t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_dict=False)))
"""
                tokenized = subprocess.check_output(
                    ["docker", "exec", "-i", name, "python", "-c", tokenizer_code],
                    input=json.dumps(prompt),
                    text=True,
                )
                payload = dict(
                    input_ids=json.loads(tokenized),
                    stream=False,
                    sampling_params=dict(
                        temperature=0, max_new_tokens=3072, ignore_eos=True
                    ),
                )
                (out / f"{case}-request.json").write_text(
                    json.dumps(
                        dict(prompt=prompt, payload=payload),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                if (out / f"{case}.jsonl").exists():
                    raise RuntimeError("Tape created before the recording request")
                (out / f"{case}.armed").touch()
                response = request("/generate", payload)
                (out / f"{case}-response.json").write_text(
                    json.dumps(response, ensure_ascii=False, indent=2) + "\n"
                )
                rows = (out / f"{case}.jsonl").read_text().splitlines()
                if len(rows) != 1024 * 48:
                    raise RuntimeError(
                        f"{case}: expected 1024 complete sweeps, got {len(rows)} rows"
                    )
                (out / f"{case}-parity.jsonl").write_text(
                    "\n".join(rows[: 256 * 48]) + "\n"
                )
                print("RECORD_DONE", case, len(rows), flush=True)
            finally:
                run(["docker", "stop", "--timeout", "10", name])
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        for case in prompts:
            print("CPU_COST", case, flush=True)
            with (out / f"{case}-cpu.log").open("w") as log:
                run(
                    [
                        "python3",
                        "-m",
                        "tools.qwen38_cache_tests.cache_cost_replay",
                        "--trace",
                        str(out / f"{case}.jsonl"),
                        "--fill-samples",
                        str(
                            ROOT
                            / "tools/qwen38_cache_tests/launch_fill_experiments_20261018/results/fill-prefill-overlap.json.gz"
                        ),
                        "--budget",
                        "6390",
                        "--warmup",
                        "128",
                        "--output",
                        str(out / f"{case}-cpu.json"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            for stage, tape, extra in [
                ("parity", f"{case}-parity.jsonl", ["--rounds", "1", "--correctness"]),
                ("timing", f"{case}.jsonl", ["--rounds", str(args.rounds)]),
            ]:
                print("GPU_START", case, stage, flush=True)
                with (out / f"{case}-{stage}.log").open("w") as log:
                    run(
                        common
                        + [
                            "--name",
                            f"qwen38-policy-gpu-{case}-{stage}",
                            "--network",
                            "none",
                            image,
                            "/results/gpu.sh",
                            "-m",
                            "tools.qwen38_cache_tests.cache_transport_replay",
                            "--trace",
                            f"/results/{tape}",
                            "--manifest",
                            MANIFEST,
                            "--budget",
                            "6390",
                            "--warmup",
                            "128",
                            "--output",
                            f"/results/{case}-{stage}",
                            *extra,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                print("GPU_DONE", case, stage, flush=True)
        (out / "completed.json").write_text(
            json.dumps(dict(status="completed", time=time.time())) + "\n"
        )
    finally:
        for name in ["qwen38-policy-record", "qwen38-policy-gpu"] + [
            f"qwen38-policy-{kind}-{case}{suffix}"
            for case in prompts
            for kind, suffix in [("record", ""), ("gpu", "-parity"), ("gpu", "-timing")]
        ]:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if stopped:
            print("RESTORE language service", flush=True)
            run(["docker", "start", "homelab-llm"])
            wait_ready("http://127.0.0.1:8329", "homelab-llm")
            print("RESTORED healthy language service", flush=True)
        current_vit = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.StartedAt}}", "homelab-vit"],
            text=True,
        ).strip()
        (out / "restored.json").write_text(
            json.dumps(
                dict(language_healthy=True, vit_unchanged=current_vit == vit_start)
            )
            + "\n"
        )
        if LAUNCHER.read_bytes() != original:
            raise RuntimeError(
                "Production launcher changed externally; it was not overwritten"
            )


if __name__ == "__main__":
    main()
