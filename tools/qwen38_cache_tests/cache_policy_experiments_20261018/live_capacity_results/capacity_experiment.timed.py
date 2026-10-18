"""Train capacity layouts, then run real migration and held-out GPU replay.

All environment settings live in gpu.sh. Production launcher is never edited.
"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.qwen38_cache_tests.cache_cost_replay import (
    FillCost,
    aggregate,
    read_json,
    read_sweeps,
    replay_layer,
)
from tools.qwen38_cache_tests.replay_cache import allocate

from .run_live_experiment import LAUNCHER, MANIFEST, ROOT, wait_ready

PREVIOUS = Path(
    "/home/user/workspace/prod/homelab-llm/models/cache-policy-live-20261018-r2"
)
SOURCE_WORKER = ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
ALGORITHMS = ["uniform", "production", "min_misses", "min_fill_ms"]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PREVIOUS / "gpu.sh", out / "gpu.sh")
    samples = (
        ROOT
        / "tools/qwen38_cache_tests/launch_fill_experiments_20261018/results/fill-prefill-overlap.json.gz"
    )
    cost = FillCost(read_json(samples), "mapped_adaptive", "median_gpu_ms")
    initial = [134] * 6 + [133] * 42
    for case in ("coding", "raft"):
        trace = PREVIOUS / f"{case}.jsonl"
        sweeps = read_sweeps(trace, 48, 512)
        assert len(sweeps) == 1024
        events = [[s[layer] for s in sweeps] for layer in range(48)]
        request = dict(
            routes=[[s[layer][0] for s in sweeps[:512]] for layer in range(48)],
            current=initial,
            num_experts=512,
            top_k=10,
            min_slots=10,
            budget=6390,
            policy="lfu",
            protected_fraction=0.7,
            candidate_max=256,
        )
        save(out / f"{case}-production-request.json", request)
        reply = json.loads(
            subprocess.check_output(
                [sys.executable, str(SOURCE_WORKER)],
                input=json.dumps(request) + "\n",
                text=True,
            )
        )
        save(out / f"{case}-production-reply.json", reply)
        capacities = [96, 112, 128, 133, 134, 144, 160, 192, 256]
        print("TRAIN", case, flush=True)
        started = time.perf_counter()
        train = [
            {c: replay_layer(e, c, 512, "lfu", 0.7, cost, 128, 512) for c in capacities}
            for e in events
        ]
        layouts = dict(uniform=initial, production=reply["target"] or initial)
        for name, field in [
            ("min_misses", "misses"),
            ("min_fill_ms", "estimated_fill_ms"),
        ]:
            layouts[name] = allocate(
                [{c: r[field] for c, r in curve.items()} for curve in train], 6390
            )["layer_cache_slots"]
        rows = {}
        for name, layout in layouts.items():
            assert sum(layout) == 6390 and min(layout) >= 30
            save(out / f"{case}-{name}-layout.json", layout)
            # CPU static-cache prediction is diagnostic only. GPU starts with
            # uniform prefix then migrates residents, so states can differ.
            rows[name] = dict(
                layout=layout,
                static_held_out=aggregate(
                    [
                        replay_layer(e, c, 512, "lfu", 0.7, cost, 512, 1024)
                        for e, c in zip(events, layout)
                    ]
                ),
            )
        save(
            out / f"{case}-training.json",
            dict(
                trace=str(trace),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                samples_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),
                variant="mapped_adaptive",
                training_prefix=512,
                training_warmup=128,
                held_out=[512, 1024],
                layouts=rows,
                seconds=time.perf_counter() - started,
            ),
        )
        print("TRAIN_DONE", case, flush=True)


def run_gpu(out, rounds):
    config = json.loads(subprocess.check_output(["docker", "inspect", "homelab-llm"]))[
        0
    ]
    if not config["State"]["Running"]:
        raise RuntimeError("production service was not running before experiment")
    original = LAUNCHER.read_bytes()
    vit_start = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.State.StartedAt}}", "homelab-vit"],
        text=True,
    ).strip()
    save(
        out / "experiment.json",
        dict(
            image=config["Image"],
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            launcher_sha256=hashlib.sha256(original).hexdigest(),
            vit_start=vit_start,
            rounds=rounds,
            seed=20261019,
            algorithms=ALGORITHMS,
            prefix_sweeps=512,
            measured_sweeps=512,
            budget=6390,
            selection="training only; production allocator exact worker with default candidate_max=256",
            scope="transport-only; common uniform warm prefix, real migration, no asynchronous analysis",
        ),
    )
    (out / "launcher-before.sh").write_bytes(original)
    subprocess.run(
        ["nvidia-smi", "-q"], stdout=(out / "hardware.txt").open("w"), check=True
    )
    name = f"qwen38-capacity-{int(time.time())}"
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
        "--name",
        name,
        "--network",
        "none",
        "--entrypoint",
        "bash",
        config["Image"],
        "/results/gpu.sh",
    ]
    runs = []
    stopped = False
    try:
        stopped = True
        subprocess.run(["docker", "stop", "--time", "45", "homelab-llm"], check=True)
        # One Docker container; each candidate is a fresh Python subprocess.
        # Avoid container-name reuse racing asynchronous --rm cleanup.
        command = common + [
            "-m",
            "tools.qwen38_cache_tests.cache_policy_experiments_20261018.capacity_experiment",
            "--inside",
            "--output",
            "/results",
            "--rounds",
            str(rounds),
        ]
        subprocess.run(command, check=True)
        save(out / "completed.json", dict(status="passed", time=time.time()))
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if stopped:
            subprocess.run(["docker", "start", "homelab-llm"], check=True)
            wait_ready("http://127.0.0.1:8329", "homelab-llm", timeout=480)
        vit_after = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.StartedAt}}", "homelab-vit"],
            text=True,
        ).strip()
        if LAUNCHER.read_bytes() != original or vit_after != vit_start:
            raise RuntimeError("launcher or ViT changed during experiment")
        save(
            out / "restored.json",
            dict(
                health=200,
                launcher_sha256=hashlib.sha256(original).hexdigest(),
                vit_start=vit_after,
                time=time.time(),
            ),
        )
        print("RESTORED language service; ViT unchanged", flush=True)


def inside(out, rounds):
    rng = random.Random(20261019)
    runs = []
    for mode in ("correctness", "timing"):
        for repeat in range(1 if mode == "correctness" else rounds):
            candidates = [
                (case, algorithm)
                for case in ("coding", "raft")
                for algorithm in ALGORITHMS
            ]
            rng.shuffle(candidates)
            for case, algorithm in candidates:
                dest = out / f"{mode}-{repeat}-{case}-{algorithm}.json"
                trace = f"/prod/homelab-llm/models/cache-policy-live-20261018-r2/{case}.jsonl"
                command = [
                    sys.executable,
                    "-m",
                    "tools.qwen38_cache_tests.cache_transport_replay",
                    "--worker",
                    "--trace",
                    trace,
                    "--manifest",
                    MANIFEST,
                    "--warmup",
                    "512",
                    "--migration-ready",
                    "--output",
                    str(dest),
                ]
                if algorithm != "uniform":
                    command += [
                        "--migrate-layout",
                        str(out / f"{case}-{algorithm}-layout.json"),
                    ]
                if mode == "correctness":
                    command += ["--correctness", "--correctness-sweeps", "16"]
                print("START", mode, repeat, case, algorithm, flush=True)
                with dest.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=900,
                    )
                result = read_json(dest)
                assert result["status"] == "passed"
                assert result["final_layout"] == read_json(
                    out / f"{case}-{algorithm}-layout.json"
                )
                if mode == "correctness":
                    assert result["correctness_checks"] == 16 * 48 * 6
                runs.append(
                    dict(
                        mode=mode,
                        repeat=repeat,
                        case=case,
                        algorithm=algorithm,
                        file=dest.name,
                    )
                )
                save(out / "runs.json", runs)
                print(
                    "DONE",
                    mode,
                    repeat,
                    case,
                    algorithm,
                    result["wall_s"],
                    result["migration_wall_s"],
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2:
        raise ValueError("at least two repeated blocks required")
    if args.inside:
        inside(args.output, args.rounds)
    elif args.run:
        run_gpu(args.output.resolve(), args.rounds)
    else:
        prepare(args.output.resolve())


if __name__ == "__main__":
    main()
