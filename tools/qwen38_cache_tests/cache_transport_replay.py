#!/usr/bin/env python3
"""Fixed-route real-pack GPU transport replay, one fresh process per candidate.

Use correctness mode separately. No model execution, no service speedup claim.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from .cache_cost_replay import read_sweeps


def worker(args):
    import torch
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
        REQUIRED_TENSORS,
        Qwen38ExpertPack,
        load_manifest,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU replay requires CUDA; cannot substitute CPU estimate")
    manifest = load_manifest(str(args.manifest))
    sweeps = read_sweeps(
        args.trace, manifest.num_layers, manifest.num_experts, args.case
    )
    if not 0 <= args.warmup < len(sweeps):
        raise ValueError("invalid warmup prefix")
    if args.layout:
        layout = json.loads(args.layout.read_text())
    else:
        layout = [
            args.budget // manifest.num_layers + (i < args.budget % manifest.num_layers)
            for i in range(manifest.num_layers)
        ]
    if len(layout) != manifest.num_layers or sum(layout) != args.budget:
        raise ValueError("layout must use the exact declared budget")
    if any(
        type(c) is not int or not 3 * manifest.top_k <= c <= manifest.num_experts
        for c in layout
    ):
        raise ValueError(
            "capacity must hold 3-token top-k bound and fit experts; arena excluded"
        )
    pack = Qwen38ExpertPack(manifest)
    runtime = Qwen38ThreadedExpertRuntime(
        pack,
        torch.device("cuda"),
        slots_per_layer=min(layout),
        layer_cache_slots=layout,
        cache_policy="lfu" if args.policy == "lfu" else "slru",
        cache_protected_fraction=0 if args.policy == "lru" else args.protected_fraction,
        adaptive_cache=False,
        stats_log_interval=0,
    )
    try:
        runtime.reserve()
        # Preserve original token order/duplicates, not only unique active sets.
        opener = __import__("gzip").open if str(args.trace).endswith(".gz") else open
        records, case = [], None
        with opener(args.trace, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "start":
                    case = row["case"]
                elif row.get("event") == "end":
                    case = None
                elif "layer" in row and (args.case is None or case == args.case):
                    records.append(row)
        if len(records) != len(sweeps) * manifest.num_layers:
            raise ValueError("record/sweep count mismatch")
        if args.correctness:
            # Payload readback is deliberately separate from performance data.
            # Keep the full warm prefix, then check every layer/tensor for a
            # bounded set of sweeps rather than repeatedly copying hundreds of GB.
            sweeps = sweeps[: args.warmup + args.correctness_sweeps]
            records = records[: len(sweeps) * manifest.num_layers]
        routes = [
            torch.tensor(r["routes"], dtype=torch.int32, device="cuda") for r in records
        ]
        correctness_checks = 0

        def execute(prepared, record):
            nonlocal correctness_checks
            if not args.correctness:
                return
            flat_ids = [e for row in record["routes"] for e in row]
            positions = [flat_ids.index(e) for e in sorted(set(flat_ids)) if e >= 0]
            position = torch.tensor(positions, device="cuda", dtype=torch.long)
            slots = prepared.topk_ids.reshape(-1)[position].long()
            for name in REQUIRED_TENSORS:
                actual = (
                    prepared.tensors[name][slots].cpu().contiguous().view(torch.uint8)
                )
                expected = []
                dtype = getattr(torch, manifest.tensors[record["layer"]][name].dtype)
                for pos in positions:
                    view = pack.expert_tensor_bytes(
                        record["layer"], flat_ids[pos], name
                    )
                    try:
                        expected.append(
                            torch.frombuffer(bytearray(view), dtype=dtype).reshape(
                                manifest.tensors[record["layer"]][name].shape[1:]
                            )
                        )
                    finally:
                        view.release()
                if not torch.equal(
                    actual, torch.stack(expected).contiguous().view(torch.uint8)
                ):
                    raise RuntimeError("real-pack payload mismatch")
                correctness_checks += 1

        boundary = args.warmup * manifest.num_layers
        for ids, row in zip(routes[:boundary], records[:boundary]):
            runtime.run(
                row["layer"], ids, lambda prepared: None, execution_phase="verify"
            )
        runtime.synchronize()
        before = runtime.stats_snapshot()
        started = time.perf_counter()
        for index, (ids, row) in enumerate(
            zip(routes[boundary:], records[boundary:]), 1
        ):
            runtime.run(
                row["layer"],
                ids,
                lambda prepared: execute(prepared, row),
                execution_phase="verify",
            )
            if args.correctness and index % (manifest.num_layers * 16) == 0:
                print(
                    "payload-checked sweeps:",
                    index // manifest.num_layers,
                    "/",
                    len(sweeps) - args.warmup,
                    flush=True,
                )
        runtime.synchronize()
        elapsed = time.perf_counter() - started
        after = runtime.stats_snapshot()
        counters = [
            "hits",
            "misses",
            "transferred_bytes",
            "h2d_copy_calls",
            "decode_direct_misses",
            "decode_direct_hits",
            "decode_device_gate_invocations",
            "decode_device_gate_hits",
            "decode_device_gate_copy_bytes",
            "decode_device_gate_copy_calls",
        ]
        delta = {
            k: after[k] - before[k]
            for k in counters
            if isinstance(before.get(k), (int, float))
            and isinstance(after.get(k), (int, float))
        }
        result = dict(
            status="passed",
            scope="real-pack transport-only; no MoE/attention/draft compute",
            mode="correctness" if args.correctness else "benchmark",
            policy=args.policy,
            protected_fraction=args.protected_fraction,
            layout=layout,
            budget=args.budget,
            warmup_sweeps=args.warmup,
            measured_sweeps=len(sweeps) - args.warmup,
            trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            wall_s=None if args.correctness else elapsed,
            diagnostic_wall_s=elapsed if args.correctness else None,
            correctness_checks=correctness_checks,
            gpu=torch.cuda.get_device_name(),
            before=before,
            after=after,
            counter_delta=delta,
            limits=[
                "Common routes fix MTP-derived expert workload, but draft/attention work omitted",
                "Each process reconstructs cache from empty plus identical warmup prefix",
                "Miss plans intentionally differ across policies; route tape hash and budget fixed",
                "Runtime planner/fill/event path retained; no full-model overlap",
                "Allocation/JIT/warmup excluded except first-use variants not exercised by prefix",
            ],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        runtime.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--case")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--budget", type=int, default=6390)
    p.add_argument("--warmup", type=int, default=128)
    p.add_argument("--layout", type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--correctness-sweeps", type=int, default=16)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for matrix; file for worker",
    )
    p.add_argument("--worker", action="store_true")
    p.add_argument("--policy", choices=["lfu", "lru", "slru"], default="lfu")
    p.add_argument("--protected-fraction", type=float, default=0.7)
    args = p.parse_args()
    if args.correctness_sweeps < 1:
        p.error("positive correctness sweep count required")
    if args.worker:
        worker(args)
        return
    if args.rounds < 1:
        p.error("positive rounds required")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261018)
    runs = []
    for repeat in range(args.rounds):
        candidates = [
            ("lfu", 0.7),
            ("lru", 0),
            ("slru", 0.5),
            ("slru", 0.7),
            ("slru", 0.8),
        ]
        rng.shuffle(candidates)
        for policy, fraction in candidates:
            output = args.output / f"{repeat}-{policy}-{fraction}.json"
            command = [
                sys.executable,
                "-m",
                "tools.qwen38_cache_tests.cache_transport_replay",
                "--worker",
                "--trace",
                str(args.trace),
                "--manifest",
                str(args.manifest),
                "--budget",
                str(args.budget),
                "--warmup",
                str(args.warmup),
                "--policy",
                policy,
                "--protected-fraction",
                str(fraction),
                "--output",
                str(output),
            ]
            if args.case:
                command.extend(["--case", args.case])
            if args.layout:
                command.extend(["--layout", str(args.layout)])
            if args.correctness:
                command.append("--correctness")
                command.extend(["--correctness-sweeps", str(args.correctness_sweeps)])
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            runs.append(
                dict(repeat=repeat, policy=policy, fraction=fraction, file=output.name)
            )
            (args.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
