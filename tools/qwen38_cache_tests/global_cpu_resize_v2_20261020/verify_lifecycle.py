#!/usr/bin/env python3
"""Require actual CPU work and two bidirectional cache-resize serving cycles."""

import argparse
import json
from pathlib import Path


def verify(directory, top_k=10):
    report = json.loads((directory / "report.json").read_text())
    log = (directory / "server.log").read_text(errors="replace")
    snapshots, enabled = [], []
    for line in log.splitlines():
        if "Qwen38 CPU hybrid enabled:" in line:
            enabled.append(line)
        if "Qwen38ExpertCacheStats[" in line:
            snapshots.append(json.loads(line[line.index("{") :]))
    errors, transitions = [], []
    page_size = report["server_info"]["page_size"]
    if report.get("status") != "passed":
        errors.append("generation canary did not pass")
    if not enabled:
        errors.append("no CPU hybrid initialization evidence")
    if "cuda graph: True" not in log:
        errors.append("no serving CUDA graph replay evidence")
    for snapshot in snapshots:
        stage = snapshot.get("kv_stage")
        slots = snapshot.get("global_cache_slots")
        backed = snapshot.get("kv_backed_tokens")
        if stage is None or slots is None:
            continue
        budgets = snapshot["expert_loan_budgets"]
        capacities = snapshot["kv_cache_stages"]
        capacity = (capacities[stage] + page_size - 1) // page_size * page_size
        if slots != budgets[stage] or backed != capacity:
            errors.append(f"inconsistent physical stage: {stage=}, {slots=}, {backed=}")
        if not transitions or stage != transitions[-1]["stage"]:
            transitions.append(
                dict(
                    stage=stage,
                    expert_slots=slots,
                    kv_tokens=backed,
                    runs=snapshot["run_invocations"],
                )
            )
    stages = [item["stage"] for item in transitions]
    if set(stages) != {0, 1, 2}:
        errors.append(f"three physical stages were not all observed: {stages}")
    if stages.count(2) < 2 or sum(a > b for a, b in zip(stages, stages[1:])) < 2:
        errors.append(f"two long/short return cycles were not observed: {stages}")
    if not stages or stages[-1] != 0:
        errors.append(
            "expert cache did not recover the maximum budget after short traffic"
        )
    final = snapshots[-1] if snapshots else {}
    cpu = final.get("global_cpu_hybrid", {})
    # New logs count actual CPU miss work directly. Older logs can prove it
    # by bounding each all-GPU window at M=3 times top_k misses.
    # Exceeding that bound proves misses also executed on the CPU path.
    if cpu.get("cpu_miss_experts", 0) <= 0 and cpu.get(
        "misses", 0
    ) <= 3 * top_k * cpu.get("all_gpu_calls", 0):
        errors.append("no CPU miss-computation evidence beyond the all-GPU bound")
    # An empty retiring tail requires no moves. Retention still proves that
    # resizing preserved live experts; forced moves are checked by GPU VMM tests.
    if max(cpu.get("resize_moves", 0), cpu.get("resize_retained_experts", 0)) <= 0:
        errors.append("no retained experts across resize observed")
    probes = [
        dict(
            prompt=c["prompt"],
            matched=c.get("matched"),
            response=c.get("response_text"),
        )
        for c in report["cases"]
        if "prompt" in c
    ]
    if not probes or not all(case["matched"] for case in probes):
        errors.append("semantic probe after serving transitions did not pass")
    return dict(
        status="passed" if not errors else "failed",
        errors=list(dict.fromkeys(errors)),
        cpu_initialization=enabled,
        transitions=transitions,
        snapshots_checked=len(snapshots),
        cpu_stats=cpu,
        cases=[
            dict(context_tokens=c.get("context_tokens"), seconds=c["seconds"])
            for c in report["cases"]
            if "context_tokens" in c
        ],
        probes=probes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
