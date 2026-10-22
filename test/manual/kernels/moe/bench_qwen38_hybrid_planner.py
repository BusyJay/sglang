# SPDX-License-Identifier: Apache-2.0
"""Paired CUDA-event latency of scan and indexed Qwen38 planner kernels.

Run on the serving GPU with PYTHONPATH=python python test/manual/kernels/moe/bench_qwen38_hybrid_planner.py.
The state reset happens before each timed event and is excluded from latency.
"""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.moe.qwen38_hybrid_plan import (
    COST_COLUMNS,
    MAX_ROUTES,
    hybrid_plan,
    hybrid_plan_indexed,
)
from sglang.srt.layers.moe.qwen38_heap_index import build_heap_index
from sglang.srt.layers.moe.qwen38_hybrid_policy import HybridCostModel


def state_for_case(layers, experts, slots, max_slots, free_slots):
    owners = [-1] * slots
    owners[0] = -2
    keys = [-1] * slots
    lookup = [[-1] * experts for _ in range(layers)]
    frequency = [[0.0] * experts for _ in range(layers)]
    next_expert = [0] * layers
    for slot in range(1, slots - free_slots):
        layer = (slot - 1) % layers
        expert = next_expert[layer]
        next_expert[layer] += 1
        owners[slot], keys[slot] = layer, expert
        lookup[layer][expert] = slot
        frequency[layer][expert] = 10.0 + expert / experts
    state = tuple(
        torch.tensor(value, dtype=dtype, device="cuda")
        for value, dtype in (
            (lookup, torch.int32),
            (frequency, torch.float64),
            (owners, torch.int32),
            (keys, torch.int32),
        )
    )
    index = build_heap_index(frequency, owners, keys, experts, max_slots).to("cuda")
    return state, index, next_expert[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--slots", type=int, default=6388)
    parser.add_argument("--max-slots", type=int, default=7006)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Include a real cost table in plan timing; routes remain synthetic",
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Compile a saved pre-change planner for paired indexed comparison",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    if not (1 < args.slots <= args.max_slots <= args.layers * args.experts):
        parser.error("invalid slot capacity")
    if not (
        1 <= args.tokens <= 3 and 1 <= args.top_k and args.tokens * args.top_k <= 32
    ):
        parser.error("invalid route shape")
    if args.samples < 1 or args.warmup < 0:
        parser.error("invalid sample count")
    cost_rows = []
    if args.calibration is not None:
        table = json.loads(args.calibration.read_text())
        metadata = {
            name: table[name]
            for name in ("backend", "hidden", "width", "top_k", "cores", "gpu", "cpu")
        }
        metadata.update(
            {
                name: table[name]
                for name in ("transport", "runtime_version")
                if name in table
            }
        )
        cost_rows = HybridCostModel(table, **metadata).device_cost_rows(MAX_ROUTES)
    costs = torch.tensor(cost_rows, dtype=torch.float64, device="cuda").reshape(
        -1, COST_COLUMNS
    )
    print("profitable cost rows", len(cost_rows), flush=True)
    baseline_indexed = None
    if args.baseline_source is not None:
        from sglang.kernels.jit.utils import load_jit

        path = args.baseline_source.resolve()
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        baseline = load_jit(
            "qwen38_hit_baseline",
            fingerprint,
            cuda_files=[str(path)],
            extra_include_paths=[
                str(
                    Path(__file__).resolve().parents[4]
                    / "python/sglang/kernels/jit/csrc/moe"
                )
            ],
            cuda_wrappers=[("plan_indexed", "qwen38_hybrid_plan_indexed")],
        )
        baseline_indexed = baseline.plan_indexed
    rng = random.Random(617)
    for case, free_slots in (("all_hit", 0), ("full_miss", 0), ("free_miss", 64)):
        source, source_index, first_unused = state_for_case(
            args.layers, args.experts, args.slots, args.max_slots, free_slots
        )
        if first_unused + args.top_k + args.tokens >= args.experts:
            parser.error("too few unused experts for miss case")
        base = 0 if case == "all_hit" else first_unused
        ids = torch.tensor(
            [[base + r + t for r in range(args.top_k)] for t in range(args.tokens)],
            dtype=torch.int32,
            device="cuda",
        )
        state = tuple(torch.empty_like(t) for t in source)
        index = torch.empty_like(source_index)
        gpu, cpu = torch.empty_like(ids), torch.empty_like(ids)
        plan = torch.empty(4 + 2 * ids.numel(), dtype=torch.int32, device="cuda")
        stats = torch.empty(7, dtype=torch.int64, device="cuda")
        functions = {
            "scan": hybrid_plan,
            "indexed": hybrid_plan_indexed,
        }
        if baseline_indexed is not None:
            functions["before_indexed"] = baseline_indexed
        timings = {name: [] for name in functions}
        outcomes = {}
        for sample in range(args.samples + args.warmup):
            order = list(functions)
            rng.shuffle(order)
            for mode in order:
                for target, original in zip(state, source):
                    target.copy_(original)
                index.copy_(source_index)
                stats.zero_()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                call = functions[mode]
                arguments = (ids, *state, costs, gpu, cpu, plan, stats, 0, 0.95, 1.9)
                if mode in ("indexed", "before_indexed"):
                    arguments += (index,)
                call(*arguments)
                end.record()
                end.synchronize()
                if sample == 0:
                    owner_values, key_values = (
                        tensor.tolist() for tensor in state[2:4]
                    )
                    outcomes[mode] = (
                        tuple(
                            sorted(
                                (owner, key)
                                for owner, key in zip(owner_values, key_values)
                                if owner >= 0 and key >= 0
                            )
                        ),
                        tuple(state[1].reshape(-1).tolist()),
                        tuple(stats.tolist()),
                        tuple(cpu.reshape(-1).tolist()),
                        tuple(sorted(plan[4 : 4 + 2 * plan[1].item() : 2].tolist())),
                    )
                if sample >= args.warmup:
                    timings[mode].append(begin.elapsed_time(end) * 1000)
        if not all(value == outcomes["scan"] for value in outcomes.values()):
            raise AssertionError(f"planner decisions differ for {case}")
        print(
            case,
            {
                name + suffix: round(value, 3)
                for name, values in timings.items()
                for suffix, value in (
                    ("_p50_us", statistics.median(values)),
                    (
                        "_p95_us",
                        sorted(values)[
                            min(len(values) - 1, (95 * len(values) + 99) // 100 - 1)
                        ],
                    ),
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
