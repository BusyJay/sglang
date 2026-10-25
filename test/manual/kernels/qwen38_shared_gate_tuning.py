"""Compare C2 consumer launch layouts while preserving the gate reduction."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_gate_sigmoid_mul_add_kernel,
    fused_gate_sigmoid_mul_add,
)
from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate


@triton.jit
def tiled_add(G, S, R, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.extra.cuda.gdc_wait()
    gate = tl.load(G + row)
    shared = tl.load(S + row * WIDTH + offsets, offsets < WIDTH, 0).to(tl.float32)
    routed = tl.load(R + row * WIDTH + offsets, offsets < WIDTH, 0).to(tl.float32)
    tl.extra.cuda.gdc_launch_dependents()
    tl.store(R + row * WIDTH + offsets, gate * shared + routed, offsets < WIDTH)


def measure(fn):
    for _ in range(5):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(200):
            fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    values = []
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 200)
    return statistics.median(values)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=("sweep", "paired"), default="paired")
args = parser.parse_args()
torch.set_grad_enabled(False)
torch.manual_seed(42)
if args.mode == "paired":
    from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate_add

    paired = []
    for rows in (1, 2, 3):
        hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.02
        shared = torch.randn_like(hidden)
        routed = torch.randn_like(hidden)
        out = torch.empty_like(routed)
        gate = shared_expert_gate(hidden, weight)

        def old():
            out.copy_(routed)
            _fused_gate_sigmoid_mul_add_kernel[(rows,)](
                None,
                gate,
                shared,
                out,
                hidden_dim=2560,
                BLOCK_SIZE=4096,
                DO_ADD=True,
                USE_PDL=True,
                PRECOMPUTED_GATE=True,
                num_warps=16,
                launch_pdl=True,
            )

        def new():
            out.copy_(routed)
            shared_expert_gate_add(gate, shared, out)

        for seed in range(20):
            hidden.normal_()
            shared.normal_()
            routed.normal_()
            shared_expert_gate(hidden, weight, out=gate)
            expected = routed.clone()
            fused_gate_sigmoid_mul_add(hidden, weight, shared, expected)
            old()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            new()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        for phase in ("consumer", "serial"):

            def total(fn):
                if phase == "serial":
                    shared_expert_gate(hidden, weight, out=gate)
                fn()

            observations = []
            for cycle in range(4):
                for name, fn in [
                    ("old", old),
                    ("new", new),
                    ("new", new),
                    ("old", old),
                ]:
                    observations.append(dict(arm=name, us=measure(lambda: total(fn))))
            medians = {
                name: statistics.median(
                    [v["us"] for v in observations if v["arm"] == name]
                )
                for name in ("old", "new")
            }
            result = dict(
                rows=rows,
                phase=phase,
                includes_reset_copy=True,
                medians_us=medians,
                improvement_percent=(medians["old"] / medians["new"] - 1) * 100,
                observations=observations,
                bitwise_cases=20,
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "observations"}),
                flush=True,
            )
            paired.append(result)
    args.output.write_text(json.dumps(paired, indent=2) + "\n")
    raise SystemExit(0)

results = []
for rows in (1, 2, 3):
    hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.02
    shared = torch.randn_like(hidden)
    routed = torch.randn_like(hidden)
    out = torch.empty_like(routed)
    gate = shared_expert_gate(hidden, weight)
    baseline = routed.clone()
    fused_gate_sigmoid_mul_add(hidden, weight, shared, baseline)
    for block, warps in [
        (4096, 16),
        (4096, 8),
        (4096, 4),
        (4096, 2),
        (4096, 1),
        (128, 1),
        (256, 1),
        (512, 2),
        (128, 4),
        (256, 4),
        (512, 4),
        (1024, 4),
        (128, 8),
        (256, 8),
    ]:

        def consumer():
            if block == 4096:
                _fused_gate_sigmoid_mul_add_kernel[(rows,)](
                    None,
                    gate,
                    shared,
                    out,
                    hidden_dim=2560,
                    BLOCK_SIZE=block,
                    DO_ADD=True,
                    USE_PDL=True,
                    PRECOMPUTED_GATE=True,
                    num_warps=warps,
                    launch_pdl=True,
                )
            else:
                tiled_add[(rows, triton.cdiv(2560, block))](
                    gate,
                    shared,
                    out,
                    WIDTH=2560,
                    BLOCK=block,
                    num_warps=warps,
                    launch_pdl=True,
                )

        out.copy_(routed)
        consumer()
        torch.testing.assert_close(out, baseline, rtol=0, atol=0)

        def reset_consume():
            out.copy_(routed)
            consumer()

        def serial():
            shared_expert_gate(hidden, weight, out=gate)
            reset_consume()

        times = [measure(reset_consume), measure(serial), measure(reset_consume)]
        result = dict(
            rows=rows,
            block=block,
            warps=warps,
            consumer_with_reset_us=statistics.median([times[0], times[2]]),
            serial_with_reset_us=times[1],
            bitwise_equal=True,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
args.output.write_text(json.dumps(results, indent=2) + "\n")
