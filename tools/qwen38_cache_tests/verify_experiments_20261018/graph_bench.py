#!/usr/bin/env python3
"""Measure kernels in CUDA graphs to exclude Python enqueue gaps; GPU required."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w4a16_verify_gpu_check import swizzle
from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe


def timing(graph, operations, samples=30):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / operations)
    return dict(
        median_us=statistics.median(times) * 1000,
        p95_us=sorted(times)[int(0.95 * (samples - 1))] * 1000,
    )


def planner():
    reports = []
    module = _module()
    for slots in (80, 144, 192, 256):
        for tokens in (2, 3):
            for churn in (0, 3, 30):
                for mode in (1, 2):
                    state = [
                        torch.zeros(513, dtype=torch.int32, device="cuda"),
                        torch.full((512,), -1, dtype=torch.int32, device="cuda"),
                        torch.full((slots,), -1, dtype=torch.int32, device="cuda"),
                        torch.zeros(slots, device="cuda"),
                        torch.zeros(slots, dtype=torch.int64, device="cuda"),
                        torch.zeros(slots, dtype=torch.int64, device="cuda"),
                        torch.zeros(1, dtype=torch.int64, device="cuda"),
                        torch.zeros(4 + 4 * slots, dtype=torch.int32, device="cuda"),
                    ]
                    ids = torch.empty(
                        (1, tokens * 10), dtype=torch.int32, device="cuda"
                    )
                    output = torch.empty_like(ids)
                    sequence = [
                        (
                            torch.arange(tokens * 10, device="cuda", dtype=torch.int32)
                            + i * churn
                        )
                        .remainder(512)
                        .view_as(ids)
                        for i in range(128)
                    ]

                    def run():
                        for routes in sequence:
                            ids.copy_(routes)
                            module.decode_plan_parallel(
                                ids, output, *state, 0.95, 0, 0.7, 0, 0, mode
                            )

                    run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    reports.append(
                        dict(
                            slots=slots,
                            tokens=tokens,
                            churn=churn,
                            planner_mode=mode,
                            **timing(graph, len(sequence)),
                        )
                    )
                    del graph
    return reports


def projection():
    torch.manual_seed(20261018)
    experts, hidden, width, topk = 32, 2560, 640, 10
    tensors = {}
    for prefix, rows, cols in (("w13", 2 * width, hidden), ("w2", hidden, width)):
        tensors[prefix + "_weight"] = torch.randint(
            0, 256, (experts, rows, cols // 2), dtype=torch.uint8, device="cuda"
        )
        scales = (
            torch.randint(1, 6, (experts, rows, cols // 16))
            .float()
            .mul(0.125)
            .to(torch.float8_e4m3fn)
        )
        tensors[prefix + "_blockscale_swizzled"] = swizzle(scales)
        tensors[prefix + "_input_scale_quant"] = torch.ones(1, device="cuda")
        tensors["g1_alphas" if prefix == "w13" else "g2_alphas"] = torch.full(
            (experts,), 0.025, device="cuda"
        )
    reports = []
    for tokens in (2, 3):
        for pattern in ("disjoint", "overlap"):
            ids = torch.arange(tokens * topk, dtype=torch.int32, device="cuda").reshape(
                tokens, topk
            )
            if pattern == "overlap":
                ids.remainder_(topk)
                ids[:, 1] = ids[:, 0]
            x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.2
            router = torch.randn(tokens, topk, device="cuda").softmax(-1)
            expected = cached_w4a16_moe(
                x, ids, router, tensors, split_k=1, verify_mode="baseline"
            )
            for mode in ("baseline", "fused", "reuse"):

                def run():
                    return cached_w4a16_moe(
                        x, ids, router, tensors, split_k=1, verify_mode=mode
                    )

                torch.testing.assert_close(run(), expected, atol=0.004, rtol=0.02)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        output = run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, atol=0.004, rtol=0.02)
                reports.append(
                    dict(tokens=tokens, pattern=pattern, mode=mode, **timing(graph, 32))
                )
                del output, graph
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required")
    report = dict(
        gpu=torch.cuda.get_device_name(),
        planner=planner(),
        projection=projection(),
        scope="Graph replay; planner includes route D2D copy, projection uses resident synthetic weights. No host fills or service overhead.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
