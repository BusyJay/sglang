#!/usr/bin/env python3
"""Independent-reference parity and A/B latency for cached W4A16 small verify."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe


def swizzle(scale):
    # Independent CPU layout encoder, including padded rows/scale columns.
    e, rows, cols = scale.shape
    pr, pc = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    r = torch.arange(rows)[:, None]
    c = torch.arange(cols)[None, :]
    offsets = (
        ((r // 128 * (pc // 4) + c // 4) * 32 + r % 32) * 4 + r % 128 // 32
    ) * 4 + c % 4
    packed = torch.zeros(e, pr * pc, dtype=torch.uint8)
    packed[:, offsets.flatten()] = scale.view(torch.uint8).reshape(e, -1)
    return packed.reshape(e, pr, pc).cuda().view(torch.float8_e4m3fn)


def dequant(weight, scale, global_scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return (
        values * scale.float().repeat_interleave(16, -1) * global_scale[:, None, None]
    ).bfloat16()


def reference(x, ids, router, w13, w2):
    width = w13.shape[1] // 2
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if not 0 <= expert < w13.shape[0]:
                continue
            up = w13[expert, :width].float() @ x[token].float()
            gate = w13[expert, width:].float() @ x[token].float()
            inter = (torch.nn.functional.silu(gate) * up).bfloat16()
            output[token] += (w2[expert].float() @ inter.float()) * router[token, slot]
    return output.bfloat16()


def latency(fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "peak_scratch_bytes": torch.cuda.max_memory_allocated() - before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be >= 1")
    if args.iterations < 5:
        parser.error("iterations must be >= 5")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required; cannot skip parity or claim speedup")
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, h, n, topk = 32, 2560, 640, 10
    tensors, dense = {}, {}
    for prefix, rows, cols in [("w13", 2 * n, h), ("w2", h, n)]:
        w = torch.randint(
            0, 256, (e, rows, cols // 2), dtype=torch.uint8, device="cuda"
        )
        scales = (
            torch.randint(1, 6, (e, rows, cols // 16))
            .float()
            .mul(0.125)
            .to(torch.float8_e4m3fn)
        )
        global_scale = torch.linspace(0.015, 0.035, e, device="cuda")
        input_scale = torch.tensor(0.7 if prefix == "w13" else 1.3, device="cuda")
        tensors[prefix + "_weight"] = w
        tensors[prefix + "_blockscale_swizzled"] = swizzle(scales)
        tensors[prefix + "_input_scale_quant"] = 1 / input_scale
        tensors["g1_alphas" if prefix == "w13" else "g2_alphas"] = (
            global_scale * input_scale
        )
        dense[prefix] = dequant(w, scales.cuda(), global_scale)
    reports = []
    for tokens in (1, 2, 3, 4):
        for pattern in ("disjoint", "overlap", "invalid"):
            ids = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
                    tokens, topk
                )
                % e
            )
            if pattern == "overlap":
                ids.remainder_(topk)
                ids[:, 1] = ids[:, 0]
            if pattern == "invalid":
                ids[:, 0], ids[:, 1] = -1, e
            x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.2
            router = torch.randn(tokens, topk, device="cuda").softmax(-1)
            expected = reference(x, ids, router, dense["w13"], dense["w2"])
            original_ids, original_x = ids.clone(), x.clone()
            for split in (0, 1, 2, 4, 8):
                ids.copy_(original_ids)
                x.copy_(original_x)

                def run(x=x, ids=ids, router=router, split=split):
                    return cached_w4a16_moe(x, ids, router, tensors, split_k=split)

                got = run()
                torch.testing.assert_close(got, expected, atol=0.004, rtol=0.02)
                error = float((got.float() - expected.float()).abs().max())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                # Changed routes and activations must be read on replay, not at capture.
                ids.copy_(ids.roll(1, 1))
                x.mul_(0.9)
                graph.replay()
                torch.testing.assert_close(captured, run(), atol=0, rtol=0)
                torch.testing.assert_close(
                    captured,
                    reference(x, ids, router, dense["w13"], dense["w2"]),
                    atol=0.004,
                    rtol=0.02,
                )
                ids.copy_(original_ids)
                x.copy_(original_x)
                del captured, graph
                measurements = [
                    latency(run, args.iterations) for _ in range(args.rounds)
                ]
                record = {
                    "tokens": tokens,
                    "pattern": pattern,
                    "split_k": split,
                    "max_abs_error": error,
                    "graph_replay": True,
                    "latency": {
                        "median_ms": statistics.median(
                            m["median_ms"] for m in measurements
                        ),
                        "p95_ms": max(m["p95_ms"] for m in measurements),
                        "peak_scratch_bytes": max(
                            m["peak_scratch_bytes"] for m in measurements
                        ),
                    },
                    "rounds": measurements,
                }
                reports.append(record)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "iterations": args.iterations,
                "round_count": args.rounds,
                "reports": reports,
                "scope": "Resident synthetic cache, excludes planner, fills and service",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
