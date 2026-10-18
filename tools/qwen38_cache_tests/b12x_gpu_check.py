#!/usr/bin/env python3
"""SM12x B12X native cache parity, migration and A/B microbench (no model)."""

import argparse
import json
from pathlib import Path
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe
from tools.qwen38_cache_tests.w4a16_verify_gpu_check import (
    dequant,
    latency,
    reference,
    swizzle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 5:
        parser.error("iterations must be >= 5")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("SM12x GPU required")
    torch.manual_seed(20261018)
    torch.backends.cuda.matmul.allow_tf32 = False
    e, h, n, topk = 32, 2560, 640, 10
    before = torch.cuda.memory_allocated()
    backend = CachedB12xW4A16(torch.device("cuda"), h, n, e)
    scratch_bytes = torch.cuda.memory_allocated() - before
    tensors, dense, raw_scales = {}, {}, {}
    for prefix, rows, cols in [("w13", 2 * n, h), ("w2", h, n)]:
        w = torch.randint(
            0, 256, (e, rows, cols // 2), dtype=torch.uint8, device="cuda"
        )
        scales = (torch.randint(1, 6, (e, rows, cols // 16)).float() * 0.125).to(
            torch.float8_e4m3fn
        )
        global_scale = torch.linspace(0.015, 0.035, e, device="cuda")
        input_scale = 0.7 if prefix == "w13" else 1.3
        tensors[prefix + "_weight"] = w
        tensors[prefix + "_blockscale_swizzled"] = swizzle(scales)
        tensors[prefix + "_input_scale_quant"] = torch.tensor(
            1 / input_scale, device="cuda"
        )
        tensors["g1_alphas" if prefix == "w13" else "g2_alphas"] = (
            global_scale * input_scale
        )
        dense[prefix] = dequant(w, scales.cuda(), global_scale)
        raw_scales[prefix] = scales.cuda()
    reports = []
    for tokens in (1, 2, 3):
        for pattern in ("disjoint", "overlap", "invalid", "all_invalid"):
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
            if pattern == "all_invalid":
                ids.fill_(-1)
            for dtype in (torch.int32, torch.int64):
                ids = ids.to(dtype)
                x = torch.randn(tokens, h, dtype=torch.bfloat16, device="cuda") * 0.2
                router = torch.randn(tokens, topk, device="cuda").softmax(-1)

                def run():
                    return backend.run(x, ids, router, tensors)

                expected = reference(x, ids, router, dense["w13"], dense["w2"])
                got = run()
                torch.testing.assert_close(got, expected, atol=0.004, rtol=0.02)
                error = float((got.float() - expected.float()).abs().max())
                # Changing IDs, activations, router and alpha must affect replay.
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                x.mul_(0.9)
                ids.copy_(ids.roll(1, 1))
                router.copy_(router.roll(2, 1))
                tensors["g2_alphas"].mul_(0.9)
                dense["w2"] = dequant(
                    tensors["w2_weight"],
                    raw_scales["w2"],
                    tensors["g2_alphas"] * tensors["w2_input_scale_quant"],
                )
                graph.replay()
                torch.testing.assert_close(captured, run(), atol=0, rtol=0)
                torch.testing.assert_close(
                    captured,
                    reference(x, ids, router, dense["w13"], dense["w2"]),
                    atol=0.004,
                    rtol=0.02,
                )
                b12x_graph_latency = latency(graph.replay, args.iterations)
                del graph, captured
                # Real boundary runs eagerly; alternate streams with shared scratch.
                other = torch.cuda.Stream()
                other.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(other):
                    alternate = run()
                again = run()
                torch.cuda.synchronize()
                torch.testing.assert_close(alternate, again, atol=0, rtol=0)
                reports.append(
                    dict(
                        tokens=tokens,
                        pattern=pattern,
                        ids_dtype=str(dtype),
                        max_abs_error=error,
                        graph_replay=True,
                        b12x_graph_replay=b12x_graph_latency,
                        alternate_stream=True,
                        b12x=latency(run, args.iterations),
                        triton_fused=latency(
                            lambda: cached_w4a16_moe(
                                x, ids, router, tensors, split_k=1, verify_mode="fused"
                            ),
                            args.iterations,
                        ),
                    )
                )
                print(f"passed tokens={tokens} pattern={pattern} ids={dtype}", flush=True)
    # Capacity changes and replacement pool storage must not retain old pointers.
    for capacity in (16, 32, 16):
        replacement = {
            k: (v[:capacity].clone() if v.ndim else v.clone())
            for k, v in tensors.items()
        }
        x = torch.randn(1, h, dtype=torch.bfloat16, device="cuda") * 0.2
        ids = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk)
        router = torch.full((1, topk), 0.1, device="cuda")
        torch.testing.assert_close(
            backend.run(x, ids, router, replacement),
            reference(x, ids, router, dense["w13"][:capacity], dense["w2"][:capacity]),
            atol=0.004,
            rtol=0.02,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                status="passed",
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                scratch_bytes_per_layer=scratch_bytes,
                migration=True,
                reports=reports,
                scope="Synthetic resident cache; excludes planner/fill/model/service",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
