#!/usr/bin/env python3
"""Native MXFP8, FP32-weight norm and GDN parity/latency checks; no server."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def timing(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required; correctness checks cannot skip")
    torch.manual_seed(135)
    reports = {}
    from sglang.srt.layers.quantization.qwen38_fp32_norm import fp32_grouped_gemma_norm

    norm = []
    for tokens in (1, 4, 64):
        for hidden, group in ((10240, 2560), (7680, 2560), (160, 160)):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.1

            def reference():
                value = x.float().reshape(tokens, hidden // group, group)
                out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                return (out.reshape_as(x) * (1 + w)).bfloat16()

            fn = lambda: fp32_grouped_gemma_norm(x, w, group, 1e-6)
            torch.testing.assert_close(fn(), reference(), atol=0.016, rtol=0.008)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fn()
            x.mul_(0.8)
            graph.replay()
            torch.testing.assert_close(captured, fn(), atol=0, rtol=0)
            norm.append(
                {
                    "tokens": tokens,
                    "hidden": hidden,
                    "group": group,
                    "fused_ms": timing(fn),
                    "torch_ms": timing(reference),
                }
            )
    reports["fp32_weight_norm"] = norm
    # Exercise SGLang's native SM120 MXFP8 cutlass path and its loader scale layout.
    from flashinfer import block_scale_interleave, mxfp8_quantize

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    mx = []
    for tokens in (1, 4, 64):
        for outputs, inputs in ((640, 2560), (2560, 640)):
            source = (
                torch.randn(outputs, inputs, device="cuda", dtype=torch.bfloat16) * 0.1
            )
            weight, ws = mxfp8_quantize(source, is_sf_swizzled_layout=False)
            # FlashInfer 0.6.18 returns the linear scale plane flattened.
            ws = ws.reshape(outputs, inputs // 32)
            wscale = block_scale_interleave(ws.contiguous())
            x = torch.randn(tokens, inputs, device="cuda", dtype=torch.bfloat16) * 0.2
            xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=False)

            def dequant(q, scale):
                scale = scale.reshape(-1, q.shape[-1] // 32)[: q.shape[0]]
                return q.float() * torch.exp2(scale.float() - 127).repeat_interleave(
                    32, -1
                )

            expected = dequant(xq, xs) @ dequant(weight, ws).t()
            fn = lambda: flashinfer_mxfp8_blockscaled_linear(
                x, weight, wscale, backend="cutlass"
            )
            got = fn()
            torch.testing.assert_close(got.float(), expected, atol=0.03, rtol=0.03)
            mx.append(
                {
                    "tokens": tokens,
                    "N": outputs,
                    "K": inputs,
                    "ms": timing(fn),
                    "max_abs_error": float((got.float() - expected).abs().max()),
                }
            )
    reports["native_mxfp8_cutlass"] = mx
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    gdn = []
    for state_dtype in (torch.float32, torch.bfloat16):
        batch, heads, dim = 2, 2, 128
        q, k = [
            torch.randn(batch, 1, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(2)
        ]
        v = torch.randn_like(q)
        a, b = [
            torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        A = torch.zeros(heads, device="cuda")
        bias = torch.zeros(heads, device="cuda")
        original = (
            torch.randn(batch, heads, dim, dim, device="cuda", dtype=state_dtype) * 0.01
        )
        state = original.clone()
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)

        def run():
            state.copy_(original)
            return fused_sigmoid_gating_delta_rule_update(
                A, a, bias, 1.0, 20.0, q, k, v, b, state, slots, scale=dim**-0.5
            )

        expected_state = original.float().transpose(-1, -2).clone()
        expected_state *= torch.exp(-torch.exp(A)[None, :] * F.softplus(a.float()))[
            ..., None, None
        ]
        delta = (
            v[:, 0].float() - (expected_state * k[:, 0].float()[..., :, None]).sum(-2)
        ) * torch.sigmoid(b.float())[..., None]
        expected_state += k[:, 0].float()[..., :, None] * delta[..., None, :]
        expected = (
            (expected_state * (q[:, 0].float() * dim**-0.5)[..., :, None])
            .sum(-2)
            .unsqueeze(1)
        )
        got = run()
        torch.testing.assert_close(got.float(), expected, atol=0.008, rtol=0.02)
        torch.testing.assert_close(
            state.float(), expected_state.transpose(-1, -2), atol=0.004, rtol=0.02
        )
        gdn.append(
            {"state_dtype": str(state_dtype), "ms_with_state_reset": timing(run)}
        )
    reports["gdn_recurrent"] = gdn
    Path(args.output).write_text(
        json.dumps(
            {
                "status": "passed",
                "gpu": torch.cuda.get_device_name(),
                "checks": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
