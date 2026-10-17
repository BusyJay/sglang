#!/usr/bin/env python3
"""SM120 MTP A4 smoke, graph replay and layer benchmark; no model downloads."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experts", type=int, default=32, choices=(32, 512))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            "This check requires an SM120 GPU; GPU checks cannot be skipped"
        )
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist()
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.qwen38_fp4_mtp import (
        Qwen38Fp4MtpMethod,
        fp4_mtp_moe,
    )

    torch.manual_seed(103)
    e, h, n = args.experts, 2560, 640
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, n // 2), device="cuda", dtype=torch.uint8)
    s13 = torch.full((e, 2 * n, h // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
    s2 = torch.full((e, h, n // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
    g13 = torch.tensor([[0.03, 0.05]] * e, device="cuda")
    g2 = torch.full((e,), 0.03, device="cuda")
    # Independent checkpoint-order tensors are retained only in this test.
    layer = SimpleNamespace(
        num_local_experts=e,
        w13_weight=w13.clone(),
        w2_weight=w2,
        w13_weight_scale=s13.clone(),
        w2_weight_scale=s2.clone(),
        w13_weight_scale_2=g13,
        w2_weight_scale_2=g2,
    )
    method = Qwen38Fp4MtpMethod.__new__(Qwen38Fp4MtpMethod)
    method.activation_mode = "w4a4"
    method.create_moe_runner(
        layer,
        MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size_per_partition=n,
            top_k=10,
            params_dtype=torch.bfloat16,
        ),
    )
    scale_pointers = (
        layer.w13_weight_scale.data_ptr(),
        layer.w2_weight_scale.data_ptr(),
    )
    torch.cuda.synchronize()
    before_prepare = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    prepare_resident_bytes = torch.cuda.memory_allocated() - before_prepare
    prepare_peak_bytes = torch.cuda.max_memory_allocated() - before_prepare
    assert layer.w13_blockscale_swizzled.data_ptr() == scale_pointers[0]
    assert layer.w2_blockscale_swizzled.data_ptr() == scale_pointers[1]
    assert prepare_resident_bytes < 1 << 20, prepare_resident_bytes
    assert prepare_peak_bytes < 8 << 20, prepare_peak_bytes
    reports = []
    for tokens in (1, 3, 128, 257):
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.randint(0, e, (tokens, 10), device="cuda", dtype=torch.int32)
        ids[:, 1] = ids[:, 0]  # duplicated expert routes
        ids[:, -1] = -1
        weights = torch.full((tokens, 10), 0.1, device="cuda")
        weights[:, -1] = 0
        topk = StandardTopKOutput(weights, ids, torch.zeros(tokens, e, device="cuda"))
        dispatch = StandardDispatchOutput(x, None, topk)

        def a4():
            return method.apply(layer, dispatch).hidden_states

        def a16():
            return fp4_mtp_moe(x, ids, weights, w13, w2, s13, s2, g13, g2)

        result, reference = a4(), a16()
        torch.cuda.synchronize()
        if not torch.isfinite(result).all():
            raise AssertionError("A4 produced non-finite outputs")
        delta = result.float() - reference.float()
        relative_rmse = float(
            delta.square().mean().sqrt() / reference.float().square().mean().sqrt()
        )
        # Report A4 quantization error; A16 parity is deliberately not required.
        row = {
            "tokens": tokens,
            "relative_rmse_vs_a16": relative_rmse,
            "max_abs_error_vs_a16": float(delta.abs().max()),
        }
        for name, fn in (("w4a16", a16), ("w4a4", a4)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(20):
                fn()
            end.record()
            end.synchronize()
            row[name + "_ms"] = start.elapsed_time(end) / 20
        if tokens <= 3:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    a4()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = a4()
            x.mul_(0.75)
            graph.replay()
            eager = a4()
            # CUTLASS's fused finalize can use non-associative atomics.
            torch.testing.assert_close(
                captured.float(), eager.float(), atol=0.01, rtol=0.02
            )
            row["graph_replay"] = "passed"
        reports.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "experts": e,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "prepare_resident_bytes": prepare_resident_bytes,
                "prepare_peak_bytes": prepare_peak_bytes,
                "macro_benchmark": "not_run",
                "cases": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
