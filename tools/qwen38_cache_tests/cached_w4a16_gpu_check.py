#!/usr/bin/env python3
"""SM120 cache W4A16 reference/parity and W4A4 comparison; no model/server."""

import argparse
import json
import tempfile
from pathlib import Path

import torch


def decode(weight, scale, global_scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    value = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return (value * scale.float().repeat_interleave(16, -1) * global_scale).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--verify-mailbox-only", action="store_true")
    parser.add_argument("--cache-overlap-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required; cannot skip correctness/benchmark")
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist()
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.srt.model_loader.qwen38_nvfp4_pack.export import export_runtime_layer
    from sglang.srt.model_loader.qwen38_nvfp4_pack.parity import check_exported_layer
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    e, h, n, topk = 32, 2560, 640, 10
    raw = {}
    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
            a2a_backend=MoeA2ABackend.NONE,
        ),
        get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
        torch.device("cuda"),
    ):
        layer = FusedMoE(
            num_experts=e,
            hidden_size=h,
            intermediate_size=n,
            layer_id=0,
            top_k=topk,
            params_dtype=torch.bfloat16,
            quant_config=ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16
            ),
            gate_up_interleaved=False,
            inplace=False,
        )
        torch.manual_seed(791)
        for expert in range(e):
            for shard, rows, cols, target, global_scale, activation_scale in [
                ("w1", n, h, "w13", 0.03, 0.2),
                ("w3", n, h, "w13", 0.03, 0.2),
                ("w2", h, n, "w2", 0.04, 0.4),
            ]:
                packed = torch.randint(
                    0, 256, (rows, cols // 2), device="cuda", dtype=torch.uint8
                )
                scale = torch.full(
                    (rows, cols // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn
                )
                raw[expert, shard] = (packed, scale, global_scale)
                for suffix, tensor in [
                    ("weight", packed),
                    ("weight_scale", scale),
                    ("weight_scale_2", torch.tensor(global_scale, device="cuda")),
                    ("input_scale", torch.tensor(activation_scale, device="cuda")),
                ]:
                    name = target + "_" + suffix
                    layer.weight_loader(
                        getattr(layer, name),
                        tensor,
                        name,
                        shard_id=shard,
                        expert_id=expert,
                    )
        layer.quant_method.process_weights_after_loading(layer)

        def reference(x, ids, router):
            result = torch.zeros_like(x, dtype=torch.float32)
            for t in range(x.shape[0]):
                for k in range(topk):
                    expert = int(ids[t, k])
                    if expert < 0:
                        continue
                    gate = decode(*raw[expert, "w1"]).float() @ x[t].float()
                    up = decode(*raw[expert, "w3"]).float() @ x[t].float()
                    inter = (torch.nn.functional.silu(gate) * up).bfloat16()
                    down = decode(*raw[expert, "w2"]).float() @ inter.float()
                    result[t] += down * router[t, k]
            return result.bfloat16()

        reports = []
        with tempfile.TemporaryDirectory() as directory:
            exported = Path(directory) / "layer"
            export_runtime_layer(layer, exported)
            if args.cache_overlap_only:
                reports = []
                for graph in (False, True):
                    reports.append(
                        check_exported_layer(
                            layer,
                            exported,
                            "threaded",
                            slots=16,
                            use_w4a16=True,
                            prefill_w4a16=False,
                            reference=reference,
                            rtol=0.02,
                            atol=0.02,
                            decode_graph=graph,
                            graph_width=1,
                            prefill_direct_cache=True,
                            decode_early_cache=True,
                            benchmark_iterations=args.iterations,
                        )
                    )
                reports.append(
                    check_exported_layer(
                        layer,
                        exported,
                        "threaded",
                        slots=16,
                        use_w4a16=True,
                        reference=reference,
                        rtol=0.02,
                        atol=0.02,
                        prefill_direct_cache=True,
                        decode_early_cache=True,
                        shared_overlap=True,
                    )
                )
                result = {
                    "status": "passed",
                    "reports": reports,
                    "scope": "synthetic cache overlap parity; no full-model throughput claim",
                }
                Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2))
                return
            if args.verify_mailbox_only:
                report = check_exported_layer(
                    layer,
                    exported,
                    "threaded",
                    slots=32,
                    use_w4a16=True,
                    prefill_w4a16=False,
                    reference=reference,
                    rtol=0.02,
                    atol=0.02,
                    decode_graph=True,
                    graph_width=3,
                    benchmark_iterations=args.iterations,
                )
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return
            for mode, graph in [
                ("sync", False),
                ("threaded", False),
                ("threaded", True),
            ]:
                reports.append(
                    check_exported_layer(
                        layer,
                        exported,
                        mode,
                        slots=16,
                        use_w4a16=True,
                        reference=reference,
                        rtol=0.01,
                        atol=0.004,
                        decode_graph=graph,
                        graph_width=3 if graph else 1,
                        benchmark_iterations=args.iterations,
                    )
                )
            reports.append(
                check_exported_layer(
                    layer,
                    exported,
                    "threaded",
                    slots=16,
                    use_w4a16=True,
                    reference=reference,
                    shared_overlap=True,
                    rtol=0.01,
                    atol=0.004,
                )
            )
            for graph, graph_width in [(False, 1), (True, 1), (True, 3)]:
                reports.append(
                    check_exported_layer(
                        layer,
                        exported,
                        "threaded",
                        slots=16,
                        use_w4a16=True,
                        prefill_w4a16=False,
                        reference=reference,
                        rtol=0.02,
                        atol=0.02,
                        decode_graph=graph,
                        graph_width=graph_width,
                    )
                )
            # Old path remains executable and independently compares to resident
            # CUTLASS; do not demand equality between different activation modes.
            reports.append(
                check_exported_layer(
                    layer,
                    exported,
                    "threaded",
                    slots=16,
                    benchmark_iterations=args.iterations,
                )
            )
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe

        tensors = {
            name: getattr(layer, name)
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_blockscale_swizzled",
                "w2_blockscale_swizzled",
                "g1_alphas",
                "g2_alphas",
                "w13_input_scale_quant",
                "w2_input_scale_quant",
            )
        }
        direct_cases = []
        for tokens in (1, 4, 17, 65):
            x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
            ids = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
                    tokens, topk
                )
                % e
            )
            ids[:, 1] = ids[:, 0]  # duplicate expert
            ids[:, -1] = -1  # padded route
            weights = torch.full((tokens, topk), 1 / topk, device="cuda")
            expected = reference(x, ids, weights)
            actual = cached_w4a16_moe(x, ids, weights, tensors)
            torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.01)
            # Vary bound without changing numerical semantics.
            alternate = cached_w4a16_moe(x, ids, weights, tensors, chunk_tokens=3)
            torch.testing.assert_close(actual, alternate, atol=0, rtol=0)
            direct_cases.append(
                {
                    "tokens": tokens,
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                }
            )
        comparison = []
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        for tokens in (1, 4, 16, 64):
            x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
            ids = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
                    tokens, topk
                )
                % e
            )
            weights = torch.full((tokens, topk), 1 / topk, device="cuda")
            topk_output = StandardTopKOutput(
                weights, ids, torch.empty(tokens, e, device="cuda")
            )
            ref = reference(x, ids, weights)

            def a4():
                return layer.quant_method.apply(
                    layer, StandardDispatchOutput(x, None, topk_output)
                ).hidden_states

            def a16():
                return cached_w4a16_moe(x, ids, weights, tensors)

            measurements = {"tokens": tokens}
            for name, fn in (("w4a4", a4), ("w4a16", a16)):
                output = fn()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_allocated = torch.cuda.memory_allocated()
                begin.record()
                for _ in range(args.iterations):
                    output = fn()
                end.record()
                torch.cuda.synchronize()
                measurements[name] = {
                    "mean_ms": begin.elapsed_time(end) / args.iterations,
                    "peak_increment_bytes": torch.cuda.max_memory_allocated()
                    - base_allocated,
                    "rmse_to_a16_reference": float(
                        (output.float() - ref.float()).square().mean().sqrt()
                    ),
                }
            measurements["a16_over_a4_latency"] = (
                measurements["w4a16"]["mean_ms"] / measurements["w4a4"]["mean_ms"]
            )
            comparison.append(measurements)
        report = {
            "status": "passed",
            "gpu": torch.cuda.get_device_name(),
            "geometry": {"experts": e, "hidden": h, "width": n, "topk": topk},
            "reports": reports,
            "direct_cases": direct_cases,
            "same_weight_activation_comparison": comparison,
            "scope": "synthetic cache miss/hit/eviction/arena and breakable graph, no e2e",
            "inventory_note": "32 expert test inventory; timing is diagnostic, not the full 512 expert model.",
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
