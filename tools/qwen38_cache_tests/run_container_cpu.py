#!/usr/bin/env python3
"""Run small CPU fixtures, then check imports against image dependencies."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def main():
    result = subprocess.run([sys.executable, "tools/qwen38_cache_tests/run_cpu.py"])
    if result.returncode:
        return result.returncode
    modules = [
        "sglang.srt.arg_groups.qwen38_cache_hook",
        "sglang.srt.layers.moe.qwen38_cache_runtime",
        "sglang.srt.layers.moe.qwen38_cache_async",
        "sglang.srt.layers.moe.qwen38_cache_device",
        "sglang.srt.layers.moe.qwen38_cache_threaded",
        "sglang.srt.model_loader.qwen38_nvfp4_pack.convert",
        "sglang.srt.model_loader.qwen38_nvfp4_pack.parity",
        "sglang.srt.layers.quantization.qwen38_nvfp4_cache",
        "sglang.srt.model_loader.qwen38_nvfp4_cache_loader",
        "sglang.srt.model_loader.qwen38_mtp_loader",
        "sglang.srt.models.qwen4_exp",
        "sglang.srt.models.qwen4_exp_mtp",
    ]
    for module in modules:
        importlib.import_module(module)
        print(f"Imported {module}", flush=True)

    # AST fixtures bypass the record-to-runtime projection. Check real config
    # bags so launch defaults cannot disappear between declaration and forward.
    from sglang.srt.arg_groups.qwen38_cache_hook import handle_qwen38_nvfp4_cache
    from sglang.srt.runtime_context import ParallelContext, RuntimeContext
    from sglang.srt.server_args import ServerArgs

    for steps in (1, 2, 3):
        args = ServerArgs(
            model_path="dummy",
            load_format="qwen38_nvfp4_cache",
            language_model_only=True,
            moe_a2a_backend="none",
            speculative_algorithm="NEXTN",
            speculative_draft_load_format="qwen38_mtp",
            speculative_num_steps=steps,
            model_loader_extra_config={
                "runtime_mode": "threaded",
                "cache_slots_total": 5040,
                "adaptive_cache": True,
            },
        )
        handle_qwen38_nvfp4_cache(args)
        context = RuntimeContext(ParallelContext())
        context.set_server_args(args)
        assert context.config_bag("spec").speculative_num_draft_tokens == steps + 1
        assert context.config_bag("exec").mamba.mamba_ssm_dtype == "bfloat16"
        assert context.config_bag("spec").speculative_draft_kv_cache_dtype == "bfloat16"
    print("Real MTP ServerArgs projection passed for steps 1/2/3.", flush=True)
    import torch

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.qwen38_cache_runtime import PreparedExperts
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptMixedPrecisionConfig,
    )
    from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
        Qwen38CachedModelOptConfig,
        Qwen38CachedNvFp4MoeMethod,
    )

    config = json.loads(Path("flash-next-plan/model-metadata/config.json").read_text())
    delegate = ModelOptMixedPrecisionConfig.from_config(config["quantization_config"])
    wrapped = Qwen38CachedModelOptConfig(delegate, None)
    for layer_id in range(48):
        assert wrapped.resolve_quant_algo(f"model.layers.{layer_id}.mlp") == "NVFP4"
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    probe = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(probe)
    assert isinstance(
        wrapped.get_quant_method(probe, "model.layers.0.mlp.experts"),
        Qwen38CachedNvFp4MoeMethod,
    )
    manifest = SimpleNamespace(
        num_experts=512, num_layers=48, hidden_size=2560, intermediate_size=640
    )
    method = Qwen38CachedNvFp4MoeMethod(manifest, "model.layers.0.mlp")
    layer = torch.nn.Module()
    layer.layer_id = 0
    method.create_weights(layer, 512, 2560, 640, torch.bfloat16)
    assert not list(layer.parameters()), (
        "cached MoE allocated resident expert parameters"
    )
    method.create_moe_runner(
        layer,
        MoeRunnerConfig(
            num_experts=512,
            num_local_experts=512,
            hidden_size=2560,
            top_k=10,
        ),
    )
    assert method.fuse_routed_scaling_factor_in_topk
    ids = torch.tensor([[7, 20]], dtype=torch.int32)
    remapped = torch.tensor([[1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.3, 0.7]])
    hidden = torch.zeros((1, 16), dtype=torch.bfloat16)
    topk = StandardTopKOutput(weights, ids, torch.zeros((1, 512)))
    dispatch = StandardDispatchOutput(hidden, None, topk)
    tensors = {
        name: torch.empty(2, 8, dtype=torch.uint8)
        for name in [
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
        ]
    }
    tensors.update({name: torch.ones(2) for name in ["g1_alphas", "g2_alphas"]})
    tensors.update(
        {
            name: torch.ones(1)
            for name in ["w13_input_scale_quant", "w2_input_scale_quant"]
        }
    )
    prepared = PreparedExperts(tensors, remapped, 56, False)
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        manifest=manifest,
        slots_for_layer=lambda layer_id: 2,
        run=lambda layer_id, ids, execute: execute(prepared),
    )
    method.bind_runtime(runtime)
    runner = mock.Mock()
    runner.run.return_value = StandardCombineInput(hidden)
    with mock.patch("sglang.srt.layers.moe.moe_runner.MoeRunner", return_value=runner):
        result = method.apply(layer, dispatch)
    routed_dispatch, quant_info = runner.run.call_args.args
    assert torch.equal(routed_dispatch.topk_output.topk_ids, remapped)
    assert torch.equal(routed_dispatch.topk_output.topk_weights, weights)
    assert len(quant_info.quant_scales) == 6
    assert quant_info.quant_scales[0] is tensors["w13_input_scale_quant"]
    assert quant_info.quant_scales[3] is tensors["w2_input_scale_quant"]
    assert result.hidden_states is hidden
    assert torch.equal(topk.topk_ids, ids), "original routing was mutated"
    print("Real ModelOpt config, parameter-free allocation and runner wiring passed.")
    print("Container CPU tests and SGLang import checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
