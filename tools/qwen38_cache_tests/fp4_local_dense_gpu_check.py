#!/usr/bin/env python3
"""Load real MXFP8 checkpoint projections through native merged-linear loaders."""

import argparse
import json
from pathlib import Path

import torch
from flashinfer import mxfp8_quantize
from safetensors import safe_open

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.model_loader.qwen38_fp4_local import canonicalize_mxfp8_scale_name
from sglang.srt.model_loader.qwen38_nvfp4_common import cached_checkpoint_weight_files
from sglang.srt.model_loader.qwen38_nvfp4_pack.convert import initialize_fixture
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration


def check(checkpoint):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU required")
    torch.manual_seed(20261017)
    root = Path(checkpoint)
    cached_checkpoint_weight_files(root)
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    quant = ModelOptMixedPrecisionConfig.from_config(config["quantization_config"])
    quant.packed_modules_mapping = (
        Qwen4ExpForConditionalGeneration.packed_modules_mapping
    )
    ctx, flags, parallel, backend, a2a = initialize_fixture()

    def read(name):
        with safe_open(str(root / index[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def decode(weight, scale):
        scales = torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
        return weight.float() * scales

    reports = []
    with (
        ctx.override_server_args(model_path="dummy"),
        flags.moe.override(
            runner_backend=backend.FLASHINFER_CUTLASS, a2a_backend=a2a.NONE
        ),
        parallel.override(tp_size=1, tp_rank=0),
        torch.device("cuda"),
    ):
        # BA's 48-row shards exercise partial scale tiles; shared gate/up and
        # GDN QKVZ exercise two- and four-way merged checkpoint loading.
        cases = [
            (
                "model.layers.0.mlp.shared_expert.gate_up_proj",
                ["gate_proj", "up_proj"],
                [640, 640],
            ),
            (
                "model.layers.0.linear_attn.in_proj_ba",
                ["in_proj_b", "in_proj_a"],
                [48, 48],
            ),
            (
                "model.layers.0.linear_attn.in_proj_qkvz",
                ["in_proj_qkv", "in_proj_z"],
                [2048, 2048, 6144, 6144],
            ),
        ]
        for prefix, sources, sizes in cases:
            layer = MergedColumnParallelLinear(
                2560,
                sizes,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix=prefix,
                tp_size=1,
                tp_rank=0,
            )
            weights, scales = [], []
            for i, source in enumerate(sources):
                name = (
                    prefix.rsplit(".", 1)[0].replace(
                        "model.", "model.language_model.", 1
                    )
                    + "."
                    + source
                )
                weight, scale = read(name + ".weight"), read(name + ".weight_scale")
                weights.append(weight.cuda())
                scales.append(scale.cuda())
                shard = (
                    (0, 1, 2)
                    if source == "in_proj_qkv"
                    else (3 if source == "in_proj_z" else i)
                )
                for suffix, value in (("weight", weight), ("weight_scale", scale)):
                    mapped = canonicalize_mxfp8_scale_name(name + "." + suffix, quant)
                    param = getattr(layer, mapped.rsplit(".", 1)[-1])
                    loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
                        layer, param.weight_loader
                    )
                    loader(param, value, shard)
            reference_weight = decode(torch.cat(weights), torch.cat(scales))
            torch.testing.assert_close(
                layer.weight.view(torch.uint8),
                torch.cat(weights).view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                layer.weight_scale_inv, torch.cat(scales), atol=0, rtol=0
            )
            layer.quant_method.process_weights_after_loading(layer)
            for tokens in (1, 4, 64):
                x = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16) * 0.2
                qx, sx = mxfp8_quantize(x, is_sf_swizzled_layout=False)
                expected = (
                    decode(qx, sx.reshape(-1, 80)[:tokens]) @ reference_weight.t()
                )
                actual = layer(x)[0]
                assert actual.is_contiguous()
                torch.testing.assert_close(
                    actual.float(), expected, atol=0.03, rtol=0.03
                )
                reports.append(
                    {
                        "projection": prefix,
                        "tokens": tokens,
                        "max_abs_error": (actual.float() - expected).abs().max().item(),
                    }
                )
            if sizes == [48, 48]:
                # Padding must also preserve the prequantized-input, bias and
                # graph paths while logical checkpoint parameters stay N=96.
                bias = torch.randn(96, device="cuda", dtype=torch.bfloat16) * 0.01
                qx, sx = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                biased = layer.quant_method.apply(layer, (qx, sx), bias)
                torch.testing.assert_close(
                    biased.float(), expected + bias.float(), atol=0.03, rtol=0.03
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = layer(x)[0]
                x.mul_(0.5)
                graph.replay()
                torch.testing.assert_close(captured, layer(x)[0], atol=0, rtol=0)
                assert layer.weight.shape == (96, 2560)
                assert layer.weight_mxfp8_padded.shape == (128, 2560)
                reports.append(
                    {"projection": prefix, "padded_bias_tuple_graph": "passed"}
                )
            del layer
    return {"status": "passed", "checks": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = check(args.checkpoint)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
