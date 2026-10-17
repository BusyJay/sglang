#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert local checkpoint experts one layer at a time on SM120.

Never calls HF download APIs. Names and complete projection/scale coverage are
validated against the local safetensors index before any tensors are read.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECTIONS = {
    "gate_proj": ("w13", "w1"),
    "up_proj": ("w13", "w3"),
    "down_proj": ("w2", "w2"),
}
from sglang.srt.model_loader.qwen38_fp4_local import decode_nvfp4_csf, scale_sources

SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def layer_checkpoint_entries(index, layer_id, num_experts):
    prefix = f"model.language_model.layers.{layer_id}.mlp.experts."
    entries = {}
    for expert in range(num_experts):
        for projection, (target, shard) in PROJECTIONS.items():
            for suffix in SUFFIXES:
                name = f"{prefix}{expert}.{projection}.{suffix}"
                sources = (
                    scale_sources(index, name) if suffix == "weight_scale" else (name,)
                )
                for source in sources:
                    if source not in index:
                        raise ValueError(f"checkpoint index is missing {source}")
                    entries[source] = (
                        index[source],
                        f"{target}_{suffix}",
                        shard,
                        expert,
                    )
    return entries


def initialize_fixture():

    from sglang.test.layer_ut_utils import init_single_process_dist

    # Gloo is used only for the required world=1 topology setup.
    init_single_process_dist()
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    return get_context(), get_flags(), get_parallel(), MoeRunnerBackend, MoeA2ABackend


def create_resident_layer(config, layer_id):
    import torch

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptMixedPrecisionConfig,
    )

    delegate = ModelOptMixedPrecisionConfig.from_config(config["quantization_config"])
    text = config.get("text_config", config)
    prefix = f"model.layers.{layer_id}.mlp.experts"
    if delegate.resolve_quant_algo(prefix) != "NVFP4":
        raise ValueError(f"{prefix} is not an NVFP4 layer")
    with torch.device("cuda"):
        return FusedMoE(
            num_experts=text["num_experts"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"],
            layer_id=layer_id,
            top_k=text["num_experts_per_tok"],
            params_dtype=torch.bfloat16,
            quant_config=delegate,
            prefix=prefix,
            activation=text.get("hidden_act", "silu"),
            gate_up_interleaved=False,
            inplace=False,
        )


def load_layer_checkpoint(layer, checkpoint, index, layer_id):
    from safetensors import safe_open

    entries = layer_checkpoint_entries(index, layer_id, layer.num_experts)
    # Lexical containment plus ".." rejection keeps names inside the requested
    # checkpoint; HuggingFace caches legitimately reach sibling blob storage
    # through relative symlinks, which is_file() follows without moving the
    # lexical path outside the directory.
    root = Path(checkpoint).expanduser()
    by_file: dict[Path, list] = {}
    for name, (filename, target, shard, expert) in entries.items():
        if filename.startswith("/") or ".." in Path(filename).parts:
            raise ValueError(f"invalid or missing checkpoint shard {filename}")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"invalid or missing checkpoint shard {filename}")
        by_file.setdefault(path, []).append((name, target, shard, expert))
    # Pairs can span safetensors files. Hold only this layer's compressed scales;
    # decode each pair as soon as its second stream arrives.
    pending = {}
    for path, tensors in sorted(by_file.items()):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name, target, shard, expert in tensors:
                tensor = handle.get_tensor(name)
                if ".nvfp4_csf_" in name:
                    base, stream = name.rsplit(".nvfp4_csf_", 1)
                    parts = pending.setdefault(base, {})
                    parts[stream] = tensor
                    if len(parts) != 2:
                        continue
                    weight_name = base[: -len("weight_scale")] + "weight"
                    # Logical scale geometry is determined by the expert shape.
                    projection = weight_name.rsplit(".", 2)[-2]
                    text_rows, text_cols = (
                        (
                            layer.w2_weight.shape[1],
                            layer.intermediate_size_per_partition,
                        )
                        if projection == "down_proj"
                        else (
                            layer.intermediate_size_per_partition,
                            layer.w2_weight.shape[1],
                        )
                    )
                    tensor = decode_nvfp4_csf(
                        parts["fixed"],
                        parts["exceptions"],
                        (text_rows, text_cols // 16),
                    )
                    del pending[base]
                layer.weight_loader(
                    getattr(layer, target),
                    tensor,
                    target,
                    shard_id=shard,
                    expert_id=expert,
                )
    if pending:
        raise ValueError("incomplete CSF streams after reading checkpoint")
    # The cached CUTLASS ABI has one GEMM1 alpha per expert. Refuse a
    # checkpoint requiring different gate/up alphas rather than corrupt it.
    import torch

    global_scales = layer.w13_weight_scale_2
    if (
        global_scales.ndim == 2
        and global_scales.shape[1] == 2
        and not torch.equal(global_scales[:, 0], global_scales[:, 1])
    ):
        raise ValueError("CUTLASS cache pack requires equal gate/up global scales")
    layer.quant_method.process_weights_after_loading(layer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", help="existing Hugging Face hub cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument(
        "--layers-per-process",
        type=int,
        default=4,
        help="bound CUDA lifetime during parity by converting this many layers per subprocess",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=["sync", "async", "device", "threaded"],
        default="sync",
    )
    parser.add_argument(
        "--parity",
        action="store_true",
        help="compare resident vs cached CUTLASS before releasing each layer",
    )
    args = parser.parse_args()
    if args.layers_per_process < 1:
        parser.error("--layers-per-process must be positive")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("conversion requires SM120 CUDA")
    from sglang.srt.model_loader.qwen38_nvfp4_common import (
        cached_checkpoint_weight_files,
        resolve_cached_checkpoint,
        validate_cached_checkpoint_revision,
    )

    checkpoint = resolve_cached_checkpoint(
        args.checkpoint, revision=args.checkpoint_revision, cache_dir=args.cache_dir
    )
    cached_checkpoint_weight_files(checkpoint)
    validate_cached_checkpoint_revision(checkpoint, object(), args.checkpoint_revision)
    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    text = config.get("text_config", config)
    layers = (
        args.layers
        if args.layers is not None
        else list(range(text["num_hidden_layers"]))
    )
    if len(set(layers)) != len(layers) or any(
        x < 0 or x >= text["num_hidden_layers"] for x in layers
    ):
        parser.error("invalid or duplicate layer IDs")
    # Validate all requested coverage first, before CUDA model construction.
    for layer_id in layers:
        layer_checkpoint_entries(index, layer_id, text["num_experts"])
    if args.parity and len(layers) > args.layers_per_process:
        # CUDA/backend caches can retain resident tensors beyond Python GC.
        # A fresh process bounds those allocations over a full 48-layer audit.
        # Children still run parity for every requested layer; no skip/resume
        # shortcut is used, and assembly only follows the last successful batch.
        for start in range(0, len(layers), args.layers_per_process):
            batch = layers[start : start + args.layers_per_process]
            command = [
                sys.executable,
                "-m",
                "sglang.srt.model_loader.qwen38_nvfp4_pack.convert",
                "--checkpoint",
                str(checkpoint),
                "--output",
                args.output,
                "--checkpoint-revision",
                args.checkpoint_revision,
                "--sglang-commit",
                args.sglang_commit,
                "--runtime-mode",
                args.runtime_mode,
                "--layers-per-process",
                str(args.layers_per_process),
                "--parity",
                "--layers",
                *map(str, batch),
            ]
            if args.resume:
                command.append("--resume")
            if args.assemble and start + len(batch) == len(layers):
                command.append("--assemble")
            print(f"Converting isolated parity batch {batch}", flush=True)
            subprocess.run(command, check=True)
        return 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from .build import build_pack
    from .export import export_runtime_layer

    ctx, flags, parallel, backend, a2a = initialize_fixture()
    with (
        ctx.override_server_args(model_path="dummy"),
        flags.moe.override(
            runner_backend=backend.FLASHINFER_CUTLASS, a2a_backend=a2a.NONE
        ),
        parallel.override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
    ):
        for layer_id in layers:
            destination = output / f"layer-{layer_id:03d}"
            marker = destination / "conversion.json"
            if args.resume and marker.is_file():
                saved = json.loads(marker.read_text())
                if saved != {
                    "checkpoint_revision": args.checkpoint_revision,
                    "sglang_commit": args.sglang_commit,
                    "layer_id": layer_id,
                }:
                    raise ValueError(f"resume provenance mismatch: {destination}")
                if args.parity:
                    raise ValueError(
                        "--resume cannot skip parity; rerun on a fresh output"
                    )
                from .format import PACK_TENSORS, TensorRecord

                metadata = json.loads((destination / "metadata.json").read_text())
                for name in PACK_TENSORS:
                    info = metadata[name]
                    tensor_path = destination / info["file"]
                    size = tensor_path.stat().st_size
                    record = TensorRecord(0, size, tuple(info["shape"]), info["dtype"])
                    errors = record.validate(payload_nbytes=size, label=name)
                    if errors:
                        raise ValueError(f"invalid resumed export: {errors}")
                continue
            # Unmarked leftovers from interrupted reruns may occupy this dir;
            # resume only skips provenance-marked exports, so anything else here
            # is stale and must be cleared by the converting process itself
            # (prior artifacts can be owned by other users on the pack volume).
            if destination.exists():
                shutil.rmtree(destination)
            layer = create_resident_layer(config, layer_id)
            load_layer_checkpoint(layer, checkpoint, index, layer_id)
            export_runtime_layer(layer, destination)
            if args.parity:
                from sglang.srt.model_loader.qwen38_nvfp4_pack.parity import (
                    check_exported_layer,
                )

                report = check_exported_layer(layer, destination, args.runtime_mode)
                (destination / "parity.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            marker.write_text(
                json.dumps(
                    {
                        "checkpoint_revision": args.checkpoint_revision,
                        "sglang_commit": args.sglang_commit,
                        "layer_id": layer_id,
                    },
                    indent=2,
                )
                + "\n"
            )
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Converted layer {layer_id}", flush=True)
    if args.assemble:
        # Every layer needs matching provenance before assembling a production pack.
        for layer_id in range(text["num_hidden_layers"]):
            saved = json.loads(
                (output / f"layer-{layer_id:03d}" / "conversion.json").read_text()
            )
            if saved != {
                "checkpoint_revision": args.checkpoint_revision,
                "sglang_commit": args.sglang_commit,
                "layer_id": layer_id,
            }:
                raise ValueError("cannot assemble exports from different revisions")
        build_pack(
            output,
            output / "experts.bin",
            output / "manifest.json",
            checkpoint_revision=args.checkpoint_revision,
            sglang_commit=args.sglang_commit,
            num_layers=text["num_hidden_layers"],
            num_experts=text["num_experts"],
            top_k=text["num_experts_per_tok"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
