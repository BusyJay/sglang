#!/usr/bin/env python3
"""Audit an offline FP4-Local snapshot before allocating GPU weights."""

import argparse
import json
from pathlib import Path

from sglang.srt.model_loader.qwen38_fp4_local import is_fp4_local_config
from sglang.srt.model_loader.qwen38_mtp_common import required_expert_tensors
from sglang.srt.model_loader.qwen38_nvfp4_common import (
    cached_checkpoint_weight_files,
    resolve_cached_checkpoint,
    validate_cached_checkpoint_revision,
)
from sglang.srt.model_loader.qwen38_nvfp4_pack.convert import layer_checkpoint_entries
from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index


def audit(checkpoint, revision=None):
    root = resolve_cached_checkpoint(checkpoint, revision=revision)
    if revision:
        validate_cached_checkpoint_revision(root, object(), revision)
    files = cached_checkpoint_weight_files(root)
    config = json.loads((root / "config.json").read_text())
    if not is_fp4_local_config(config):
        raise ValueError("expected FP4-Local with NVFP4 PLE")
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    coverage = {
        str(layer): len(layer_checkpoint_entries(index, layer, 512))
        for layer in range(48)
    }
    missing = required_expert_tensors("W4A16_NVFP4") - index.keys()
    if missing:
        raise ValueError(f"missing FP4 MTP tensors: {sorted(missing)[:4]}")
    ple = {}
    for file in files:
        ple.update(read_ple_checkpoint_index(file))
    weights = {
        k.removesuffix(".weight"): v for k, v in ple.items() if k.endswith(".weight")
    }
    if len(weights) != 128:
        raise ValueError(f"expected 128 PLE weight shards, found {len(weights)}")
    import torch
    from safetensors import safe_open

    global_scales = {}
    for name in sorted(
        {base.split(".shard_")[0] + ".weight_scale_2" for base in weights}
    ):
        if name not in index:
            raise ValueError(f"missing FP4 PLE global scale: {name}")
        with safe_open(str(root / index[name]), framework="pt", device="cpu") as handle:
            scale = handle.get_tensor(name)
        if (
            scale.dtype != torch.float32
            or scale.numel() != 1
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError(f"invalid FP4 PLE global scale: {name}")
        global_scales[name] = scale.item()

    total = 0
    for base, weight in weights.items():
        scale = ple.get(base + ".weight_scale")
        if (
            weight.dtype != torch.uint8
            or weight.shape[1] != 80
            or scale is None
            or scale.shape != (weight.shape[0], 10)
            or scale.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(f"invalid packed PLE pair: {base}")
        total += weight.shape[0] * 90
    return {
        "status": "passed",
        "checkpoint": str(root),
        "revision": revision,
        "main_expert_source_tensors": coverage,
        "mtp_required_tensors": 4608,
        "ple_bytes": total,
        "ple_global_scales": global_scales,
        "weights_downloaded": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.revision)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
