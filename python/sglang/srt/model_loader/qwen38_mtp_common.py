# SPDX-License-Identifier: Apache-2.0
"""Dependency-light index selection and tensor contracts for same-card MTP."""

from __future__ import annotations

import json
import re
from pathlib import Path


def mtp_checkpoint_files(checkpoint_dir):
    path = Path(checkpoint_dir)
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    selected = {
        name: shard for name, shard in mapping.items() if name.startswith("mtp.")
    }
    if not selected:
        raise ValueError("checkpoint index contains no mtp.* tensors")
    files = []
    for shard in sorted(set(selected.values())):
        if (
            not isinstance(shard, str)
            or not shard
            or Path(shard).is_absolute()
            or ".." in Path(shard).parts
            or Path(shard).suffix != ".safetensors"
        ):
            raise ValueError(f"invalid MTP shard name: {shard!r}")
        filename = path / shard
        if not filename.is_file():
            raise ValueError(f"missing MTP shard: {filename}")
        files.append(str(filename))
    return selected, files


def required_expert_tensors(quant_algo="FP8_BLOCK_SCALES"):
    return {
        f"mtp.layers.0.mlp.experts.{expert}.{proj}.{suffix}"
        for expert in range(512)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in (
            ("weight", "weight_scale", "weight_scale_2")
            if quant_algo == "W4A16_NVFP4"
            else ("weight", "weight_scale_inv")
        )
    }


def validate_mtp_tensor(name, shape, dtype, quant_algo="FP8_BLOCK_SCALES"):
    if quant_algo == "W4A16_NVFP4":
        return validate_fp4_mtp_tensor(name, shape, dtype)
    match = re.fullmatch(
        r"mtp.layers.0.mlp.experts.(\d+).(gate_proj|up_proj|down_proj).(weight|weight_scale_inv)",
        name,
    )
    if match:
        expert, proj, suffix = match.groups()
        if int(expert) >= 512:
            raise ValueError(f"invalid MTP expert: {name}")
        expected = (2560, 640) if proj == "down_proj" else (640, 2560)
        if suffix == "weight_scale_inv":
            expected = tuple(size // 128 for size in expected)
        expected_dtype = (
            "torch.float8_e4m3fn" if suffix == "weight" else "torch.bfloat16"
        )
        if tuple(shape) != expected or str(dtype) != expected_dtype:
            raise ValueError(
                f"invalid MTP tensor {name}: shape={shape}, dtype={dtype}; expected {expected}, {expected_dtype}"
            )
    elif str(dtype) != "torch.bfloat16":
        raise ValueError(f"MTP non-expert tensor must be BF16: {name}, {dtype}")


def mtp_weights_iterator(selected, files, quant_algo="FP8_BLOCK_SCALES"):
    from safetensors import safe_open

    missing = required_expert_tensors(quant_algo) - selected.keys()
    if missing:
        raise ValueError(f"missing MTP expert weights/scales: {sorted(missing)[:8]}")
    seen = set()
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in selected:
                    continue
                shard_parts = Path(selected[name]).parts
                if Path(filename).parts[-len(shard_parts) :] != shard_parts:
                    continue
                tensor = handle.get_tensor(name)
                validate_mtp_tensor(name, tensor.shape, tensor.dtype, quant_algo)
                if name.endswith(
                    ("weight_scale_inv", "weight_scale", "weight_scale_2")
                ):
                    import torch

                    scale = tensor.float()
                    if not torch.isfinite(scale).all() or not (scale > 0).all():
                        raise ValueError(
                            f"MTP scale must be finite and positive: {name}"
                        )
                seen.add(name)
                yield name, tensor
    if seen != selected.keys():
        raise ValueError(
            f"indexed MTP tensors missing from shards: {sorted(selected.keys() - seen)[:8]}"
        )


def validate_fp4_mtp_tensor(name, shape, dtype):
    match = re.fullmatch(
        r"mtp.layers.0.mlp.experts.(\d+).(gate_proj|up_proj|down_proj).(weight|weight_scale|weight_scale_2|input_scale)",
        name,
    )
    if match:
        expert, proj, suffix = match.groups()
        if int(expert) >= 512:
            raise ValueError(f"invalid MTP expert: {name}")
        rows, cols = (2560, 640) if proj == "down_proj" else (640, 2560)
        expected = (
            ((rows, cols // 2), "torch.uint8")
            if suffix == "weight"
            else (
                ((rows, cols // 16), "torch.float8_e4m3fn")
                if suffix == "weight_scale"
                else ((), "torch.float32")
            )
        )
        if (
            tuple(shape)
            not in ((expected[0], (1,)) if not expected[0] else (expected[0],))
            or str(dtype) != expected[1]
        ):
            raise ValueError(f"invalid NVFP4 MTP tensor {name}: {shape}, {dtype}")
    elif str(dtype) != "torch.bfloat16" and not (
        str(dtype) == "torch.float32"
        and "hyper_connection" in name
        and name.endswith("norm.weight")
    ):
        raise ValueError(f"invalid non-expert MTP dtype: {name}: {dtype}")
