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
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"invalid MTP shard name: {shard!r}")
        filename = path / shard
        if not filename.is_file():
            raise ValueError(f"missing MTP shard: {filename}")
        files.append(str(filename))
    return selected, files


def required_expert_tensors():
    return {
        f"mtp.layers.0.mlp.experts.{expert}.{proj}.{suffix}"
        for expert in range(512)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    }


def validate_mtp_tensor(name, shape, dtype):
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


def mtp_weights_iterator(selected, files):
    from safetensors import safe_open

    missing = required_expert_tensors() - selected.keys()
    if missing:
        raise ValueError(f"missing MTP expert weights/scales: {sorted(missing)[:8]}")
    seen = set()
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in selected or selected[name] != Path(filename).name:
                    continue
                tensor = handle.get_tensor(name)
                validate_mtp_tensor(name, tensor.shape, tensor.dtype)
                if name.endswith("weight_scale_inv"):
                    import torch

                    if not torch.isfinite(tensor).all() or not (tensor > 0).all():
                        raise ValueError(
                            f"MTP scale must be finite and positive: {name}"
                        )
                seen.add(name)
                yield name, tensor
    if seen != selected.keys():
        raise ValueError(
            f"indexed MTP tensors missing from shards: {sorted(selected.keys() - seen)[:8]}"
        )
