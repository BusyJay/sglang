# SPDX-License-Identifier: Apache-2.0
"""Export one layer after the resident ModelOpt SM120 post-load transform.

This helper performs no model download or checkpoint conversion. Call it on
an existing resident FusedMoE fixture after process_weights_after_loading().
A CPU device is accepted only for synthetic format/control-path tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .format import LAYER_TENSORS, PACK_TENSORS, REQUIRED_TENSORS, SUPPORTED_DTYPES


def export_runtime_layer(layer, output, *, synthetic=False):
    import torch

    output = Path(output)
    tensors = {}
    for name in PACK_TENSORS:
        tensor = getattr(layer, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"post-transformed layer is missing {name}")
        if not synthetic:
            if tensor.device.type != "cuda" or torch.cuda.get_device_capability(
                tensor.device
            ) != (12, 0):
                raise ValueError("runtime export requires the SM120 ModelOpt transform")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported transformed dtype {dtype}")
        if name in LAYER_TENSORS:
            if tensor.numel() != 1 or dtype != "float32":
                raise ValueError(f"{name} must be a scalar float32 input scale")
            tensor = tensor.reshape(1)
        elif tensor.ndim == 0 or tensor.shape[0] != layer.num_experts:
            raise ValueError(f"{name} does not contain every expert")
        tensors[name] = tensor.detach()
    # Validate the actual consumer's view requirements, without invoking a kernel.
    for name in ("w13_weight", "w2_weight"):
        if tensors[name].dtype != torch.uint8 or tensors[name].shape[-1] % 8:
            raise ValueError(f"{name} is not CUTLASS-compatible packed NVFP4")
    for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
        if tensors[name].element_size() != 1 or tensors[name].shape[-1] % 4:
            raise ValueError(f"{name} is not CUTLASS-compatible swizzled scale storage")
    for name in ("g1_alphas", "g2_alphas"):
        if tensors[name].dtype != torch.float32 or tensors[name].shape != (
            layer.num_experts,
        ):
            raise ValueError(f"{name} must contain one float32 alpha per expert")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, tensor in tensors.items():
        filename = f"{name}.bin"
        # Stream per expert, avoiding a second complete layer-sized RAM copy.
        with (output / filename).open("wb") as stream:
            chunks = tensor if name in REQUIRED_TENSORS else [tensor]
            for chunk in chunks:
                raw = chunk.contiguous().cpu().reshape(-1).view(torch.uint8)
                # NumPy exposes contiguous storage without creating millions
                # of Python integer objects for a real expert.
                stream.write(memoryview(raw.numpy()))
        metadata[name] = {
            "file": filename,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "metadata.json")
    return metadata
