# SPDX-License-Identifier: Apache-2.0
"""Lossless checkpoint adapters for Qwen3.8 FP4-Local.

CSF slab contract: sixteen row bases followed by row-major low-nibble-first
codes; uint32 exceptions contain a 24-bit logical position and an 8-bit value.
See FlashInfer csf_encode.py (local-inference-lab/flashinfer). Decode at pack
build time, never at an expert cache miss. No B12X runtime dependency.
"""

from __future__ import annotations


def decode_nvfp4_csf(fixed, exceptions, shape):
    import torch

    rows, columns = map(int, shape)
    if rows <= 0 or rows % 16 or columns <= 0 or columns % 2:
        raise ValueError("CSF requires sixteen-row slabs and even scale columns")
    if rows * columns > 1 << 24:
        raise ValueError("CSF matrix exceeds the 24-bit position field")
    expected = (rows // 16, 16 * (1 + columns // 2))
    if fixed.dtype != torch.uint8 or tuple(fixed.shape) != expected:
        raise ValueError(f"invalid CSF fixed stream: expected uint8 {expected}")
    if exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
        raise ValueError("CSF exceptions must be a uint32 vector")
    if fixed.device.type != "cpu" or exceptions.device.type != "cpu":
        raise ValueError("checkpoint CSF decoding requires CPU tensors")
    slab = fixed.contiguous()
    bases = slab[:, :16].reshape(rows, 1).to(torch.int16)
    packed = slab[:, 16:].reshape(rows, columns // 2)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, columns)
    values = bases + codes.to(torch.int16)
    if (values > 255).any():
        raise ValueError("CSF base plus code overflows a scale byte")
    output = values.to(torch.uint8).reshape(-1)
    records = exceptions.to(torch.int64)
    positions = records & 0xFFFFFF
    if positions.numel():
        if (
            positions >= output.numel()
        ).any() or positions.unique().numel() != positions.numel():
            raise ValueError("CSF exception positions are out of bounds or duplicated")
        output[positions] = (records >> 24).to(torch.uint8)
    # Reinterpret the restored bytes; do not numerically cast uint8 to FP8.
    return output.reshape(rows, columns).view(torch.float8_e4m3fn)


def scale_sources(index, name):
    """Return either a plain scale or a complete CSF pair; reject ambiguity."""
    fixed, exceptions = name + ".nvfp4_csf_fixed", name + ".nvfp4_csf_exceptions"
    plain = name in index
    encoded = fixed in index or exceptions in index
    if plain and encoded:
        raise ValueError(f"ambiguous plain/CSF scale: {name}")
    if encoded:
        if fixed not in index or exceptions not in index:
            raise ValueError(f"incomplete CSF scale: {name}")
        return fixed, exceptions
    if not plain:
        raise ValueError(f"checkpoint index is missing {name}")
    return (name,)


def is_fp4_local_config(config):
    text = config.get("text_config", config)
    return text.get("ple_embedding_dtype") == "nvfp4"


def canonicalize_mxfp8_scale_name(name, quant_config):
    """ModelOpt calls UE8M0 scales weight_scale; SGLang calls them *_inv."""
    if name.endswith(".weight_scale"):
        prefix = name[: -len(".weight_scale")]
        if quant_config.resolve_quant_algo(prefix) == "MXFP8":
            return prefix + ".weight_scale_inv"
    return name
