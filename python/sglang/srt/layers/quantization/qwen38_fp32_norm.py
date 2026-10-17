# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 Gemma grouped RMSNorm with checkpoint FP32 weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(
    X, W, Y, H: tl.constexpr, G: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    group = tl.program_id(0)
    offset = tl.arange(0, B)
    base = group * G
    value = tl.load(X + base + offset, mask=offset < G, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value, 0) / G + EPS)
    weight = tl.load(W + (base % H) + offset, mask=offset < G, other=0.0).to(tl.float32)
    tl.store(Y + base + offset, value * scale * (1.0 + weight), mask=offset < G)


def fp32_grouped_gemma_norm(x, weight, group_size, eps):
    hidden = x.shape[-1]
    group_size = group_size or hidden
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.float32
        or x.device != weight.device
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape != (hidden,)
        or hidden % group_size
    ):
        raise ValueError(
            "fused FP32-weight norm requires contiguous CUDA BF16/FP32 tensors"
        )
    output = torch.empty_like(x)
    if x.numel():
        _norm[(x.numel() // group_size,)](
            x,
            weight,
            output,
            H=hidden,
            G=group_size,
            EPS=eps,
            B=triton.next_power_of_2(group_size),
        )
    return output
