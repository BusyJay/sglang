"""Experimental SM120 MXFP8 projections with a fixed M=1/3 reduction.

Payload and scales use the existing FlashInfer E4M3 / 128x4 E8M0 contract.
Each tile handles all three possible rows and accumulates K in order without
atomics, split-K scratch, or shape-dependent autotuning.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import _mx_scale_offsets

# Runtime opt-in additionally checks the semantic role at model construction.
PROJECTION_SHAPES = {
    "shared_gate_up": (1280, 2560),
    "shared_down": (2560, 640),
    "gdn_qkvz": (16384, 2560),
    "gdn_ba": (96, 2560),
    "gdn_out": (2560, 6144),
    "attention_qkv": (13312, 2560),
    "attention_out": (2560, 6144),
}


@triton.jit
def _small_m_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.arange(0, 16)
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    groups = tl.arange(0, BLOCK_K // 32)
    scale_blocks = tl.cdiv(K // 32, 4)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        g = k0 // 32 + groups
        a = tl.load(
            a_ptr + m[:, None] * K + k[None, :],
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + n[:, None] * K + k[None, :],
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        sa = tl.load(
            a_scale_ptr + _mx_scale_offsets(m[:, None], g[None, :], scale_blocks),
            mask=(m[:, None] < M) & (g[None, :] < K // 32),
            other=0,
        )
        sw = tl.load(
            w_scale_ptr + _mx_scale_offsets(n[:, None], g[None, :], scale_blocks),
            mask=(n[:, None] < N) & (g[None, :] < K // 32),
            other=0,
        )
        acc = tl.dot_scaled(a, sa, "e4m3", tl.trans(w), sw, "e4m3", acc)
    tl.store(
        out_ptr + m[:, None] * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def qwen38_mxfp8_small_m(
    q, weight, input_scale, weight_scale, *, block_n=32, block_k=128
):
    """Return BF16 [M,N] for a whitelisted, prequantized SM120 projection."""
    m, k = q.shape
    n = weight.shape[0]
    allowed = set(PROJECTION_SHAPES.values()) | {(128, 2560)}  # padded GDN BA
    if m not in (1, 3) or (n, k) not in allowed:
        raise ValueError(f"unsupported Qwen38 small-M projection: {(m, n, k)}")
    if block_n not in (16, 32, 64) or block_k not in (128, 256):
        raise ValueError("unsupported small-M tile")
    if weight.shape[1] != k:
        raise ValueError("projection K mismatch")
    for t, dtype in (
        (q, torch.float8_e4m3fn),
        (weight, torch.float8_e4m3fn),
        (input_scale, torch.uint8),
        (weight_scale, torch.uint8),
    ):
        if t.dtype != dtype or not t.is_contiguous() or t.device != q.device:
            raise ValueError(
                "expected contiguous E4M3 payload / uint8 swizzled scales on one device"
            )
    for scale, rows in ((input_scale, m), (weight_scale, n)):
        required = triton.cdiv(rows, 128) * 128 * triton.cdiv(k // 32, 4) * 4
        if scale.numel() < required:
            raise ValueError("scale buffer is smaller than the padded 128x4 layout")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("Qwen38 small-M projection requires SM120")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=q.device)
    with torch.cuda.device(q.device):
        _small_m_kernel[(triton.cdiv(n, block_n),)](
            q,
            weight,
            input_scale,
            weight_scale,
            out,
            m,
            n,
            k,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
    return out
