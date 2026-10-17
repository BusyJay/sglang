# SPDX-License-Identifier: Apache-2.0
"""SM120 W4A16 over the existing CUTLASS expert cache ABI.

Read packed weights and swizzled E4M3 scales in place. No resident expert
expansion, duplicated cache layout, or activation FP4 quantization. The pack
stores [up, gate] and alphas with activation scales folded in; undo them here.
This conservative Triton route reduction is a measurable optional candidate,
not a claim of tensor-core throughput equivalence to B12X/CuTe DSL.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.qwen38_fp4_mtp import _decode


@triton.jit
def _scale_offset(row, col, columns: tl.constexpr):
    padded_columns: tl.constexpr = triton.cdiv(columns, 4) * 4
    return (
        ((row // 128 * (padded_columns // 4) + col // 4) * 32 + row % 32) * 4
        + (row % 128) // 32
    ) * 4 + col % 4


@triton.jit
def _cache_project(
    X,
    W,
    S,
    G,
    Q,
    IDs,
    Router,
    Y,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    E: tl.constexpr,
    FIRST: tl.constexpr,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 256,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    expert = tl.load(IDs + route).to(tl.int32)
    valid = (expert >= 0) & (expert < E)
    safe_expert = tl.where(valid, expert, 0)
    n = block * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BN,), 0.0, tl.float32)
    acc_up = tl.full((BN,), 0.0, tl.float32)
    # The CUTLASS pack folds raw activation scales into g1/g2. Undo
    # that factor for A16; Q stores their reciprocal, in both projections.
    global_scale = tl.load(G + safe_expert) * tl.load(Q)
    global_up = global_scale
    for kb in range(tl.cdiv(K, BK)):
        col = kb * BK + k
        xrow = route // TOPK if FIRST else route
        x = tl.load(X + xrow * K + col, mask=col < K, other=0.0).to(tl.float32)
        # Existing CUTLASS ABI stores [up, gate], plain contiguous halves.
        logical_row = n + (N if FIRST else 0)
        rows = 2 * N if FIRST else N
        wr = safe_expert * rows + logical_row
        byte = tl.load(
            W + wr[:, None] * (K // 2) + col[None, :] // 2,
            mask=(n[:, None] < N) & (col[None, :] < K) & valid,
            other=0,
        )
        code = (byte >> ((col[None, :] % 2) * 4)) & 15
        scale = tl.load(
            S
            + safe_expert * (tl.cdiv(rows, 128) * 128) * (tl.cdiv(K // 16, 4) * 4)
            + _scale_offset(logical_row[:, None], col[None, :] // 16, K // 16),
            mask=(n[:, None] < N) & (col[None, :] < K) & valid,
            other=0.0,
        ).to(tl.float32)
        w = _decode(code, scale, global_scale).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        if FIRST:
            wr_up = wr - N
            byte_up = tl.load(
                W + wr_up[:, None] * (K // 2) + col[None, :] // 2,
                mask=(n[:, None] < N) & (col[None, :] < K) & valid,
                other=0,
            )
            code_up = (byte_up >> ((col[None, :] % 2) * 4)) & 15
            scale_up = tl.load(
                S
                + safe_expert * (tl.cdiv(rows, 128) * 128) * (tl.cdiv(K // 16, 4) * 4)
                + _scale_offset(n[:, None], col[None, :] // 16, K // 16),
                mask=(n[:, None] < N) & (col[None, :] < K) & valid,
                other=0.0,
            ).to(tl.float32)
            w_up = _decode(code_up, scale_up, global_up).to(tl.bfloat16).to(tl.float32)
            acc_up += tl.sum(w_up * x[None, :], axis=1)
    if FIRST:
        result = (acc / (1.0 + tl.exp(-acc))) * acc_up
    else:
        result = acc * tl.load(Router + route).to(tl.float32)
    tl.store(Y + route * N + n, tl.where(valid, result, 0.0), mask=n < N)


def validate_cache_tensors(tensors, hidden, width, capacity):
    """Metadata-only validation; no route read or CUDA synchronization."""
    if hidden % 16 or width % 16 or capacity <= 0:
        raise ValueError("cached W4A16 requires block-aligned geometry")
    expected = {
        "w13_weight": ((capacity, 2 * width, hidden // 2), torch.uint8),
        "w2_weight": ((capacity, hidden, width // 2), torch.uint8),
        "w13_blockscale_swizzled": (
            (
                capacity,
                triton.cdiv(2 * width, 128) * 128,
                triton.cdiv(hidden // 16, 4) * 4,
            ),
            torch.float8_e4m3fn,
        ),
        "w2_blockscale_swizzled": (
            (capacity, triton.cdiv(hidden, 128) * 128, triton.cdiv(width // 16, 4) * 4),
            torch.float8_e4m3fn,
        ),
        "g1_alphas": ((capacity,), torch.float32),
        "g2_alphas": ((capacity,), torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = tensors[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"incompatible CUTLASS cache ABI tensor {name}")
    for name in ("w13_input_scale_quant", "w2_input_scale_quant"):
        value = tensors[name]
        if value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(f"cached W4A16 needs scalar FP32 {name}")


def validate_runner_config(config):
    # Cache ABI fixes [up, gate] regardless of the original layer layout flag.
    if (
        config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or any(
            getattr(config, name, None) is not None
            for name in (
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
                "swiglu_limit",
            )
        )
    ):
        raise ValueError(
            "cached W4A16 supports standard split SwiGLU and output routing only"
        )


def cached_w4a16_moe(x, ids, router, tensors, *, chunk_tokens=16):
    if x.ndim != 2:
        raise ValueError("cached W4A16 requires a 2D activation")
    tokens, hidden = x.shape
    capacity, both_width, _ = tensors["w13_weight"].shape
    width = both_width // 2
    if x.dtype != torch.bfloat16 or ids.ndim != 2 or router.shape != ids.shape:
        raise ValueError("cached W4A16 requires BF16 and standard routing")
    if ids.shape[0] != tokens or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("cached W4A16 route shape/dtype differs from inputs")
    if not x.is_contiguous() or not ids.is_contiguous() or not router.is_contiguous():
        raise ValueError("cached W4A16 requires contiguous inputs")
    validate_cache_tensors(tensors, hidden, width, capacity)
    if not 1 <= chunk_tokens <= 64:
        raise ValueError("W4A16 chunk_tokens must be in 1..64")
    if not x.is_cuda or ids.device != x.device or router.device != x.device:
        raise ValueError("cached W4A16 requires CUDA inputs on the same device")
    if router.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("cached W4A16 requires floating router weights")
    if any(tensor.device != x.device for tensor in tensors.values()):
        raise ValueError("cached W4A16 tensors must share the input device")
    topk = ids.shape[1]
    if topk < 1:
        raise ValueError("cached W4A16 requires nonempty topk")
    output = torch.empty_like(x)
    if not tokens:
        return output
    # Bounded scratch even for 8192-token prefill. Kernels stay in the runtime's
    # existing reader/event scope; output ownership is per invocation.
    routes = min(tokens, chunk_tokens) * topk
    intermediate = torch.empty((routes, width), device=x.device, dtype=x.dtype)
    projected = torch.empty((routes, hidden), device=x.device, dtype=torch.float32)
    for start in range(0, tokens, chunk_tokens):
        count = min(chunk_tokens, tokens - start)
        local_ids, local_router = (
            ids[start : start + count],
            router[start : start + count],
        )
        route_count = count * topk
        _cache_project[(route_count, triton.cdiv(width, 32))](
            x[start : start + count],
            tensors["w13_weight"],
            tensors["w13_blockscale_swizzled"],
            tensors["g1_alphas"],
            tensors["w13_input_scale_quant"],
            local_ids,
            local_router,
            intermediate,
            K=hidden,
            N=width,
            TOPK=topk,
            E=capacity,
            FIRST=True,
        )
        _cache_project[(route_count, triton.cdiv(hidden, 32))](
            intermediate,
            tensors["w2_weight"],
            tensors["w2_blockscale_swizzled"],
            tensors["g2_alphas"],
            tensors["w2_input_scale_quant"],
            local_ids,
            local_router,
            projected,
            K=width,
            N=hidden,
            TOPK=topk,
            E=capacity,
            FIRST=False,
        )
        torch.sum(
            projected[:route_count].view(count, topk, hidden),
            dim=1,
            out=output[start : start + count],
        )
    return output
