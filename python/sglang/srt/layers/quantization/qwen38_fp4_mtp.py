# SPDX-License-Identifier: Apache-2.0
"""Resident NVFP4 MTP with BF16 activations and bounded tile dequantization.

The checkpoint stays packed. No whole-expert BF16 expansion and no activation
FP4 quantization. Separate gate/up global scales are preserved.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod


@triton.jit
def _decode(packed, scale, global_scale):
    mag = packed & 7
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((packed & 8) != 0, -value, value) * scale * global_scale


@triton.jit
def _project(
    X,
    W,
    S,
    G,
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
    global_scale = tl.load(G + safe_expert * (2 if FIRST else 1))
    if FIRST:
        global_up = tl.load(G + safe_expert * 2 + 1)
    for kb in range(tl.cdiv(K, BK)):
        col = kb * BK + k
        xrow = route // TOPK if FIRST else route
        x = tl.load(X + xrow * K + col, mask=col < K, other=0).to(tl.float32)
        wr = safe_expert * (2 * N if FIRST else N) + n
        byte = tl.load(
            W + wr[:, None] * (K // 2) + col[None, :] // 2,
            mask=(n[:, None] < N) & (col[None, :] < K) & valid,
            other=0,
        )
        code = (byte >> ((col[None, :] % 2) * 4)) & 15
        scale = tl.load(
            S + wr[:, None] * (K // 16) + col[None, :] // 16,
            mask=(n[:, None] < N) & (col[None, :] < K) & valid,
            other=0.0,
        ).to(tl.float32)
        w = _decode(code, scale, global_scale).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        if FIRST:
            wr_up = wr + N
            byte_up = tl.load(
                W + wr_up[:, None] * (K // 2) + col[None, :] // 2,
                mask=(n[:, None] < N) & (col[None, :] < K) & valid,
                other=0,
            )
            code_up = (byte_up >> ((col[None, :] % 2) * 4)) & 15
            scale_up = tl.load(
                S + wr_up[:, None] * (K // 16) + col[None, :] // 16,
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


def fp4_mtp_moe(x, ids, router, w13, w2, s13, s2, g13, g2):
    tokens, hidden = x.shape
    topk = ids.shape[1]
    width = w13.shape[1] // 2
    if tokens == 0:
        return torch.empty_like(x)
    if x.dtype != torch.bfloat16 or hidden % 16 or width % 16:
        raise ValueError("NVFP4 MTP requires BF16 inputs and block-aligned geometry")
    intermediate = torch.empty((tokens * topk, width), device=x.device, dtype=x.dtype)
    projected = torch.empty(
        (tokens * topk, hidden), device=x.device, dtype=torch.float32
    )
    _project[(tokens * topk, triton.cdiv(width, 32))](
        x,
        w13,
        s13,
        g13,
        ids,
        router,
        intermediate,
        K=hidden,
        N=width,
        TOPK=topk,
        E=w13.shape[0],
        FIRST=True,
    )
    _project[(tokens * topk, triton.cdiv(hidden, 32))](
        intermediate,
        w2,
        s2,
        g2,
        ids,
        router,
        projected,
        K=width,
        N=hidden,
        TOPK=topk,
        E=w13.shape[0],
        FIRST=False,
    )
    return projected.reshape(tokens, topk, hidden).sum(1).to(x.dtype)


class Qwen38Fp4MtpMethod(ModelOptNvFp4FusedMoEMethod):
    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Runtime uses logical scales directly, not CUTLASS-swizzled duplicates.
        for name in (
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
            "w13_input_scale",
            "w2_input_scale",
        ):
            delattr(layer, name)

    def process_weights_after_loading(self, layer):
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            scale = getattr(layer, name)
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("MTP NVFP4 global scales must be finite and positive")

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        self.runner = None
        if (
            moe_runner_config.activation != "silu"
            or moe_runner_config.apply_router_weight_on_input
        ):
            raise ValueError("NVFP4 MTP supports SwiGLU and output router weighting")

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        if (
            not dispatch_output.format.is_standard()
            or dispatch_output.hidden_states_scale is not None
        ):
            raise ValueError("NVFP4 MTP requires standard BF16 dispatch")
        topk = dispatch_output.topk_output
        if not hasattr(topk, "topk_ids"):
            topk = topk.to_standard()
        output = fp4_mtp_moe(
            dispatch_output.hidden_states.contiguous(),
            topk.topk_ids.contiguous(),
            topk.topk_weights.contiguous(),
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_weight_scale_2,
            layer.w2_weight_scale_2,
        )
        return StandardCombineInput(hidden_states=output)
