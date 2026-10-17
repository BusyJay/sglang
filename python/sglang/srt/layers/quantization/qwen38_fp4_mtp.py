# SPDX-License-Identifier: Apache-2.0
"""Resident NVFP4 MTP with CUTLASS A4 activations and A16 rollback.

The checkpoint stays packed. A4 is the default; A16 is selected at worker
startup for rollback. Both paths preserve separate gate/up global scales.
"""

import logging
import os

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


def fp4_mtp_moe(x, ids, router, w13, w2, s13, s2, g13, g2, *, chunk_tokens=128):
    """Keep route scratch bounded during draft extend, including long prefill."""
    tokens, hidden = x.shape
    topk = ids.shape[1]
    width = w13.shape[1] // 2
    if tokens == 0:
        return torch.empty_like(x)
    if x.dtype != torch.bfloat16 or hidden % 16 or width % 16:
        raise ValueError("NVFP4 MTP requires BF16 inputs and block-aligned geometry")
    if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("NVFP4 MTP chunk_tokens must be a positive integer")
    if tokens > chunk_tokens:
        # Draft extend also sees full prefill chunks. At 8192 tokens/topk=10,
        # the original FP32 projected buffer alone occupies 800 MiB. Keep only
        # one small tile alive, preserving the original FP32 sum/BF16 rounding.
        # Decode and graph-captured small inputs retain the original path below.
        output = torch.empty_like(x)
        for start in range(0, tokens, chunk_tokens):
            end = min(start + chunk_tokens, tokens)
            output[start:end].copy_(
                fp4_mtp_moe(
                    x[start:end],
                    ids[start:end],
                    router[start:end],
                    w13,
                    w2,
                    s13,
                    s2,
                    g13,
                    g2,
                    chunk_tokens=chunk_tokens,
                )
            )
        return output
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


@triton.jit
def _prepare_mtp_routes_kernel(
    IDs,
    Weights,
    Ratio,
    OutIDs,
    OutWeights,
    TOPK: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    ids = tl.load(IDs + token * TOPK + slot, slot < TOPK, other=-1)
    weights = tl.load(Weights + token * TOPK + slot, slot < TOPK, other=0).to(
        tl.float32
    )
    valid = (slot < TOPK) & (ids >= 0) & (ids < EXPERTS)
    same = (ids[:, None] == ids[None, :]) & valid[None, :]
    first = tl.sum((same & (slot[None, :] < slot[:, None])).to(tl.int32), 1) == 0
    combined = tl.sum(tl.where(same, weights[None, :], 0.0), 1)
    ratio = tl.load(Ratio + tl.minimum(tl.maximum(ids, 0), EXPERTS - 1))
    keep = valid & first
    tl.store(OutIDs + token * TOPK + slot, tl.where(keep, ids, -1), slot < TOPK)
    tl.store(
        OutWeights + token * TOPK + slot,
        tl.where(keep, combined * ratio, 0.0),
        slot < TOPK,
    )


def _prepare_mtp_routes(ids, weights, ratio, experts):
    # CUTLASS's reverse permutation is indexed by (token, expert), so repeated
    # expert IDs must be merged before dispatch. Otherwise one route overwrites
    # another and graph replay can read a different route's intermediate.
    if ids.is_cuda:
        out_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
        out_weights = torch.empty(
            weights.shape, dtype=torch.float32, device=weights.device
        )
        _prepare_mtp_routes_kernel[(ids.shape[0],)](
            ids.contiguous(),
            weights.contiguous(),
            ratio,
            out_ids,
            out_weights,
            TOPK=ids.shape[1],
            EXPERTS=experts,
            BLOCK=triton.next_power_of_2(ids.shape[1]),
        )
        return out_ids, out_weights
    valid = (ids >= 0) & (ids < experts)
    same = (ids[:, :, None] == ids[:, None, :]) & valid[:, None, :]
    slots = torch.arange(ids.shape[1], device=ids.device)
    first = ~(same & (slots[None, :] < slots[:, None])).any(-1)
    combined = torch.where(same, weights.float()[:, None, :], 0.0).sum(-1)
    keep = valid & first
    scaled = combined * ratio[ids.clamp(0, experts - 1).long()]
    return torch.where(keep, ids, -1).int(), torch.where(keep, scaled, 0.0)


class Qwen38Fp4MtpMethod(ModelOptNvFp4FusedMoEMethod):
    @property
    def load_up_proj_weight_first(self):
        # Keep checkpoint loading independent of the draft's global backend.
        # A16 consumes [gate, up]; A4 swaps explicitly after all scales load.
        return False

    def __init__(self, quant_config):
        super().__init__(quant_config)
        mode = os.getenv("SGLANG_QWEN38_MTP_ACTIVATION", "w4a4")
        if mode not in ("w4a16", "w4a4"):
            raise ValueError("SGLANG_QWEN38_MTP_ACTIVATION must be w4a16 or w4a4")
        self.activation_mode = mode

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Load only checkpoint-layout scales. A4 later reorders that storage
        # in place instead of retaining a second CUTLASS-layout allocation.
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
        if self.activation_mode == "w4a4":
            self._prepare_w4a4(layer)
        logging.getLogger(__name__).info(
            "Qwen38 MTP expert activation: %s", self.activation_mode
        )

    def _prepare_w4a4(self, layer):
        from sglang.srt.layers.quantization.utils import swizzle_blockscale

        if getattr(layer, "_qwen38_mtp_w4a4_prepared", False):
            raise RuntimeError("MTP W4A4 weights were already prepared")
        # CUTLASS consumes [up, gate]. Swap one expert at a time so loading
        # never duplicates the complete 800 MiB gate/up tensor.
        for tensor in (layer.w13_weight, layer.w13_weight_scale):
            half = tensor.shape[1] // 2
            for expert in range(tensor.shape[0]):
                saved = tensor.data[expert, :half].clone()
                tensor.data[expert, :half].copy_(tensor.data[expert, half:])
                tensor.data[expert, half:].copy_(saved)
        # A4 never consumes checkpoint-order scales after this point. Reorder
        # each expert back into its existing parameter storage, then alias it
        # for CUTLASS. Retaining both layouts costs 150 MiB for 512 experts and
        # unnecessarily reduces the automatically sized KV pool.
        for tensor in (layer.w13_weight_scale, layer.w2_weight_scale):
            if tensor.shape[1] % 128 or tensor.shape[2] % 4:
                raise ValueError("MTP A4 scales require unpadded CUTLASS geometry")
            for expert in range(tensor.shape[0]):
                tensor.data[expert].copy_(swizzle_blockscale(tensor[expert]))
        layer.w13_blockscale_swizzled = layer.w13_weight_scale.detach()
        layer.w2_blockscale_swizzled = layer.w2_weight_scale.detach()
        layer.w13_input_scale_quant = torch.ones(
            (), dtype=torch.float32, device=layer.w13_weight.device
        )
        layer.w2_input_scale_quant = torch.ones_like(layer.w13_input_scale_quant)
        # The fused GEMM has one alpha for both halves. Use the gate alpha,
        # then correct the linear up branch through the final route weight:
        # silu(gate * Gx) * (gate * Ux) * (up/gate).
        layer.g1_alphas = layer.w13_weight_scale_2[:, 0].contiguous()
        layer.g2_alphas = layer.w2_weight_scale_2
        layer.mtp_up_scale_ratio = (
            layer.w13_weight_scale_2[:, 1] / layer.w13_weight_scale_2[:, 0]
        ).contiguous()
        if not torch.isfinite(layer.mtp_up_scale_ratio).all():
            raise ValueError("MTP gate/up scale ratio must be finite")
        layer._qwen38_mtp_w4a4_prepared = True

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        self.runner = None
        if (
            moe_runner_config.activation != "silu"
            or moe_runner_config.apply_router_weight_on_input
            or not moe_runner_config.is_gated
            or any(
                getattr(moe_runner_config, name) is not None
                for name in (
                    "gemm1_alpha",
                    "gemm1_beta",
                    "gemm1_clamp_limit",
                    "swiglu_limit",
                )
            )
        ):
            raise ValueError("NVFP4 MTP supports SwiGLU and output router weighting")
        if self.activation_mode == "w4a4":
            from sglang.srt.layers.moe.moe_runner import (  # noqa: F401
                MoeRunner,
                flashinfer_cutlass,
            )
            from sglang.srt.layers.moe.utils import MoeRunnerBackend

            self.runner = MoeRunner(
                MoeRunnerBackend.FLASHINFER_CUTLASS, moe_runner_config
            )

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
        if self.activation_mode == "w4a4":
            return self._apply_w4a4(layer, dispatch_output, topk)
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

    def _apply_w4a4(self, layer, dispatch_output, topk):
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        if dispatch_output.hidden_states.shape[0] == 0:
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )

            return StandardCombineInput(hidden_states=dispatch_output.hidden_states)
        # Preserve the A16 loader's bounded draft-extend scratch. CUTLASS
        # allocates workspace per call; never submit a whole 8192-token chunk.
        tokens = dispatch_output.hidden_states.shape[0]
        if tokens > 128:
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )

            output = torch.empty_like(dispatch_output.hidden_states)
            for start in range(0, tokens, 128):
                end = min(start + 128, tokens)
                sliced = StandardTopKOutput(
                    topk.topk_weights[start:end],
                    topk.topk_ids[start:end],
                    (
                        topk.router_logits[start:end]
                        if topk.router_logits is not None
                        else None
                    ),
                )
                chunk = dispatch_output._replace(
                    hidden_states=dispatch_output.hidden_states[start:end],
                    topk_output=sliced,
                )
                output[start:end].copy_(
                    self._apply_w4a4(layer, chunk, sliced).hidden_states
                )
            return StandardCombineInput(hidden_states=output)
        ids, weights = _prepare_mtp_routes(
            topk.topk_ids,
            topk.topk_weights,
            layer.mtp_up_scale_ratio,
            layer.num_local_experts,
        )
        corrected = StandardTopKOutput(weights, ids, topk.router_logits)
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            output_dtype=torch.bfloat16,
            quant_scales=[
                layer.w13_input_scale_quant,
                layer.w13_blockscale_swizzled,
                layer.g1_alphas,
                layer.w2_input_scale_quant,
                layer.w2_blockscale_swizzled,
                layer.g2_alphas,
            ],
            apply_routed_scaling_factor=False,
        )
        return self.runner.run(
            dispatch_output._replace(topk_output=corrected), quant_info
        )
