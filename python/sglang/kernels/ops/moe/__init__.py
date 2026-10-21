"""Mixture-of-Experts routing / bookkeeping kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = frozenset({CapabilityRequirement.CUDA})
_HIP = frozenset({CapabilityRequirement.HIP})

register_kernel(
    KernelSpec(
        op="moe.shared_expert_gate",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.shared_expert_gate:shared_expert_gate",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
        description="FP32 sigmoid gate from a single BF16 row of width 2560, with PDL.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.moe_align_block_size",
        backend=KernelBackend.AOT,
        target="sgl_kernel:moe_align_block_size",
        format_signature=FormatSignature(
            in_place=True,
            description="align/sort expert token ids into block-padded buffers",
        ),
        description="MoE align-block-size (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.moe_align_block_size",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.moe_align:moe_align_block_size",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="MoE align-block-size (JIT variant, AOT signature)",
        ),
        description="MoE align-block-size (sglang.kernels.jit).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.topk_softmax",
        backend=KernelBackend.AOT,
        target="sgl_kernel:topk_softmax",
        capabilities=_HIP,
        format_signature=FormatSignature(
            in_place=True,
            description="top-k softmax routing weights/ids",
        ),
        description="MoE top-k softmax (sgl_kernel ROCm wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="moe.topk_softmax",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.moe_topk_softmax:topk_softmax",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="top-k softmax routing weights/ids",
        ),
        description="MoE top-k softmax (sglang.kernels.jit).",
    )
)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
    ignore_invalid_expert: bool = False,
) -> None:
    """Align and sort expert token ids into block-padded output buffers."""
    kernel = get_kernel("moe.moe_align_block_size", KernelBackend.AOT)
    if ignore_invalid_expert:
        return kernel(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            pad_sorted_token_ids,
            ignore_invalid_expert,
        )
    return kernel(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """Compute top-k softmax routing weights/ids for MoE."""
    return get_kernel("moe.topk_softmax")(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        moe_softcapping,
        correction_bias,
    )


def situ_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: float,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swizzle: bool = False,
) -> None:
    from ._jit_situ_and_mul_quant import situ_and_mul_masked_post_quant as impl

    return impl(
        input,
        output,
        output_scale,
        quant_group_size,
        masked_m,
        beta,
        linear_beta,
        scale_ue8m0,
        topk,
        transposed,
        swizzle,
    )


__all__ = ["situ_and_mul_masked_post_quant", "moe_align_block_size", "topk_softmax"]


# Triton kernels migrated from srt/layers/moe (RFC #29630, Phase 2.5);
# registered for inventory. Import them from their modules.
_PHASE25_TRITON_KERNELS = [
    ("ep_moe_kernels", "deepep_run_moe_deep_preprocess"),
    ("ep_moe_kernels", "deepep_permute_triton_kernel"),
    ("ep_moe_kernels", "deepep_post_reorder_triton_kernel"),
    ("fused_moe_triton_kernels", "invoke_fused_moe_kernel"),
    ("fused_moe_triton_kernels", "fused_moe_kernel"),
    ("fused_moe_triton_kernels", "fused_moe_kernel_gptq_awq"),
    ("mxfp8_moe_amd_gfx95", "fused_experts_mxfp8"),
    ("rocm_moe_utils", "upscale"),
    ("rocm_moe_utils", "upscale_mxfp4"),
    ("router", "fused_moe_router_shim"),
    ("deepep_waterfill_kernels", "materialize_waterfill_dispatch_fused"),
    ("fill_padded_rows", "_fill_padded_rows"),
    ("moe_fused_mul_sum", "moe_fused_mul_sum"),
]
for _mod, _fn in _PHASE25_TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"moe.{_fn.lstrip('_')}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.moe.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# Single-CTA align for tiny batches: covers the corner the AOT/JIT
# moe_align_block_size small-batch path leaves out (num_experts > 64), and is
# selected by the moe_runner call site on numel <= SMALL_NUMEL_LIMIT.
register_kernel(
    KernelSpec(
        op="moe.moe_align_small_numel",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.moe_align_small_numel:moe_align_small_numel",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True,
            description="align/sort expert token ids into block-padded buffers",
        ),
        description="MoE align-block-size, single-launch triton variant.",
    )
)

# One gather for a quantized activation and its group scales: replaces the pair
# of shuffle_rows launches the cutlass fp8 blockwise MoE used to walk the same
# dst2src map with. Byte-identical to those calls.
register_kernel(
    KernelSpec(
        op="moe.shuffle_rows_with_scales",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.shuffle_rows_with_scales:shuffle_rows_with_scales",
        capabilities=_CUDA,
        description="Row gather of quantized values plus their scales, one launch.",
    )
)


# Kernels introduced with Kimi-K3, inventoried by logical operator group.
register_kernel(
    KernelSpec(
        op="moe.situ_and_mul_masked_post_quant",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe._jit_situ_and_mul_quant:situ_and_mul_masked_post_quant",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="moe.gemma_routing_post_topk",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.gemma4_routing:gemma_routing_post_topk",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.gemma4_fused_routing",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.moe.gemma4_routing:gemma4_fused_routing",
    )
)
register_kernel(
    KernelSpec(
        op="moe.mask_topk_ids",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:mask_topk_ids",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.hash_topk",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:hash_topk",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.mega_moe_pre_dispatch",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:mega_moe_pre_dispatch",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.silu_and_mul_clamp",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:silu_and_mul_clamp",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.silu_and_mul_masked_post_quant",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:silu_and_mul_masked_post_quant",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="moe.silu_and_mul_contig_post_quant",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.dsv4:silu_and_mul_contig_post_quant",
        capabilities=frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP}),
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_refresh_route_alphas",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_route_alphas:refresh_route_alphas",
        capabilities=_CUDA,
        description="Refresh only pinned routed B12X alpha rows during concurrent fills.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_cache_partition_routes",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_expert_cache:cache_partition_routes",
        capabilities=_CUDA,
        description="Partition cache and arena expert routes using a protected GPU lookup.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.cpu_nvfp4_moe",
        backend=KernelBackend.AOT,
        target="sglang.kernels.ops.moe.cpu_nvfp4:cpu_nvfp4_moe",
        capabilities=frozenset({CapabilityRequirement(device=DeviceType.CPU)}),
        description="Explicitly built experimental AVX512 BF16 NVFP4 expert operator.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_plan",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_plan",
        capabilities=_CUDA,
        description="Device LFU admission and calibrated CPU subset routing.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_plan_indexed",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_plan_indexed",
        capabilities=_CUDA,
        description="Persistent hierarchical LFU heap planning with a free-slot ring.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_plan_stage",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_plan_stage",
        capabilities=_CUDA,
        description="Plan GPU/CPU routes and publish selected CPU work to mapped pages.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_plan_stage_indexed",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_plan_stage_indexed",
        capabilities=_CUDA,
        description="Indexed LFU planning and conditional mapped CPU publication.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_wait",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_wait",
        capabilities=_CUDA,
        description="Wait for selected CPU expert results; bypass GPU-only routes.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_stage",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_stage",
        capabilities=_CUDA,
        description="Write a small CPU expert batch into mapped host pages.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_result",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_result",
        capabilities=_CUDA,
        description="Read the mapped CPU expert result after completion.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_hybrid_merge",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.moe.qwen38_hybrid_plan:hybrid_merge",
        capabilities=_CUDA,
        description="Conditionally wait for mapped CPU output and merge into BF16.",
    )
)

register_kernel(
    KernelSpec(
        op="moe.qwen38_b12x_w4a4",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.moe.qwen38_b12x_w4a4:cached_b12x_w4a4_moe",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(12, 0))}),
        description="B12X SM12x W4A4 SwiGLU with inline slot-strided CSF scales.",
    )
)
