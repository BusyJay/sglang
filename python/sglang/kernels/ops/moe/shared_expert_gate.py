"""FP32 sigmoid gate, preserving the shared-expert fused-add reduction."""

import torch

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_gate_sigmoid_mul_add_kernel,
)


def shared_expert_gate(
    hidden: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    assert hidden.is_cuda and weight.device == hidden.device
    assert (
        hidden.ndim == 2
        and 1 <= hidden.shape[0] <= 3
        and hidden.shape[1] == 2560
        and hidden.is_contiguous()
    )
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert weight.numel() == 2560 and weight.is_contiguous()
    gate = out
    if gate is None:
        gate = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
    assert gate.shape == (hidden.shape[0],) and gate.dtype == torch.float32
    assert gate.device == hidden.device and gate.is_contiguous()
    _fused_gate_sigmoid_mul_add_kernel[(hidden.shape[0],)](
        hidden,
        weight,
        None,
        gate,
        hidden_dim=2560,
        BLOCK_SIZE=4096,
        DO_ADD=False,
        USE_PDL=True,
        GATE_ONLY=True,
        num_warps=16,
        launch_pdl=True,
    )
    return gate


def shared_expert_gate_add(gate, shared, routed):
    """Apply the saved FP32 gate with the stock fused-add arithmetic."""
    assert (
        shared.shape == routed.shape and shared.dtype == routed.dtype == torch.bfloat16
    )
    assert shared.ndim == 2 and 1 <= shared.shape[0] <= 3 and shared.shape[1] == 2560
    assert shared.is_contiguous() and routed.is_contiguous()
    assert gate.shape == (shared.shape[0],) and gate.dtype == torch.float32
    assert gate.is_contiguous() and shared.device == routed.device == gate.device
    _fused_gate_sigmoid_mul_add_kernel[(shared.shape[0],)](
        None,
        gate,
        shared,
        routed,
        hidden_dim=2560,
        BLOCK_SIZE=4096,
        DO_ADD=True,
        USE_PDL=True,
        PRECOMPUTED_GATE=True,
        # This phase only applies a scalar gate; retain the reduction's
        # 16-warp layout above, but use fewer warps for the elementwise add.
        num_warps=8,
        launch_pdl=True,
    )
    return routed
