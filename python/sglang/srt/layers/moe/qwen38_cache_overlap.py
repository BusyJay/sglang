# SPDX-License-Identifier: Apache-2.0
"""Overlap the independent shared expert with direct cache fills in decode."""

from __future__ import annotations

from contextvars import ContextVar

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

_during_fill = ContextVar("qwen38_shared_during_fill", default=None)


def current_fill_overlap():
    return _during_fill.get()


def fill_with_overlap(compute, fill_stream, ready, filled, submit_fill, independent):
    """Order cache writers/readers while an independent branch stays on compute."""
    ready.record(compute)
    fill_stream.wait_event(ready)
    try:
        with torch.cuda.stream(fill_stream):
            try:
                submit_fill()
            finally:
                # Even a failed enqueue must order any previously issued work.
                filled.record(fill_stream)
        independent()
    finally:
        # The caller records cache-reader completion after this join, including
        # when the independent branch raises. No host synchronization is needed.
        compute.wait_event(filled)


def can_overlap_shared(method, block, hidden_states, phase):
    from sglang.srt.layers.moe import get_moe_a2a_backend

    runtime = method.runtime
    config = method.moe_runner_config
    return bool(
        phase == "decode"
        and hidden_states.is_cuda
        and hidden_states.shape[0] == 1
        and runtime is not None
        and getattr(runtime, "supports_fill_overlap", False)
        and runtime.miss_mailbox is not None
        and runtime.fused_decode_plan
        and block.shared_expert is not None
        and not block.enable_shared_expert_fusion
        and block.tp_size == 1
        and get_moe_a2a_backend().is_none()
        and config is not None
        and not config.inplace
        and not config.apply_router_weight_on_input
    )


def _capture_shared_output(_block, hidden_states, _phase):
    # Capture-time graph outputs (including router IDs) have not been replayed.
    # The entire shared/routed boundary must execute only at warmup/replay.
    return torch.zeros_like(hidden_states)


@eager_on_graph(True, capture_stub=_capture_shared_output)
def run_shared_overlap(block, hidden_states, phase):
    from sglang.srt.models.qwen2_moe import (
        fused_gate_sigmoid_mul_add,
        reduce_moe_output,
    )
    from sglang.srt.runtime_context import get_forward

    shared = []

    def compute_shared():
        if shared:
            raise RuntimeError("shared expert callback executed more than once")
        shared.append(block._forward_shared_experts(hidden_states, apply_gate=False))

    token = _during_fill.set(compute_shared)
    try:
        # Preserve standard router, dispatcher and quant-method behavior. Replay
        # occurs after the model's original Python forward scopes have exited.
        with get_forward().scoped(qwen38_execution_phase=phase):
            routed = block._forward_router_experts(hidden_states)
    finally:
        _during_fill.reset(token)
    if len(shared) != 1 or shared[0] is None:
        raise RuntimeError("direct-fill transport did not execute shared expert")
    if block.shared_expert_gate is not None:
        fused_gate_sigmoid_mul_add(
            hidden_states,
            block.shared_expert_gate.weight.squeeze(),
            shared[0],
            routed,
        )
    else:
        routed.add_(shared[0])
    return reduce_moe_output(routed)
