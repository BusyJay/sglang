# SPDX-License-Identifier: Apache-2.0
"""Overlap the independent shared expert with direct cache fills in decode."""

from __future__ import annotations

import os
from contextvars import ContextVar

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    get_current_capture_pool,
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

    if os.getenv("SGLANG_QWEN38_SHARED_FILL_OVERLAP", "1") == "0":
        return False
    runtime = method.runtime
    config = method.moe_runner_config
    return bool(
        phase in ("decode", "verify")
        and hidden_states.is_cuda
        and (hidden_states.shape[0] == 1 or phase == "verify")
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


class SharedExpertCall:
    """Per-break graph state; the input crosses the break as a flat tensor arg.

    No activation closure is retained by the graph replay callback. The outer
    breakable graph pins incoming activations. Retain the shared graph's output
    until the following segment has consumed it. Eager calls never build a graph.
    """

    def __init__(self, block):
        self.block = block
        self.graph = None
        self.output = None

    def capture(self, hidden_states):
        if self.graph is not None:
            raise RuntimeError("shared expert graph already captured")
        graph = torch.cuda.CUDAGraph()
        # Called by the expert capture stub AFTER the preceding segment ends.
        # Reuse its pool rather than reserve a private allocator slab per layer.
        # Runtime reader events join this graph before the next segment starts.
        with torch.cuda.graph(
            graph,
            pool=get_current_capture_pool(),
            stream=torch.cuda.current_stream(),
        ):
            result = self.block._forward_shared_experts(hidden_states, apply_gate=False)
        self.output = result
        self.graph = graph

    def run(self, hidden_states):
        if self.graph is None:
            self.output = self.block._forward_shared_experts(
                hidden_states, apply_gate=False
            )
        else:
            self.graph.replay()


def run_shared_overlap(block, hidden_states, phase):
    from sglang.srt.models.qwen2_moe import (
        fused_gate_sigmoid_mul_add,
        reduce_moe_output,
    )

    # This function stays INSIDE the ordinary graph segments. Only quant's
    # existing expert boundary breaks capture; its stub prepares the shared
    # graph, and its runtime callback replays it alongside fill.
    call = SharedExpertCall(block)
    token = _during_fill.set((call, hidden_states))
    try:
        routed = block._forward_router_experts(hidden_states)
    finally:
        _during_fill.reset(token)
    shared = call.output
    if shared is None:
        raise RuntimeError("direct-fill transport did not execute shared expert")
    if block.shared_expert_gate is not None:
        fused_gate_sigmoid_mul_add(
            hidden_states,
            block.shared_expert_gate.weight.squeeze(),
            shared,
            routed,
        )
    else:
        routed.add_(shared)
    return (
        reduce_moe_output(routed) if getattr(block, "reduce_results", True) else routed
    )
