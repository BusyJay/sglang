# SPDX-License-Identifier: Apache-2.0
"""Compact native-order NVFP4 pair tasks from corrected router logits."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.moe.mlx_glm import _HEADER, _SELECT_LOGITS

_WORK = r"""
    threadgroup uint active[544], partner_pick[544];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < PICKS) {
        uint expert = indices[tid], prior = 0, partner = tid;
        for (uint p = 0; p < tid; ++p) prior += indices[p] == expert;
        if (!(prior & 1))
            for (uint p = tid + 1; p < PICKS; ++p)
                if (indices[p] == expert) { partner = p; break; }
        active[tid] = !(prior & 1);
        partner_pick[tid] = partner;
        owners[tid] = tid;
        partners[tid] = tid;
        counts[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (tid < PICKS && active[tid]) {
        uint destination = 0;
        for (uint p = 0; p < tid; ++p) destination += active[p];
        owners[destination] = tid;
        partners[destination] = partner_pick[tid];
        counts[destination] = partner_pick[tid] == tid ? 1 : 2;
    }
    if (tid == 0) {
        uint total = 0;
        for (uint p = 0; p < PICKS; ++p) total += active[p];
        task_count[0] = total;
    }
"""


@cache
def _kernel():
    source = _SELECT_LOGITS.replace(
        "int(threadgroup_position_in_grid.x)", "int(simdgroup_index_in_threadgroup)"
    )
    source += "\nthreadgroup_barrier(mem_flags::mem_device);\n" + _WORK
    return mx.fast.metal_kernel(
        name="sglang_router_compact_pairs",
        input_names=["logits", "bias", "scaling"],
        output_names=[
            "indices",
            "scores",
            "owners",
            "partners",
            "counts",
            "task_count",
        ],
        source=source,
        header=_HEADER,
    )


def mlx_glm_router_select_pairs(logits, bias, *, top_k, scaling, renormalize):
    """Select independent rows and compact pairs in the same dispatch.

    The native FP32 dot precedes this operator. Tasks carry original owner and
    partner positions, so projections scatter back without sorting activations.
    Trailing task slots have zero counts. Task count is GPU data, never read by
    the host to select a launch size. Supports one sequence and 1..17 rows.
    """
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or not 1 <= logits.shape[1] <= 17
        or logits.dtype != mx.float32
        or bias.dtype != mx.float32
        or bias.shape != (logits.shape[-1],)
        or not 1 <= top_k <= min(32, logits.shape[-1])
        or not 1 <= logits.shape[-1] <= 1024
    ):
        return None
    rows, experts = logits.shape[1:]
    picks = rows * top_k
    shape = (1, rows, top_k)
    return tuple(
        _kernel()(
            inputs=[logits, bias, mx.array([scaling], mx.float32)],
            template=[
                ("E", experts),
                ("TOPK", top_k),
                ("NORM", renormalize and top_k > 1),
                ("PICKS", picks),
            ],
            grid=(32 * rows, 1, 1),
            threadgroup=(32 * rows, 1, 1),
            output_shapes=[shape, shape, (picks,), (picks,), (picks,), (1,)],
            output_dtypes=[
                mx.uint32,
                mx.float32,
                mx.uint32,
                mx.uint32,
                mx.uint32,
                mx.uint32,
            ],
        )
    )
