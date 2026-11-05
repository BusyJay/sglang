# SPDX-License-Identifier: Apache-2.0
"""Residual writeback with the next mHC RMS statistics, retaining BF16 boundaries."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.layernorm.mlx_hc_mix import _SOURCE, _parallel_source

_UPDATE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint tok = threadgroup_position_in_grid.y;
    threadgroup float partial[32];
    float acc = 0;
    for (uint base = 0; base < K; base += THREADS * 4)
        for (uint i = 0; i < 4; ++i) {
            uint k = base + tid * 4 + i;
            if (k < K) {
                uint h = k / D, d = k % D;
                // These two 16-bit boundaries are part of the model path.
                T product = T(float(post[tok * 4 + h]) * float(branch[tok * D + d]));
                T value = T(float(residual[tok * K + k]) + float(product));
                updated[tok * K + k] = value;
                acc += float(value) * float(value);
            }
        }
    acc = simd_sum(acc);
    if (thread_index_in_simdgroup == 0)
        partial[simdgroup_index_in_threadgroup] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float total = tid < THREADS / 32 ? partial[tid] : 0;
        total = simd_sum(total);
        if (tid == 0)
            inverse[tok] = metal::precise::rsqrt(total / float(K) + eps[0]);
    }
"""


@cache
def _update_kernel():
    return mx.fast.metal_kernel(
        name="sglang_hc_residual_rms",
        input_names=["residual", "post", "branch", "eps"],
        output_names=["updated", "inverse"],
        source=_UPDATE,
    )


@cache
def _mix_kernel(parallel):
    source = _parallel_source() if parallel else _SOURCE
    start = source.index("threadgroup float sums[32];")
    end = source.index("float result[4]")
    source = (source[:start] + source[end:]).replace("* inverse", "* inverse[tok]")
    return mx.fast.metal_kernel(
        name="sglang_hc_boundary_mix_parallel"
        if parallel
        else "sglang_hc_boundary_mix",
        input_names=["x", "weight", "inverse"],
        output_names=["mix"],
        source=source,
    )


def mlx_hc_residual_rms_mix(residual, post, branch, weight, *, eps):
    """Write a residual and reuse its square sum in the next FP32 mix.

    Returns (updated [1,R,4,D], mix [1,R,24]) on the same shapes covered by
    mlx_hc_rms_mix, or None for an uncovered configuration. Each independent
    row retains native RMS and singleton GEMV reduction order. Row count is
    dispatch data, with no context length in shader templates. No pending
    branch or mutable model state escapes the call.
    """
    if (
        residual.ndim != 4
        or residual.shape[0] != 1
        or not 1 <= residual.shape[1] <= 17
        or residual.shape[2] != 4
        or not 256 <= residual.shape[3] <= 4096
        or residual.shape[3] % 256
        or residual.dtype not in (mx.float16, mx.bfloat16)
        or post.shape != residual.shape[:-1]
        or post.dtype != residual.dtype
        or branch.shape != residual.shape[:2] + (residual.shape[3],)
        or branch.dtype != residual.dtype
        or weight.shape != (24, 4 * residual.shape[3])
        or weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    ):
        return None
    count, dim = residual.shape[1], residual.shape[3]
    k, threads = 4 * dim, min(dim, 1024)
    updated, inverse = _update_kernel()(
        inputs=[residual, post, branch, mx.array([eps], mx.float32)],
        template=[("T", residual.dtype), ("D", dim), ("K", k), ("THREADS", threads)],
        grid=(threads, count, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[residual.shape, (count,)],
        output_dtypes=[residual.dtype, mx.float32],
    )
    parallel = (
        count >= 3
        and k == 16384
        and mx.device_info()["architecture"].startswith("applegpu_g17")
    )
    mix = _mix_kernel(parallel)(
        inputs=[updated, weight, inverse],
        template=[("K", k)],
        grid=(1536, count, 1),
        threadgroup=(512 if parallel else 256, 1, 1),
        output_shapes=[(1, count, 24)],
        output_dtypes=[mx.float32],
    )[0]
    return updated, mix
