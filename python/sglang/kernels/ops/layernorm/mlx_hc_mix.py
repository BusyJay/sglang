# SPDX-License-Identifier: Apache-2.0
"""FP32 mHC RMSNorm/mix projection with native singleton reduction order."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tok = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.x * 4;
    threadgroup float sums[32];
    threadgroup float inverse;
    // Emulate the native RMS threads before the GEMV: contiguous groups of
    // four, looped at 4096 elements, then SIMD and SIMD-group reductions.
    for (uint wave = 0; wave < RMS_THREADS / 256; ++wave) {
        const uint virtual_tid = tid + wave * 256;
        float acc = 0;
        for (uint base = 0; base < K; base += RMS_THREADS * 4)
            for (uint i = 0; i < 4; ++i) {
                const uint k = base + virtual_tid * 4 + i;
                float value = k < K ? float(x[tok * K + k]) : 0;
                acc += value * value;
            }
        acc = simd_sum(acc);
        if (lane == 0) sums[wave * 8 + sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = lane < RMS_THREADS / 32 ? sums[lane] : 0;
        total = simd_sum(total);
        if (lane == 0) inverse = metal::precise::rsqrt(total / float(K) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result[4] = {0};
    // Native high-K / small-output GEMV: BN8, SN32, TN4, TM4.
    for (uint k = tid * 4; k < K; k += 1024) {
        float normalized[4];
        for (uint i = 0; i < 4; ++i)
            normalized[i] = float(x[tok * K + k + i]) * inverse;
        for (uint r = 0; r < 4; ++r)
            for (uint i = 0; i < 4; ++i)
                result[r] += float(weight[(row + r) * K + k + i]) * normalized[i];
    }
    threadgroup float partial[8 * 4];
    for (uint r = 0; r < 4; ++r) {
        for (ushort offset = 16; offset >= 1; offset >>= 1)
            result[r] += simd_shuffle_down(result[r], offset);
        if (lane == 0) partial[sg * 4 + r] = result[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        float total = partial[tid];
        for (uint group = 1; group < 8; ++group) total += partial[group * 4 + tid];
        mix[tok * 24 + row + tid] = total;
    }
"""


def _parallel_source():
    source = _SOURCE.replace(
        "const uint tid = thread_position_in_threadgroup.x;",
        "const uint tile = thread_position_in_threadgroup.x / 256;"
        "const uint tid = thread_position_in_threadgroup.x % 256;",
    ).replace(
        "const uint sg = simdgroup_index_in_threadgroup;",
        "const uint sg = simdgroup_index_in_threadgroup % 8;",
    ).replace(
        "threadgroup_position_in_grid.x * 4;",
        "(threadgroup_position_in_grid.x * 2 + tile) * 4;",
    ).replace("for (uint wave = 0;", "if (tile == 0) for (uint wave = 0;", 1)
    source = source.replace("if (sg == 0)", "if (tile == 0 && sg == 0)", 1)
    return (
        source.replace("partial[8 * 4]", "partial[2 * 8 * 4]")
        .replace("partial[sg * 4 + r]", "partial[tile * 32 + sg * 4 + r]")
        .replace("partial[tid]", "partial[tile * 32 + tid]")
        .replace("partial[group * 4 + tid]", "partial[tile * 32 + group * 4 + tid]")
    )


@cache
def _parallel_kernel():
    return mx.fast.metal_kernel(
        name="sglang_hc_parallel_rms_mix",
        input_names=["x", "weight", "eps"],
        output_names=["mix"],
        source=_parallel_source(),
    )


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_hc_rms_mix",
        input_names=["x", "weight", "eps"],
        output_names=["mix"],
        source=_SOURCE,
    )


def mlx_hc_rms_mix(x, weight, *, eps):
    """Four-stream [1,R,4,D] -> FP32 [1,R,24], preserving each row's GEMV.

    RMS reduction virtual threads follow MLX 0.32.4, including its looped
    kernel; each projection tile reuses original input without a FP32 copy or
    normalized intermediate. Changing token count only changes dispatch size.
    """
    if (
        x.ndim != 4
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or x.shape[2] != 4
        or not 256 <= x.shape[3] <= 4096
        or x.shape[3] % 256
        or x.dtype not in (mx.float16, mx.bfloat16)
        or weight.shape != (24, 4 * x.shape[3])
        or weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    ):
        return None
    count, k = x.shape[1], 4 * x.shape[3]
    parallel = (
        count >= 3
        and k == 16384
        and mx.device_info()["architecture"].startswith("applegpu_g17")
    )
    return (_parallel_kernel() if parallel else _kernel())(
        inputs=[x, weight, mx.array([eps], mx.float32)],
        template=[("K", k), ("RMS_THREADS", min(k // 4, 1024))],
        grid=(256 * 6, count, 1),
        threadgroup=(512 if parallel else 256, 1, 1),
        output_shapes=[(1, count, 24)],
        output_dtypes=[mx.float32],
    )[0]
