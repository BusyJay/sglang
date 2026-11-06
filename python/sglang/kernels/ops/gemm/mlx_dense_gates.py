# SPDX-License-Identifier: Apache-2.0
"""Independent dense gate banks with native singleton GEMV reductions."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const int K = dims[0], COUNT = dims[1];
    const uint bank = threadgroup_position_in_grid.y;
    const int N = dims[2 + bank];
    const device T* x = X_POINTER;
    const device W* w = W_POINTER;
    device O* out = O_POINTER;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    // Native GEMV uses BN=8 for K >= 16*N, otherwise BN=1.
    const int bn = K >= 16 * N ? 8 : 1;
    const int row = (int(threadgroup_position_in_grid.x) * (8 / bn)
                     + int(sg) / bn) * 4;
    if (row >= N) return;
    const int start = int(threadgroup_position_in_grid.z) * ROWS;
    float acc[ROWS][4] = {0};
    for (int k = (int(sg) % bn * 32 + int(lane)) * 4;
         k < K; k += bn * 128) {
        float inputs[ROWS][4];
        for (int t = 0; t < ROWS; ++t)
            for (int i = 0; i < 4; ++i)
                inputs[t][i] = start+t < COUNT ? float(x[(start+t)*K+k+i]) : 0.0f;
        for (int r = 0; r < 4; ++r) {
            for (int i = 0; i < 4; ++i) {
                const float value = float(w[size_t(row+r)*K+k+i]);
                for (int t = 0; t < ROWS; ++t)
                    acc[t][r] += value * inputs[t][i];
            }
        }
    }
    for (int t = 0; t < ROWS; ++t)
        for (int r = 0; r < 4; ++r)
            for (ushort off = 16; off >= 1; off >>= 1)
                acc[t][r] += simd_shuffle_down(acc[t][r], off);
    threadgroup float partials[ROWS][8][4];
    if (bn == 8) {
        if (lane == 0)
            for (int t = 0; t < ROWS; ++t)
                for (int r = 0; r < 4; ++r)
                    partials[t][sg][r] = acc[t][r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0)
            for (int t = 0; t < ROWS; ++t)
                for (int s = 1; s < 8; ++s)
                    for (int r = 0; r < 4; ++r)
                        acc[t][r] += partials[t][s][r];
    }
    if (lane == 0 && (bn == 1 || sg == 0))
        for (int t = 0; t < ROWS; ++t)
            if (start+t < COUNT)
                for (int r = 0; r < 4; ++r)
                    out[(start+t)*N+row+r] = O(acc[t][r]);
"""


@cache
def _kernel(banks, shared):
    def pointer(prefix):
        return " : ".join(
            [f"bank == {i} ? {prefix}{i}" for i in range(banks - 1)]
            + [f"{prefix}{banks - 1}"]
        )

    source = (
        _SOURCE.replace("X_POINTER", "x0" if shared else pointer("x"))
        .replace("W_POINTER", pointer("w"))
        .replace("O_POINTER", pointer("out"))
    )
    return mx.fast.metal_kernel(
        name=f"sglang_dense_gates_{banks}_{int(shared)}",
        input_names=[f"x{i}" for i in range(1 if shared else banks)]
        + [f"w{i}" for i in range(banks)]
        + ["dims"],
        output_names=[f"out{i}" for i in range(banks)],
        source=source,
    )


def _project(inputs, weights, shared):
    x = inputs[0]
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or any(v.shape != x.shape or v.dtype != x.dtype for v in inputs)
        or any(w.ndim != 2 for w in weights)
    ):
        return None
    count, k = x.shape[1:]
    sizes = [w.shape[0] for w in weights]
    if (
        k <= 64
        or k % 128
        or any(n < 4 or n % 4 for n in sizes)
        or weights[0].dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or any(w.shape[1] != k or w.dtype != weights[0].dtype for w in weights)
    ):
        return None
    rows = min(count, 4)
    groups = max(
        (n + (4 if k >= 16 * n else 32) - 1) // (4 if k >= 16 * n else 32)
        for n in sizes
    )
    dtype = mx.result_type(x.dtype, weights[0].dtype)
    return tuple(
        _kernel(len(weights), shared)(
            inputs=[*inputs, *weights, mx.array([k, count, *sizes], mx.int32)],
            template=[
                ("T", x.dtype),
                ("W", weights[0].dtype),
                ("O", dtype),
                ("ROWS", rows),
            ],
            grid=(256 * groups, len(weights), (count + rows - 1) // rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, count, n) for n in sizes],
            output_dtypes=[dtype] * len(weights),
        )
    )


def mlx_dense_gate_rows(x, weights):
    """Three independent weights, shared real token rows, contiguous outputs.

    Preserve native TN=4 lane sums, shuffle-down, optional BN=8 reduction,
    and the output cast. No concatenated weights or padded token allocations.
    Shape data stays in runtime operands; only the row tile is specialized.
    Returns None for layouts outside this aligned GEMV implementation.
    """
    if len(weights) != 3:
        return None
    return _project((x,), weights, True)


def mlx_dense_gate_pair(inputs, weights):
    """Two independent input/weight pairs in one dispatch, separate outputs."""
    if len(inputs) != 2 or len(weights) != 2:
        return None
    return _project(inputs, weights, False)
