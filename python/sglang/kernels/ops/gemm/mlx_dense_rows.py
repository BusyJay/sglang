# SPDX-License-Identifier: Apache-2.0
# Accumulation order follows Apple's MLX GEMVKernel (gemv.h), v0.32.4.
"""Dense multi-vector GEMV sharing matrix loads with singleton row arithmetic."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row = (int(threadgroup_position_in_grid.x) * BM + int(sg)) * 4;
    if (row >= N) return;
    float acc[ROWS][4];
    for (int t = 0; t < ROWS; ++t)
        for (int r = 0; r < 4; ++r) acc[t][r] = 0.0f;
    // Native singleton GEMV uses TN=4, SN=32: identical K slices per lane.
    for (int k = int(lane) * 4; k < K; k += 128) {
        float inputs[ROWS][4];
        for (int t = 0; t < ROWS; ++t)
            for (int i = 0; i < 4; ++i)
                inputs[t][i] = float(x[t * K + k + i]);
        for (int r = 0; r < 4; ++r) {
            float values[4];
            for (int i = 0; i < 4; ++i)
                values[i] = row + r < N ? float(w[size_t(row + r) * K + k + i]) : 0.0f;
            // The same matrix values feed all token accumulators.
            for (int t = 0; t < ROWS; ++t)
                for (int i = 0; i < 4; ++i)
                    acc[t][r] += values[i] * inputs[t][i];
        }
    }
    for (int t = 0; t < ROWS; ++t) {
        for (int r = 0; r < 4; ++r) {
            for (ushort off = 16; off >= 1; off >>= 1)
                acc[t][r] += simd_shuffle_down(acc[t][r], off);
            if (lane == 0 && row + r < N) out[t * N + row + r] = T(acc[t][r]);
        }
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_dense_shared_rows",
        input_names=["x", "w"],
        output_names=["out"],
        source=_SOURCE,
    )


def mlx_dense_rows(x, weight):
    """Shared-weight GEMV for one sequence, retaining native row reductions.

    Row-major [N,K] weights; aligned native SN=32/TN=4 GEMV geometries.
    Returns None for layouts requiring a different reduction/tail kernel.
    Longer chains use tiles of up to four vectors to bound register pressure.
    """
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or weight.ndim != 2
        or x.dtype not in (mx.float16, mx.bfloat16)
        or weight.dtype != x.dtype
    ):
        return None
    count, dim = x.shape[1:]
    out_dim, width = weight.shape
    if width != dim or dim % 128 or out_dim < 4 or dim <= 64 or dim >= 16 * out_dim:
        return None
    bm = 8 if out_dim >= 4096 else 4
    outputs = []
    for start in range(0, count, 4):
        rows = min(count - start, 4)
        outputs.append(
            _kernel()(
                inputs=[x[:, start : start + rows], weight],
                template=[
                    ("T", x.dtype),
                    ("K", dim),
                    ("N", out_dim),
                    ("BM", bm),
                    ("ROWS", rows),
                ],
                grid=(32 * bm * ((out_dim + 4 * bm - 1) // (4 * bm)), 1, 1),
                threadgroup=(32 * bm, 1, 1),
                output_shapes=[(1, rows, out_dim)],
                output_dtypes=[x.dtype],
            )[0]
        )
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
