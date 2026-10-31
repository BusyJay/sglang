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
            if (lane == 0 && row + r < N) out[t * N + row + r] = O(acc[t][r]);
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
        or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
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
                    ("O", mx.result_type(x.dtype, weight.dtype)),
                    ("K", dim),
                    ("N", out_dim),
                    ("BM", bm),
                    ("ROWS", rows),
                ],
                grid=(32 * bm * ((out_dim + 4 * bm - 1) // (4 * bm)), 1, 1),
                threadgroup=(32 * bm, 1, 1),
                output_shapes=[(1, rows, out_dim)],
                output_dtypes=[mx.result_type(x.dtype, weight.dtype)],
            )[0]
        )
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)


@cache
def _argmax_kernel():
    source = _SOURCE.replace("if (row >= N) return;", "")
    epilogue = (
        "    for (int t = 0; t < ROWS; ++t) {\n        for (int r = 0; r < 4; ++r) {"
    )
    source = (
        source[: source.index(epilogue)]
        + r"""
    float best = -INFINITY;
    uint index = uint(N);
    for (int r = 0; r < 4; ++r) {
        for (ushort off = 16; off >= 1; off >>= 1)
            acc[0][r] += simd_shuffle_down(acc[0][r], off);
        float value = float(T(acc[0][r]));
        if (row + r < N && (value > best || (value == best && uint(row + r) < index))) {
            best = value;
            index = uint(row + r);
        }
    }
    threadgroup float scores[BM];
    threadgroup uint ids[BM];
    if (lane == 0) { scores[sg] = best; ids[sg] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int i = 0; i < BM; ++i) {
            if (scores[i] > best || (scores[i] == best && ids[i] < index)) {
                best = scores[i]; index = ids[i];
            }
        }
        block_scores[threadgroup_position_in_grid.x] = T(best);
        block_ids[threadgroup_position_in_grid.x] = index;
    }
"""
    )
    return mx.fast.metal_kernel(
        name="sglang_dense_greedy",
        input_names=["x", "w"],
        output_names=["block_scores", "block_ids"],
        source=source,
    )


def mlx_dense_argmax(x, weight):
    """Greedy dense GEMV with rounded-logit top-1 epilogue and stable ties.

    Only block winners are materialized; checkpoint weights and native GEMV
    accumulation remain unchanged. Covers finite singleton FP16/BF16 logits.
    """
    if (
        x.ndim != 3
        or x.shape[:2] != (1, 1)
        or weight.ndim != 2
        or x.dtype not in (mx.float16, mx.bfloat16)
        or weight.dtype != x.dtype
    ):
        return None
    n, k = weight.shape
    if x.shape[-1] != k or k % 128 or n < 4 or k >= 16 * n:
        return None
    bm = 8 if n >= 4096 else 4
    blocks = (n + 4 * bm - 1) // (4 * bm)
    scores, ids = _argmax_kernel()(
        inputs=[x, weight],
        template=[("T", x.dtype), ("K", k), ("N", n), ("BM", bm), ("ROWS", 1)],
        grid=(32 * bm * blocks, 1, 1),
        threadgroup=(32 * bm, 1, 1),
        output_shapes=[(blocks,), (blocks,)],
        output_dtypes=[x.dtype, mx.uint32],
    )
    return mx.take(ids, mx.argmax(scores)).reshape(1, 1)


_HEAD_TRANSPOSE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint start = threadgroup_position_in_grid.z * ROWS;
    const uint col = threadgroup_position_in_grid.x * (BN * 16) + sg * 16 + (lane % 4) * 4;
    float acc[ROWS][4] = {0};
    for (uint k = (lane / 4) * 4; k < K; k += 32) {
        for (uint j = 0; j < 4; ++j) {
            float values[4];
            for (uint i = 0; i < 4; ++i)
                values[i] = float(w[(size_t(h) * K + k + j) * N + col + i]);
            for (uint t = 0; t < ROWS; ++t) {
                const float input = float(x[(size_t(h) * COUNT + start + t) * K + k + j]);
                for (uint i = 0; i < 4; ++i) acc[t][i] += input * values[i];
            }
        }
    }
    for (uint t = 0; t < ROWS; ++t)
        for (uint i = 0; i < 4; ++i) {
            for (ushort off = 4; off >= 1; off >>= 1)
                acc[t][i] += simd_shuffle_down(acc[t][i], 4 * off);
            if (lane < 4)
                out[(size_t(h) * COUNT + start + t) * N + col + i] = O(acc[t][i]);
        }
"""


@cache
def _head_kernel(transposed):
    source = (
        _HEAD_TRANSPOSE
        if transposed
        else _SOURCE.replace(
            "    const uint lane",
            "    const uint h = threadgroup_position_in_grid.y;\n"
            "    const uint start = threadgroup_position_in_grid.z * ROWS;\n"
            "    const uint lane",
            1,
        )
        .replace("x[t * K + k + i]", "x[(size_t(h) * COUNT + start + t) * K + k + i]")
        .replace(
            "w[size_t(row + r) * K + k + i]", "w[(size_t(h) * N + row + r) * K + k + i]"
        )
        .replace(
            "out[t * N + row + r]", "out[(size_t(h) * COUNT + start + t) * N + row + r]"
        )
    )
    return mx.fast.metal_kernel(
        name="sglang_headed_gemvt_rows" if transposed else "sglang_headed_gemv_rows",
        input_names=["x", "w"],
        output_names=["out"],
        source=source,
    )


def mlx_headed_dense_rows(x, weight, *, transposed):
    """Batch singleton latent GEMV across heads and tokens, reusing weights.

    X [1,H,R,K]; weights [H,K,N] for GEMVT, [H,N,K] for GEMV.
    The covered native GEMVT geometry uses SM=8/SN=4/TM=TN=4.
    No per-token Python matmul or change to the native reduction boundary.
    """
    if x.ndim != 4 or x.shape[0] != 1 or weight.ndim != 3:
        return None
    _, heads, count, k = x.shape
    n = weight.shape[-1] if transposed else weight.shape[-2]
    if (
        weight.shape[0] != heads
        or not 1 <= count <= 17
        or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or weight.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    ):
        return None
    if transposed:
        if weight.shape[1] != k or k % 32 or n < 16 or n % 16 or n >= 2048:
            return None
        bn, bm = (4 if n >= 512 else 2), 1
        if n % (bn * 16):
            return None
        groups, threads = n // (bn * 16), bn * 32
    else:
        if weight.shape[2] != k or k % 128 or n < 4 or k >= 16 * n:
            return None
        bm, bn = (8 if n >= 4096 else 4), 1
        groups, threads = (n + 4 * bm - 1) // (4 * bm), bm * 32
    rows = 3 if count % 3 == 0 else 1
    return _head_kernel(transposed)(
        inputs=[x, weight],
        template=[
            ("O", mx.result_type(x.dtype, weight.dtype)),
            ("K", k),
            ("N", n),
            ("COUNT", count),
            ("ROWS", rows),
            ("BM", bm),
            ("BN", bn),
        ],
        grid=(threads * groups, heads, count // rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(1, heads, count, n)],
        output_dtypes=[mx.result_type(x.dtype, weight.dtype)],
    )[0]
