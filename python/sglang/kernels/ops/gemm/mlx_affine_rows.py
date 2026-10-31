# SPDX-License-Identifier: Apache-2.0
# Reuses the native MLX QMV arithmetic transcribed in ops.moe.mlx_glm.
"""Bounded affine matrix rows with the singleton MLX QMV reduction order."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.moe.mlx_glm import _HEADER

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * 4;
    const int pick = int(threadgroup_position_in_grid.y);
    const int tok = pick / TOPK;
    const size_t base = size_t(indices[pick]) * N + row;
    float value[4] = {0};
    glm_qmv_rows<T, K, GS, BITS, 4>(
        (const device uint8_t*)w + base * (K * BITS / 8),
        s + base * (K / GS), b + base * (K / GS),
        x + size_t(PER_EXPERT ? pick : tok) * K, lane, value);
    for (int j = 0; j < 4; j++) {
        T result = T(simd_sum(value[j]));
        if (lane == 0) out[size_t(pick) * N + row + j] = result;
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_affine_qmv_rows",
        input_names=["x", "w", "s", "b", "indices"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


def mlx_affine_rows(x, weight, scales, biases, *, bits, group_size, indices=None):
    """1..17 rows, optional expert indices; return None outside covered tiles.

    Dense output is [1, rows, N]; expert output is [1, rows, topk, N].
    Each SIMD group retains the one-row QMV accumulation and rounding.
    """
    routed = indices is not None
    if x.ndim not in (3, 4) or x.shape[0] != 1 or not 1 <= x.shape[1] <= 17:
        return None
    per_expert = x.ndim == 4
    count, dim = x.shape[1], x.shape[-1]
    topk = indices.shape[-1] if routed else 1
    if (
        (not routed and per_expert)
        or (routed and (indices.ndim != 3 or indices.shape[:2] != (1, count)))
        or (per_expert and x.shape[2] != topk)
        or not 1 <= topk <= 32
        or bits not in (4, 5, 6, 8)
        or group_size not in (32, 64, 128)
        or x.dtype not in (mx.float16, mx.bfloat16)
        or weight.ndim != (3 if routed else 2)
        or weight.dtype != mx.uint32
        or scales is None
        or biases is None
        or scales.shape != biases.shape
        or scales.dtype != x.dtype
        or biases.dtype != x.dtype
    ):
        return None
    pack = 8 if bits == 5 else 4 if bits == 6 else 32 // bits
    out_dim, packed = weight.shape[-2:]
    if (
        packed * 32 != dim * bits
        or dim % (32 * pack * 2)
        or dim % group_size
        or out_dim % 16
        or scales.shape != weight.shape[:-1] + (dim // group_size,)
        or (routed and indices.dtype not in (mx.uint32, mx.int32))
    ):
        return None
    if not routed:
        weight, scales, biases = [
            mx.expand_dims(v, 0) for v in (weight, scales, biases)
        ]
        indices = mx.zeros((1, count, 1), mx.uint32)
    out = _kernel()(
        inputs=[x, weight, scales, biases, indices],
        template=[
            ("T", x.dtype),
            ("K", dim),
            ("N", out_dim),
            ("GS", group_size),
            ("BITS", bits),
            ("TOPK", topk),
            ("PER_EXPERT", per_expert),
        ],
        grid=(128 * (out_dim // 16), count * topk, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, count, topk, out_dim)],
        output_dtypes=[x.dtype],
    )[0]
    return out if routed else out.squeeze(2)
