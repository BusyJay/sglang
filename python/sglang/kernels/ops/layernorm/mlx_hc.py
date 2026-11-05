"""Four-stream mHC Sinkhorn, collapse/RMSNorm and residual mixing for MLX."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint tok = threadgroup_position_in_grid.y;
    threadgroup float pre[4];
    threadgroup float partial[32];
    // The first SIMD group owns the coefficient work. All its lanes execute
    // shuffles; lanes 0..3 publish the four independent rows/columns.
    if (tid < 32) {
        const int r = int(tid % 4);
        if (tid < 4) {
            const int i = r;
            volatile float pre_product = mix[tok * 24 + i] * scale[0];
            float v = pre_product + base[i];
            float e = 1 / (1 + metal::precise::exp(metal::abs(v)));
            pre[i] = (v < 0 ? e : 1 - e) + params[0];
            volatile float post_product = mix[tok * 24 + 4 + i] * scale[1];
            v = post_product + base[4 + i];
            e = 1 / (1 + metal::precise::exp(metal::abs(v)));
            post[tok * 4 + i] = T(2 * (v < 0 ? e : 1 - e));
        }
        float values[4] = {0, 0, 0, 0};
        if (tid < 4) {
            float maximum = -INFINITY;
            for (int c = 0; c < 4; c++) {
                const int i = 4 * r + c;
                volatile float comb_product = mix[tok * 24 + 8 + i] * scale[2];
                values[c] = comb_product + base[8 + i];
                maximum = max(maximum, values[c]);
            }
            float total = 0;
            for (int c = 0; c < 4; c++) {
                values[c] = fast::exp(values[c] - maximum);
                total += values[c];
            }
            for (int c = 0; c < 4; c++) {
                volatile float value = values[c] * (1.0f / total);
                values[c] = value + params[0];
            }
        }
        // Initial column normalization, then alternating rows and columns.
        for (int it = 0; it < 2 * ITERS - 1; it++) {
            // Transpose ownership without changing a line's summation order.
            // Each source lane c supplies its element r. Snapshot the row
            // before overwriting it, including the diagonal elements.
            const float4 row = float4(values[0], values[1], values[2], values[3]);
            for (int c = 0; c < 4; c++) {
                values[c] = simd_shuffle(row, ushort(c))[r];
            }
            float total = 0;
            for (int c = 0; c < 4; c++) {
                total += values[c];
            }
            for (int c = 0; c < 4; c++) {
                values[c] /= total + params[0];
            }
        }
        // The odd number of stages always finishes with column ownership.
        if (tid < 4) {
            for (int c = 0; c < 4; c++) {
                comb_out[tok * 16 + 4 * c + r] = T(values[c]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float squares = 0;
    for (int d = int(tid) * 4; d < int(tid) * 4 + 4; d++) {
        float collapsed = 0;
        for (int h = 0; h < 4; h++) {
            volatile float product = pre[h] * float(x[(tok * 4 + h) * D + d]);
            collapsed += product;
        }
        T rounded = T(collapsed);
        normed[tok * D + d] = rounded;
        squares += float(rounded) * float(rounded);

    }
    float sum = simd_sum(squares);
    if (thread_index_in_simdgroup == 0) {
        partial[simdgroup_index_in_threadgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = tid < THREADS / 32 ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) partial[0] = metal::precise::rsqrt(v / float(D) + params[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = int(tid) * 4; d < int(tid) * 4 + 4; d++) {
        // MLX rounds normalization before applying its learned 16-bit weight.
        O normalized = O(float(normed[tok * D + d]) * partial[0]);
        normed[tok * D + d] = normalized * norm_weight[d];
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_hc_collapse_norm",
        input_names=["x", "mix", "base", "scale", "norm_weight", "params"],
        output_names=["normed", "post", "comb_out"],
        source=_SOURCE,
    )


def mlx_hc_collapse_norm(x, mix, base, scale, norm_weight, *, eps, norm_eps, iters):
    """Fuse four-stream mHC after its FP32 mix projection; else return None.

    One sequence with 1..17 token rows is covered. The projection remains an ordinary MLX
    operation so its matmul precision follows the caller's runtime setting.
    """
    if (
        x.ndim != 4
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or x.shape[2] != 4
        or x.dtype not in (mx.bfloat16, mx.float16)
        or not 128 <= x.shape[-1] <= 4096
        or x.shape[-1] % 128
        or not 1 <= iters <= 32
        or mix.size != x.shape[1] * 24
        or base.shape != (24,)
        or scale.shape != (3,)
        or mix.dtype != mx.float32
        or base.dtype != mx.float32
        or scale.dtype != mx.float32
        or norm_weight.shape != (x.shape[-1],)
        or norm_weight.dtype not in (x.dtype, mx.float32)
    ):
        return None
    count, dim = x.shape[1], x.shape[-1]
    normed, post, comb = _kernel()(
        inputs=[
            x,
            mix,
            base,
            scale,
            norm_weight,
            mx.array([eps, norm_eps], mx.float32),
        ],
        template=[("T", x.dtype), ("O", mx.result_type(x.dtype, norm_weight.dtype)), ("D", dim), ("ITERS", iters), ("THREADS", dim // 4)],
        grid=(dim // 4, count, 1),
        threadgroup=(dim // 4, 1, 1),
        output_shapes=[(1, count, dim), (1, count, 4), (1, count, 4, 4)],
        output_dtypes=[mx.result_type(x.dtype, norm_weight.dtype), x.dtype, x.dtype],
    )

    # Preserve MLX's hardware matmul reduction and rounding. An equivalent
    # scalar four-term dot can change BF16 residuals near cancellation.
    residual = comb.swapaxes(-1, -2) @ x
    return normed, post, residual
