"""Four-stream mHC Sinkhorn, collapse/RMSNorm and residual mixing for MLX."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup float pre[4];
    threadgroup float comb[16];
    threadgroup float partial[32];
    if (tid == 0) {
        for (int i = 0; i < 4; i++) {
            volatile float pre_product = mix[i] * scale[0];
            float v = pre_product + base[i];
            float e = 1 / (1 + metal::precise::exp(metal::abs(v)));
            pre[i] = (v < 0 ? e : 1 - e) + params[0];
            volatile float post_product = mix[4 + i] * scale[1];
            v = post_product + base[4 + i];
            e = 1 / (1 + metal::precise::exp(metal::abs(v)));
            post[i] = T(2 * (v < 0 ? e : 1 - e));
        }
        for (int r = 0; r < 4; r++) {
            float maximum = -INFINITY;
            for (int c = 0; c < 4; c++) {
                const int i = 4 * r + c;
                volatile float comb_product = mix[8 + i] * scale[2];
                comb[i] = comb_product + base[8 + i];
                maximum = max(maximum, comb[i]);
            }
            float total = 0;
            for (int c = 0; c < 4; c++) {
                comb[4 * r + c] = fast::exp(comb[4 * r + c] - maximum);
                total += comb[4 * r + c];
            }
            for (int c = 0; c < 4; c++) {
                volatile float value = comb[4 * r + c] * (1.0f / total);
                comb[4 * r + c] = value + params[0];
            }
        }
        // Initial column normalization, then alternating rows and columns.
        for (int it = 0; it < 2 * ITERS - 1; it++) {
            bool column = (it % 2) == 0;
            for (int r = 0; r < 4; r++) {
                float total = 0;
                for (int c = 0; c < 4; c++) {
                    total += comb[column ? 4 * c + r : 4 * r + c];
                }
                for (int c = 0; c < 4; c++) {
                    comb[column ? 4 * c + r : 4 * r + c] /= total + params[0];
                }
            }
        }
    }
    if (tid == 0) {
        for (int i = 0; i < 16; i++) comb_out[i] = T(comb[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float squares = 0;
    for (int d = int(tid) * 4; d < int(tid) * 4 + 4; d++) {
        float collapsed = 0;
        for (int h = 0; h < 4; h++) {
            volatile float product = pre[h] * float(x[h * D + d]);
            collapsed += product;
        }
        T rounded = T(collapsed);
        normed[d] = rounded;
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
        T normalized = T(float(normed[d]) * partial[0]);
        normed[d] = normalized * norm_weight[d];
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

    Only one-token decode is covered. The projection remains an ordinary MLX
    operation so its matmul precision follows the caller's runtime setting.
    """
    if (
        x.ndim != 4
        or x.shape[:3] != (1, 1, 4)
        or x.dtype not in (mx.bfloat16, mx.float16)
        or not 128 <= x.shape[-1] <= 4096
        or x.shape[-1] % 128
        or not 1 <= iters <= 32
        or mix.size != 24
        or base.shape != (24,)
        or scale.shape != (3,)
        or mix.dtype != mx.float32
        or base.dtype != mx.float32
        or scale.dtype != mx.float32
        or norm_weight.shape != (x.shape[-1],)
        or norm_weight.dtype != x.dtype
    ):
        return None
    dim = x.shape[-1]
    normed, post, comb = _kernel()(
        inputs=[
            x,
            mix,
            base,
            scale,
            norm_weight,
            mx.array([eps, norm_eps], mx.float32),
        ],
        template=[("T", x.dtype), ("D", dim), ("ITERS", iters), ("THREADS", dim // 4)],
        grid=(dim // 4, 1, 1),
        threadgroup=(dim // 4, 1, 1),
        output_shapes=[(1, 1, dim), (1, 1, 4), (1, 1, 4, 4)],
        output_dtypes=[x.dtype] * 3,
    )

    # Preserve MLX's hardware matmul reduction and rounding. An equivalent
    # scalar four-term dot can change BF16 residuals near cancellation.
    return normed, post, comb.swapaxes(-1, -2) @ x
