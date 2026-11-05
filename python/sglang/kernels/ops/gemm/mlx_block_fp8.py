# SPDX-License-Identifier: Apache-2.0
"""Paired block-FP8 projections share input loads and preserve reductions."""

from functools import cache

import mlx.core as mx

_DECODE = r"""
inline float mlx_e4m3(uchar code) {
    ushort magnitude = code & 127;
    ushort sign_bit = ushort((code >> 7) & 1) << 15;
    ushort bits = (magnitude << 7) | (((magnitude + 1) >> 7) << 14) | sign_bit;
    half value = as_type<half>(bits);
    return float(value * 256.0);
}
"""

_SOURCE = r"""
const uint K = dims[0], N = dims[1], COUNT = dims[2];
uint lane = thread_index_in_simdgroup;
uint row = (threadgroup_position_in_grid.x * 4 +
    simdgroup_index_in_threadgroup) * RPS;
uint start = threadgroup_position_in_grid.z * ROWS;
if (row >= N) return;
float g[ROWS][RPS] = {0}, u[ROWS][RPS] = {0};
for (uint block = 0; block < K / 128; ++block) {
    float gs[RPS], us[RPS];
    for (uint r = 0; r < RPS; ++r) {
        gs[r] = row + r < N ? gate_scales[((row+r)/128)*(K/128)+block] : 0;
        us[r] = row + r < N ? up_scales[((row+r)/128)*(K/128)+block] : 0;
    }
    for (uint step = 0; step < 4; ++step) {
        uint k = block * 128 + step * 32 + lane;
        float inputs[ROWS];
        for (uint t = 0; t < ROWS; ++t) {
            inputs[t] = start + t < COUNT ?
                mlx_e4m3(xq[(start+t)*K+k]) * xscale[(start+t)*(K/32)+k/32] : 0;
        }
        for (uint r = 0; r < RPS; ++r) {
            if (row + r >= N) continue;
            float gv = mlx_e4m3(gate_weight[size_t(row+r)*K+k]) * gs[r];
            float uv = mlx_e4m3(up_weight[size_t(row+r)*K+k]) * us[r];
            for (uint t = 0; t < ROWS; ++t) {
                g[t][r] += inputs[t] * gv;
                u[t][r] += inputs[t] * uv;
            }
        }
    }
}
for (uint t = 0; t < ROWS; ++t) {
    for (uint r = 0; r < RPS; ++r) {
        for (ushort off = 16; off >= 1; off >>= 1) {
            g[t][r] += simd_shuffle_down(g[t][r], off);
            u[t][r] += simd_shuffle_down(u[t][r], off);
        }
        if (lane == 0 && start+t < COUNT && row+r < N) {
            gate[(start+t)*N+row+r] = O(g[t][r]);
            up[(start+t)*N+row+r] = O(u[t][r]);
        }
    }
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_mlx_block_fp8_pair",
        input_names=[
            "gate_weight",
            "gate_scales",
            "up_weight",
            "up_scales",
            "xq",
            "xscale",
            "dims",
        ],
        output_names=["gate", "up"],
        source=_SOURCE,
        header=_DECODE,
    )


def mlx_block_fp8_pair(
    x, gate_weight, gate_scales, up_weight, up_scales, quantized_input
):
    """Project the same quantized input through two /128 block-FP8 matrices.

    ``x`` supplies the output shape/dtype. Its codes/scales are passed explicitly,
    so callers can retain their activation quantization policy. Intermediate
    gate/up values round separately to x.dtype, before any activation fusion.
    """
    return _pair(x, gate_weight, gate_scales, up_weight, up_scales, quantized_input)


def _pair(x, gw, gs, uw, us, quantized, *, output_rows=2):
    if x.ndim < 1 or x.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("Expected BF16 or FP32 activation rows")
    if (
        gw.ndim != 2
        or uw.shape != gw.shape
        or gw.dtype != mx.uint8
        or uw.dtype != mx.uint8
    ):
        raise ValueError("Expected equal uint8 gate/up matrices")
    n, k = gw.shape
    count = x.size // k if k else 0
    if not n or not k or k % 128 or x.shape[-1] != k or not count:
        raise ValueError("Expected nonempty matrices with input width divisible by128")
    # Serialized ModelOpt scales retain singleton axes; the native
    # projector consumes their row-major flattened /128 block layout.
    scale_size = ((n + 127) // 128) * (k // 128)
    if any(s.dtype != mx.float32 or s.size != scale_size for s in (gs, us)):
        raise ValueError("Expected FP32 /128x128 block scales")
    xq, xs = quantized
    if (
        xq.dtype != mx.uint8
        or xq.shape != (count, k)
        or xs.dtype != mx.float32
        or xs.shape != (count, k // 32)
    ):
        raise ValueError("Expected explicit /32 input codes and scales")
    if output_rows not in (1, 2, 4) or max(n * k, count * k, count * n) >= 2**32:
        raise ValueError("Expected supported output tile and uint32 indexing")
    rows = min(count, 4)
    result = _kernel()(
        inputs=[
            gw,
            gs.reshape(-1),
            uw,
            us.reshape(-1),
            xq,
            xs,
            mx.array([k, n, count], mx.uint32),
        ],
        template=[("O", x.dtype), ("RPS", output_rows), ("ROWS", rows)],
        grid=(
            ((n + 4 * output_rows - 1) // (4 * output_rows)) * 128,
            1,
            (count + rows - 1) // rows,
        ),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, n), (count, n)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return tuple(v.reshape(*x.shape[:-1], n) for v in result)
