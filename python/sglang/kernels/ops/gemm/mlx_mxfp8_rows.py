# SPDX-License-Identifier: MIT
# Copyright © 2023 Apple Inc.
# Copyright © 2025-2026 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MXFP8 A16 projection with native wide-kernel accumulation order.

Each 32-value dot receives its original UE8M0 scale before accumulation.
The KL16 lane assignment and shuffle tree match native tile2/tile4 QMV.
Real vectors use 1..8 rows per dispatch without padded allocations; larger
batches share weights in four-row tiles. K, N and row count are runtime
parameters, so growing prefill/verification lengths do not specialize source.
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
// Same half-intermediate E4M3 decode as Apple's fp8_e4m3::operator float().
inline float sglang_e4m3(uchar bits) {
    ushort v = bits & 127;
    ushort sign_bit = ushort((bits >> 7) & 1) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return float(converted * 256.0);
}
// Preserve native code-0 subnormal and code-255 behavior too.
inline float sglang_ue8m0(uchar bits) {
    uint out = bits == 0 ? 0x400000u : (uint(bits) << 23);
    return as_type<float>(out);
}
"""

_SOURCE = r"""
const int K = dims[0], N = dims[1], M = dims[2];
constexpr int results_per_simdgroup = 32 / KL;
const uint lane = thread_index_in_simdgroup;
const uint sgid = simdgroup_index_in_threadgroup;
const int k_lane = lane % KL;
const int out_row = threadgroup_position_in_grid.y *
    (results_per_simdgroup * 2) + results_per_simdgroup * sgid + lane / KL;
const int vec0 = threadgroup_position_in_grid.x * TILE;
// Match native tail reads: clamp output row, suppress its final store.
const int row = min(out_row, N - 1);
const int groups = K / 32;
const device uchar* wrow = (const device uchar*)weight + size_t(row) * K;
const auto* srow = scales + size_t(row) * groups;
const device T* xv[TILE];
#pragma unroll
for (int v = 0; v < TILE; ++v) {
    xv[v] = x + size_t(min(vec0 + v, M - 1)) * K;
}
float result[TILE] = {0};
for (int g = k_lane; g < groups; g += KL) {
    const int k0 = g * 32;
    float s = sglang_ue8m0(srow[g]);
    const device uchar* wg = wrow + k0;
    float4 w4[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w4[i] = float4(
            sglang_e4m3(wg[4 * i]),
            sglang_e4m3(wg[4 * i + 1]),
            sglang_e4m3(wg[4 * i + 2]),
            sglang_e4m3(wg[4 * i + 3]));
    }
#pragma unroll
    for (int v = 0; v < TILE; ++v) {
        const device vec<T, 4>* xv4 = (const device vec<T, 4>*)(xv[v] + k0);
        float acc = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += dot(w4[j], float4(xv4[j]));
        }
        result[v] += s * acc;
    }
}
for (int v = 0; v < TILE; ++v) {
    result[v] += simd_shuffle_down(result[v], 8);
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
}
if (k_lane == 0 && out_row < N) {
    for (int v = 0; v < TILE; ++v) {
        if (vec0 + v < M) out[size_t(vec0 + v) * N + out_row] = T(result[v]);
    }
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_mlx_mxfp8_rows_wide_order",
        input_names=["x", "weight", "scales", "dims"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


def mlx_mxfp8_rows(x, weight, scales, *, tilegroups=None):
    """Project BF16/F32 vectors with unchanged MXFP8 operands.

    ``tilegroups`` is vectors per threadgroup (1..8), not SIMD-group count;
    every threadgroup has two SIMD groups / 64 threads. Arbitrary positive
    vector counts use ceil(count/tilegroups) tiles without allocating padding.
    Weights can be uint8 [N,K] or native packed uint32 [N,K/4]. Output shape
    and dtype follow x. K/N/count are runtime parameters, not JIT templates.
    """
    kl = 16
    if x.ndim < 1 or x.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("Expected BF16 or F32 activation vectors")
    if weight.ndim != 2 or weight.dtype not in (mx.uint8, mx.uint32):
        raise ValueError("Expected uint8 or native packed uint32 MXFP8 weights")
    n = weight.shape[0]
    k = weight.shape[1] * (4 if weight.dtype == mx.uint32 else 1)
    if n <= 0 or k <= 0 or k % 32 or x.shape[-1] != k:
        raise ValueError("Expected nonempty [N,K] weights with K divisible by 32")
    if scales.dtype != mx.uint8 or scales.shape != (n, k // 32):
        raise ValueError("Expected uint8 UE8M0 scales [N,K/32]")
    count = x.size // k
    if count <= 0 or max(k, n, count) >= 2**31:
        raise ValueError("Expected positive dimensions fitting native int parameters")
    if tilegroups is None:
        tilegroups = min(count, 8)
    if tilegroups not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("tilegroups must be 1..8")
    rows_per_tg = 64 // kl
    return _kernel()(
        inputs=[x.reshape(count, k), weight, scales, mx.array([k, n, count], mx.int32)],
        template=[("T", x.dtype), ("TILE", tilegroups), ("KL", kl)],
        # grid is THREADS, whereas the native dispatch grid is THREADGROUPS.
        grid=(
            32 * ((count + tilegroups - 1) // tilegroups),
            2 * ((n + rows_per_tg - 1) // rows_per_tg),
            1,
        ),
        threadgroup=(32, 2, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*x.shape[:-1], n)
