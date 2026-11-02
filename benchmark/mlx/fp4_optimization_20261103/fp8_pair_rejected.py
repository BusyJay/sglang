# SPDX-License-Identifier: Apache-2.0
"""Packed E4M3 projections with block scales and shared verification rows."""

from functools import cache

import mlx.core as mx

_DECODE = r"""
inline float e4m3(uchar code) {
    uint mag = uint(code) & 127;
    if (mag == 127) return NAN;
    uint exp = mag >> 3, mant = mag & 7;
    float v = exp == 0 ? float(mant) * 0.001953125f : as_type<float>(((exp + 120) << 23) | (mant << 20));
    return code & 128 ? -v : v;
}
"""
_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint row = (threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup) * RPS;
uint start = threadgroup_position_in_grid.z * ROWS;
if (row >= N) return;
float acc[ROWS][RPS] = {0};
for (uint k = lane; k < K; k += 32) {
    float inputs[ROWS];
    for (uint t = 0; t < ROWS; ++t) {
        inputs[t] = start + t < COUNT ? (A8 ? e4m3(xq[(start + t) * K + k]) * xscale[(start + t) * (K / 32) + k / 32] : float(x[(start + t) * K + k])) : 0.0f;
    }
    for (uint r = 0; r < RPS; ++r) {
        if (row + r >= N) continue;
        uint wb = k / BLOCK;
        float ws = MX ? ldexp(1.0f, int(scales[(row + r) * (K / BLOCK) + wb]) - 127) : block_scales[((row + r) / 128) * (K / BLOCK) + wb];
        float wv = e4m3(weight[size_t(row + r) * K + k]) * ws;
        for (uint t = 0; t < ROWS; ++t) acc[t][r] += inputs[t] * wv;
    }
}
for (uint t = 0; t < ROWS; ++t) for (uint r = 0; r < RPS; ++r) {
    for (ushort off = 16; off >= 1; off >>= 1) acc[t][r] += simd_shuffle_down(acc[t][r], off);
    if (lane == 0 && start + t < COUNT && row + r < N) out[(start + t) * N + row + r] = O(acc[t][r]);
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_fp8_shared_rows",
        input_names=["x", "weight", "scales", "block_scales", "xq", "xscale"],
        output_names=["out"],
        source=_SOURCE,
        header=_DECODE,
    )


def mlx_fp8_rows(x, weight, scales, *, block_size=32, quantized_input=None):
    """E4M3 x A16 with UE8M0 /32 or FP32 /128x128 scales.

    Reduction is blockwise FP32; four rows share every decoded weight.
    Activation quantization, when requested by the execution plan, precedes
    this operator explicitly. Output retains the input floating dtype.
    """
    if x.ndim < 2 or weight.ndim != 2 or weight.dtype != mx.uint8:
        raise ValueError("Expected floating rows and packed E4M3 byte matrix")
    n, k = weight.shape
    if x.shape[-1] != k or k % block_size or block_size not in (32, 128):
        raise ValueError("Invalid FP8 block geometry")
    shape = x.shape
    count = x.size // k
    value = x.reshape(count, k)
    if block_size == 32:
        if scales.dtype != mx.uint8 or scales.shape != (n, k // 32):
            raise ValueError("Invalid UE8M0 scales")
        ws = scales
        bs = mx.zeros((1,), mx.float32)
    else:
        if scales.dtype != mx.float32 or scales.size != ((n + 127) // 128) * (k // 128):
            raise ValueError("Invalid block FP32 scales")
        ws = mx.zeros((1,), mx.uint8)
        bs = scales.reshape(-1)
    rps = 2 if block_size == 128 and n >= 4096 else 4 if n >= 4096 else 1
    xq, xs = (
        quantized_input
        if quantized_input is not None
        else (mx.zeros((1,), mx.uint8), mx.zeros((1,), mx.float32))
    )
    return _kernel()(
        inputs=[value, weight, ws, bs, xq, xs],
        template=[
            ("O", x.dtype),
            ("RPS", rps),
            ("N", n),
            ("K", k),
            ("BLOCK", block_size),
            ("MX", block_size == 32),
            ("A8", quantized_input is not None),
            ("COUNT", count),
            ("ROWS", min(count, 4)),
        ],
        grid=(
            ((n + 4 * rps - 1) // (4 * rps)) * 128,
            1,
            (count + min(count, 4) - 1) // min(count, 4),
        ),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*shape[:-1], n)


_QUANT = r"""
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x;
uint block = threadgroup_position_in_grid.y;
float maximum = 0.0f;
for (uint i = lane; i < 32; i += 32) maximum = max(maximum, abs(float(x[row * K + block * 32 + i])));
maximum = simd_max(maximum);
float scale = maximum > 0.0f ? exp2(ceil(log2(maximum / 448.0f))) : 1.0f;
if (lane == 0) scales[row * (K / 32) + block] = scale;
for (uint i = lane; i < 32; i += 32) {
    float input = float(x[row * K + block * 32 + i]);
    float v = min(abs(input / scale), 448.0f);
    uint lo = 0, hi = 126;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (e4m3(uchar(mid)) < v) lo = mid + 1; else hi = mid;
    }
    uint code = lo;
    if (lo > 0) {
        float upper = e4m3(uchar(lo)), lower = e4m3(uchar(lo - 1));
        if (v - lower < upper - v || (v - lower == upper - v && (lo & 1))) code = lo - 1;
    }
    out[row * K + block * 32 + i] = uchar(code | (signbit(input) ? 128 : 0));
}
"""


@cache
def _activation_kernel():
    return mx.fast.metal_kernel(
        name="sglang_fp8_dynamic_32",
        input_names=["x"],
        output_names=["out", "scales"],
        source=_QUANT,
        header=_DECODE,
    )


def mlx_fp8_dynamic_32(x):
    """Dynamic per-row /32 E4M3 with ceil-UE8M0 scales and ties-to-even."""
    k = x.shape[-1]
    count = x.size // k
    if k % 32:
        raise ValueError("Dynamic FP8 activation requires /32 blocks")
    return tuple(
        _activation_kernel()(
            inputs=[x.reshape(count, k)],
            template=[("K", k)],
            grid=(count * 32, k // 32, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(count, k), (count, k // 32)],
            output_dtypes=[mx.uint8, mx.float32],
        )
    )


_PAIR_SOURCE = (
    _SOURCE.replace(
        "if (row >= N) return;",
        "if (row >= 2 * N) return;\n"
        "uint output_row = row;\n"
        "bool second = row >= N;\n"
        "row = row % N;\n"
        "const device uchar* weight = second ? up_weight : gate_weight;\n"
        "const device float* block_scales = second ? up_scales : gate_scales;\n"
        "const constant uchar* scales = dummy;",
        1,
    )
    .replace("out[(start + t) * N + row + r]", "out[(start + t) * (2 * N) + output_row + r]")
)


@cache
def _pair_kernel():
    return mx.fast.metal_kernel(
        name="sglang_fp8_gate_up_rows",
        input_names=["x", "gate_weight", "gate_scales", "up_weight", "up_scales", "dummy", "xq", "xscale"],
        output_names=["out"],
        source=_PAIR_SOURCE,
        header=_DECODE,
    )


def mlx_fp8_pair_rows(x, gate_weight, gate_scales, up_weight, up_scales, quantized_input):
    """Two block-FP8 projections share activation quantization and dispatch.

    Each output keeps the original lane-strided accumulation and rounding.
    Original weight arrays retain independent ownership without concatenation.
    """
    n, k = gate_weight.shape
    if up_weight.shape != (n, k) or n % 128 or k % 128 or x.shape[-1] != k:
        raise ValueError("Expected aligned equal block-FP8 gate/up projections")
    if any(w.dtype != mx.uint8 for w in (gate_weight, up_weight)):
        raise ValueError("Expected original E4M3 weight bytes")
    if any(s.dtype != mx.float32 or s.size != (n // 128) * (k // 128) for s in (gate_scales, up_scales)):
        raise ValueError("Expected original FP32 /128x128 scales")
    count = x.size // k
    rows = min(count, 4)
    rps = 2 if n >= 4096 else 1
    value = _pair_kernel()(
        inputs=[x.reshape(count, k), gate_weight, gate_scales, up_weight, up_scales,
                mx.zeros((1,), mx.uint8), *quantized_input],
        template=[("O", x.dtype), ("RPS", rps), ("N", n), ("K", k),
                  ("BLOCK", 128), ("MX", False), ("A8", True),
                  ("COUNT", count), ("ROWS", rows)],
        grid=(((2 * n + 4 * rps - 1) // (4 * rps)) * 128, 1, (count + rows - 1) // rows),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, 2 * n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*x.shape[:-1], 2 * n)
    return tuple(mx.split(value, 2, axis=-1))
