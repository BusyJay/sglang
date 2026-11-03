# SPDX-License-Identifier: Apache-2.0
"""Dynamic /32 E4M3 encoding with power-of-two scales and ties to even."""

from functools import cache

import mlx.core as mx

_ENCODE = r"""
inline uchar mlx_encode_e4m3(float input, float scale) {
    float v = min(abs(input / scale), 448.0f);
    uint code;
    if (v < 0.015625f) {
        code = uint(rint(v * 512.0f));
    } else {
        uint bits = as_type<uint>(v);
        int exponent = int((bits >> 23) & 255) - 127;
        uint mantissa = bits & 0x7fffff;
        code = uint((exponent + 7) * 8) + (mantissa >> 20);
        uint tail = mantissa & 0xfffff;
        code += uint(tail > 0x80000 || (tail == 0x80000 && (code & 1)));
        code = min(code, 126u);
    }
    return uchar(code | (signbit(input) ? 128 : 0));
}
"""

_SOURCE = r"""
const uint K = dims[0];
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x;
uint block = threadgroup_position_in_grid.y;
uint i = row * K + block * 32 + lane;
float value = float(x[i]);
float maximum = simd_max(abs(value));
float scale = maximum > 0.0f ? exp2(ceil(log2(maximum / 448.0f))) : 1.0f;
if (lane == 0) scales[row * (K / 32) + block] = scale;
out[i] = mlx_encode_e4m3(value, scale);
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_mlx_fp8_dynamic_direct",
        input_names=["x", "dims"],
        output_names=["out", "scales"],
        source=_SOURCE,
        header=_ENCODE,
    )


def mlx_fp8_dynamic_32(x):
    """Encode finite BF16/F32 rows, retaining signed zero and E4M3 rounding.

    Dimensions are runtime parameters; changing row counts does not specialize
    the shader. Returns uint8 codes and FP32 power-of-two scales.
    """
    if x.ndim < 1 or x.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("Expected BF16 or FP32 rows")
    k = x.shape[-1]
    count = x.size // k if k else 0
    if not k or k % 32 or not count or x.size >= 2**32:
        raise ValueError("Expected nonempty /32 rows fitting uint32 indexing")
    return tuple(
        _kernel()(
            inputs=[x.reshape(count, k), mx.array([k], mx.uint32)],
            grid=(count * 32, k // 32, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(count, k), (count, k // 32)],
            output_dtypes=[mx.uint8, mx.float32],
        )
    )
