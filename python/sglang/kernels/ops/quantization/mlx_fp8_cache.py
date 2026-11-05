# SPDX-License-Identifier: Apache-2.0
"""Snapshot-safe E4M3 rows with native UE8M0 /32 scale semantics."""

from functools import cache

import mlx.core as mx

# Native MLX 0.32.4 FP8 conversion helpers (mlx, MIT license).
_HEADER = r"""

struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) thread {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      // Default behavior saturates to min/max
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) thread {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator bfloat16_t() thread {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() thread {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};

// Smallest E8M0 >= x. Scales are amax/max_element, so rounding one down
// leaves the block's largest elements outside the element range, where they
// saturate. Matches the CUDA backend, which rounds up via cutlass ue8m0.
inline float mx_scale_round_up(float x) {
  fp8_e8m0 s(x);
  if (s.bits < 0xFE && float(s) < x) {
    s.bits += 1;
  }
  return float(s);
}
"""
_PACK = r"""
const uint dim = dims[0];
const uint row = threadgroup_position_in_grid.x;
const uint block = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
float value = float(values[row * dim + block * 32 + lane]);
float maximum = simd_max(abs(value));
float rounded = mx_scale_round_up(maximum / 448.0f);
fp8_e8m0 scale(rounded);
float inverse = 1.0f / float(scale);
const uint offset = row * (dim + dim / 32);
packed[offset + block * 32 + lane] = fp8_e4m3(value * inverse).bits;
if (!lane) packed[offset + dim + block] = scale.bits;
"""
_UNPACK = r"""
const uint dim = dims[0];
const uint i = thread_position_in_grid.x;
if (i >= dims[1]) return;
const uint col = i % dim;
const uint offset = (i / dim) * (dim + dim / 32);
fp8_e4m3 value(0.0f);
fp8_e8m0 scale(1.0f);
value.bits = packed[offset + col];
scale.bits = packed[offset + dim + col / 32];
values[i] = bfloat(float(value) * float(scale));
"""


@cache
def _kernels():
    return (
        mx.fast.metal_kernel(
            name="sglang_mlx_fp8_cache_pack",
            input_names=["values", "dims"],
            output_names=["packed"],
            source=_PACK,
            header=_HEADER,
        ),
        mx.fast.metal_kernel(
            name="sglang_mlx_fp8_cache_unpack",
            input_names=["packed", "dims"],
            output_names=["values"],
            source=_UNPACK,
            header=_HEADER,
        ),
    )


def mlx_fp8_cache_pack(values):
    """Fuse native MXFP8 encoding and row/scale interleaving in one dispatch."""
    if values.ndim < 1 or values.dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("FP8 cache packing requires BF16 or FP32 rows")
    dim = values.shape[-1]
    if not dim or dim % 32 or not values.size or values.size >= 2**32:
        raise ValueError("FP8 cache requires nonempty /32 rows and uint32 indexing")
    shape = (*values.shape[:-1], dim + dim // 32)
    return _kernels()[0](
        inputs=[values, mx.array([dim], mx.uint32)],
        grid=(values.size // dim * 32, dim // 32, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.uint8],
    )[0]


def mlx_fp8_cache_unpack(packed, dim=512):
    """Dense logical view; sparse attention reads the packed cache directly."""
    if (
        packed.ndim < 1
        or packed.dtype != mx.uint8
        or not dim
        or dim % 32
        or packed.shape[-1] != dim + dim // 32
    ):
        raise ValueError("Invalid packed FP8 cache row")
    shape = (*packed.shape[:-1], dim)
    count = packed.size // packed.shape[-1] * dim
    if not count:
        return mx.zeros(shape, mx.bfloat16)
    return _kernels()[1](
        inputs=[packed, mx.array([dim, count], mx.uint32)],
        grid=((count + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
    )[0]
