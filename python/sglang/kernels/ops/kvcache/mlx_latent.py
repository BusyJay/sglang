# SPDX-License-Identifier: Apache-2.0
"""Gather shared MLX latent KV and convert at the attention promotion boundary."""

from functools import cache

import mlx.core as mx

_LAYOUT = r"""
    const uint i = thread_position_in_grid.x;
    const uint width = layout[0];
    const uint rows = layout[1];
    if (i >= rows * width * D) return;
    const uint row = i / (width * D);
    const uint position = (i / D) % width;
    const uint key = uint(max(int(indices[row * width + position]), 0));
    const uint split = layout[2];
"""

_PACKED = r"""
    const device uchar* source = key < split ? kv : tail;
    const size_t base = size_t(key < split ? key : key - split) * (D + D/32);
    const uint col = i % D;
    fp8_e4m3 value(0.0f);
    value.bits = source[base + col];
    fp8_e8m0 scale(0.0f);
    scale.bits = source[base + D + col/32];
    // Logical FP8 cache reads first produce BF16, then attention promotes.
    out[i] = O(bfloat16_t(float(value) * float(scale)));
"""
_FLOATING = r"""
    out[i] = key < split ? O(kv[size_t(key) * D + i % D])
                        : O(tail[size_t(key - split) * D + i % D]);
"""


@cache
def _kernel(packed):
    from sglang.kernels.ops.quantization.mlx_fp8_cache import _HEADER

    return mx.fast.metal_kernel(
        name="sglang_latent_gather_fp8" if packed else "sglang_latent_gather_cast",
        input_names=["kv", "tail", "indices", "layout"],
        output_names=["out"],
        source=_LAYOUT + (_PACKED if packed else _FLOATING),
        header=_HEADER,
    )


def mlx_latent_gather_cast(kv, indices, dtype, *, tail=None):
    """KV [1,L,D], indices [1,R,W] -> [R,1,W,D] with explicit promotion.

    Packed E4M3/UE8M0 rows decode only selected latents; no dense history
    allocation is needed. Negative padding selects row zero, matching the
    caller's mask convention.
    Valid indices are below the combined prefix and tail length. Width/rows
    are runtime arguments so context growth does not compile another shader.
    """
    packed = kv.dtype == mx.uint8
    if (
        kv.ndim != 3
        or indices.ndim != 3
        or kv.shape[0] != 1
        or indices.shape[0] != 1
        or kv.dtype not in (mx.uint8, mx.float16, mx.bfloat16, mx.float32)
        or dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or indices.dtype not in (mx.int32, mx.uint32)
        or mx.result_type(mx.bfloat16 if packed else kv.dtype, dtype) != dtype
    ):
        return None
    rows, width = indices.shape[1:]
    stored_dim = kv.shape[-1]
    if packed and (stored_dim % 33 or stored_dim == 0):
        return None
    dim = stored_dim // 33 * 32 if packed else stored_dim
    if tail is None:
        tail = kv
    elif (
        tail.ndim != 3
        or tail.shape[0] != 1
        or tail.shape[2] != stored_dim
        or tail.dtype != kv.dtype
    ):
        return None
    if not width:
        return mx.zeros((rows, 1, 0, dim), dtype)
    return _kernel(packed)(
        inputs=[kv, tail, indices, mx.array([width, rows, kv.shape[1]], mx.uint32)],
        template=[("O", dtype), ("D", dim)],
        grid=(rows * width * dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 1, width, dim)],
        output_dtypes=[dtype],
    )[0]
