# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed A16 projection retaining E2M1/E4M3 and global calibration."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.gemm.mlx_fp8_rows import _DECODE

_HEADER = (
    _DECODE
    + r"""
inline float e2m1(uint code) {
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return code & 8 ? -values[code & 7] : values[code & 7];
}
"""
)
_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint feature = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
uint pick = threadgroup_position_in_grid.y;
uint expert = indices[pick];
uint token = pick / PICKS;
if (feature >= N) return;
float sum = 0.0f;
for (uint group = lane; group < K / 16; group += 32) {
    float scale = e4m3(scales[(size_t(expert) * N + feature) * (K / 16) + group]);
    float partial = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        uint k = group * 16 + i;
        uint byte = weight[(size_t(expert) * N + feature) * (K / 2) + k / 2];
        float wv = e2m1((byte >> ((k % 2) * 4)) & 15) * scale;
        float xv = float(x[(EXPERT_INPUT ? pick : token) * K + k]);
        partial += xv * wv;
    }
    sum += partial;
}
for (ushort off = 16; off >= 1; off >>= 1) sum += simd_shuffle_down(sum, off);
if (lane == 0) out[pick * N + feature] = O(sum * global_scale[expert]);
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_nvfp4_routed_a16",
        input_names=["x", "weight", "scales", "global_scale", "indices"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


def mlx_nvfp4_routed(x, weight, scales, global_scale, indices):
    """Packed per-expert projection with explicit selected indices.

    Codes and scales stay native. The output rounds once to activation dtype;
    routing-weighted combination remains the caller's original ordered sum.
    """
    experts, n, packed = weight.shape
    k = packed * 2
    if (
        weight.dtype != mx.uint8
        or scales.dtype != mx.uint8
        or scales.shape != (experts, n, k // 16)
        or global_scale.dtype != mx.float32
    ):
        raise ValueError("Invalid NVFP4 bank")
    if x.shape[-1] != k:
        raise ValueError("NVFP4 input width differs")
    picks = indices.shape[-1]
    count = indices.size
    expert_input = x.size // k == count
    result = _kernel()(
        inputs=[
            x.reshape(-1, k),
            weight,
            scales,
            global_scale.reshape(-1),
            indices.reshape(-1).astype(mx.uint32),
        ],
        template=[
            ("O", x.dtype),
            ("N", n),
            ("K", k),
            ("PICKS", picks),
            ("EXPERT_INPUT", expert_input),
        ],
        grid=(((n + 3) // 4) * 128, count, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0]
    return result.reshape(*indices.shape, 1, n)
