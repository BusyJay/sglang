# SPDX-License-Identifier: Apache-2.0
# Reuses the native MLX QMV arithmetic transcribed in ops.moe.mlx_glm.
"""Bounded affine matrix rows with the singleton MLX QMV reduction order."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.moe.mlx_glm import _HEADER, _group_picks

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


_GATE_UP_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row = int(threadgroup_position_in_grid.x) * 4 + int(sg);
    const int start = int(threadgroup_position_in_grid.y) * ROWS;
    constexpr int PACK = glm_pack_factor<BITS>();
    constexpr int BYTES = glm_bytes_per_pack<BITS>();
    constexpr int VPT = 2 * PACK;
    constexpr int BLOCK = 32 * VPT;
    float value[ROWS] = {0};
    float up_value[ROWS] = {0};
    for (int k = 0; k < K; k += BLOCK) {
        float inputs[ROWS][VPT], sums[ROWS];
        #pragma clang loop unroll(full)
        for (int t = 0; t < ROWS; ++t)
            sums[t] = glm_load_vector<T,VPT,BITS>(x+size_t(start+t)*K+k+lane*VPT,inputs[t]);
        uchar packed[2*BYTES];
        const device uchar* src=(const device uchar*)w+size_t(row)*(K*BITS/8)+(k/PACK+lane*2)*BYTES;
        #pragma clang loop unroll(full)
        for(int i=0;i<2*BYTES;++i) packed[i]=src[i];
        float scale=s[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)];
        float bias=b[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)];

        uchar up_packed[2*BYTES];
        const device uchar* up_src=(const device uchar*)uw+size_t(row)*(K*BITS/8)+(k/PACK+lane*2)*BYTES;
        for(int i=0;i<2*BYTES;++i) up_packed[i]=up_src[i];
        float up_scale=us[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)];
        float up_bias=ub[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)];
        #pragma clang loop unroll(full)
        for(int t=0;t<ROWS;++t) {
            value[t]+=glm_qdot_shared<VPT,BITS>(packed,inputs[t],scale,bias,sums[t]);
            up_value[t]+=glm_qdot_shared<VPT,BITS>(up_packed,inputs[t],up_scale,up_bias,sums[t]);
        }

    }
    #pragma clang loop unroll(full)
    for(int t=0;t<ROWS;++t) {

        T gate=T(simd_sum(value[t]));
        T up=T(simd_sum(up_value[t]));
        if (HAS_LIMIT) { gate=T(min(float(gate),float(T(limit[0])))); up=T(clamp(float(up),float(T(-limit[0])),float(T(limit[0])))); }
        T sigmoid=glm_sigmoid<T>(gate);
        T activated=gate*sigmoid;
        T result=activated*up;

        if(lane==0) out[size_t(start+t)*N+row]=result;
    }
"""


# The same packed arithmetic consumes a thread-local tile loaded once for all
# matching expert rows. Keep each row's QMV order and affine rounding intact.
_SHARED_HEADER = _HEADER + _HEADER[
    _HEADER.index(
        "template <int values_per_thread, int bits>\ninline float glm_qdot"
    ) : _HEADER.index("// qmv_fast_impl")
].replace("glm_qdot(", "glm_qdot_shared(").replace("const device", "const thread")


@cache
def _gate_up_kernel():
    return mx.fast.metal_kernel(
        name="sglang_affine_shared_swiglu",
        input_names=["x", "w", "s", "b", "uw", "us", "ub", "limit"],
        output_names=["out"],
        source=_GATE_UP_SOURCE,
        header=_SHARED_HEADER,
    )


def mlx_affine_swiglu_rows(x, gate, up, *, bits, group_size, limit=None):
    """Shared dense gate/up input and weight tiles with rounded SwiGLU epilogue."""
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or bits not in (4, 5, 6, 8)
        or group_size not in (32, 64, 128)
        or x.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    pack = 8 if bits == 5 else 4 if bits == 6 else 32 // bits
    for weight, scales, biases in (gate, up):
        if (
            weight.ndim != 2
            or weight.dtype != mx.uint32
            or scales is None
            or biases is None
            or scales.dtype != x.dtype
            or biases.dtype != x.dtype
            or scales.shape != biases.shape
            or weight.shape[-1] * 32 != x.shape[-1] * bits
            or x.shape[-1] % (32 * pack * 2)
            or x.shape[-1] % group_size
            or weight.shape[0] % 16
            or scales.shape != (weight.shape[0], x.shape[-1] // group_size)
        ):
            return None
    if any(a.shape != b.shape for a, b in zip(gate, up)):
        return None
    out_dim = gate[0].shape[0]
    outputs = []
    for start in range(0, x.shape[1], 3):
        rows = min(3, x.shape[1] - start)
        outputs.append(
            _gate_up_kernel()(
                inputs=[
                    x[:, start : start + rows],
                    *gate,
                    *up,
                    mx.array([limit if limit is not None else 0], mx.float32),
                ],
                template=[
                    ("T", x.dtype),
                    ("K", x.shape[-1]),
                    ("N", out_dim),
                    ("GS", group_size),
                    ("BITS", bits),
                    ("ROWS", rows),
                    ("HAS_LIMIT", limit is not None),
                ],
                grid=(128 * (out_dim // 4), 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(1, rows, out_dim)],
                output_dtypes=[x.dtype],
            )[0]
        )
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)


_MIXED_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row = int(threadgroup_position_in_grid.x) * 4 + int(sg);
    constexpr int PACK = glm_pack_factor<BITS>();
    constexpr int BYTES = glm_bytes_per_pack<BITS>();
    constexpr int VPT = 2 * PACK;
    constexpr int BLOCK = 32 * VPT;
    float value[ROWS] = {0};
    for (int k = 0; k < K; k += BLOCK) {
        float inputs[ROWS][VPT], sums[ROWS];
        #pragma clang loop unroll(full)
        for (int t = 0; t < ROWS; ++t)
            sums[t] = glm_load_vector<X,VPT,BITS>(x+size_t(t)*K+k+lane*VPT,inputs[t]);
        uchar packed[2*BYTES];
        const device uchar* src=(const device uchar*)w+size_t(row)*(K*BITS/8)+(k/PACK+lane*2)*BYTES;
        for(int i=0;i<2*BYTES;++i) packed[i]=src[i];
        float scale=float(s[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]);
        float bias=float(b[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]);
        #pragma clang loop unroll(full)
        for(int t=0;t<ROWS;++t)
            value[t]+=glm_qdot_shared<VPT,BITS>(packed,inputs[t],scale,bias,sums[t]);
    }
    #pragma clang loop unroll(full)
    for(int t=0;t<ROWS;++t) {
        O result=O(simd_sum(value[t]));
        if(lane==0) out[size_t(t)*N+row]=result;
    }
"""


@cache
def _mixed_kernel():
    return mx.fast.metal_kernel(
        name="sglang_affine_mixed_shared_rows",
        input_names=["x", "w", "s", "b"],
        output_names=["out"],
        source=_MIXED_SOURCE,
        header=_SHARED_HEADER,
    )


def mlx_affine_mixed_rows(x, weight, scales, biases, *, bits, group_size):
    """Share packed weights without changing native activation/metadata promotion.

    No dequantized weight or promoted metadata allocation is retained. Input
    values and affine metadata convert to float in the QMV accumulation only.
    """
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or weight.ndim != 2
        or weight.dtype != mx.uint32
        or scales is None
        or biases is None
        or scales.shape != biases.shape
        or scales.dtype != biases.dtype
        or scales.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or bits not in (4, 5, 6, 8)
        or group_size not in (32, 64, 128)
    ):
        return None
    count, dim = x.shape[1:]
    out_dim = weight.shape[0]
    pack = 8 if bits == 5 else 4 if bits == 6 else 32 // bits
    if (
        weight.shape[1] * 32 != dim * bits
        or dim % (64 * pack)
        or dim % group_size
        or out_dim % 4
        or scales.shape != (out_dim, dim // group_size)
    ):
        return None
    dtype = mx.result_type(x.dtype, scales.dtype, biases.dtype)
    outputs = []
    for start in range(0, count, 3):
        rows = min(3, count - start)
        outputs.append(
            _mixed_kernel()(
                inputs=[x[:, start : start + rows], weight, scales, biases],
                template=[
                    ("X", x.dtype),
                    ("O", dtype),
                    ("K", dim),
                    ("N", out_dim),
                    ("GS", group_size),
                    ("BITS", bits),
                    ("ROWS", rows),
                ],
                grid=(128 * (out_dim // 4), 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(1, rows, out_dim)],
                output_dtypes=[dtype],
            )[0]
        )
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, 1)


_PACKED_LOAD = r"""
template <int BYTES>
inline void glm_load_packed(const device uchar* base, size_t stride,
                            size_t offset, thread uchar* result) {
    if (stride == 1 && BYTES != 8) {
        // uint32-aligned loads also cover unaligned 5/6-bit byte offsets.
        const device uint* words = (const device uint*)base;
        const size_t first = offset / 4;
        const uint shift = uint(offset % 4);
        uint loaded[(BYTES + 6) / 4];
        const uint count = (shift + BYTES + 3) / 4;
        for (uint i=0; i<count; ++i) loaded[i] = words[first+i];
        for (uint i=0; i<BYTES; ++i)
            result[i] = uchar(loaded[(shift+i)/4] >> (((shift+i)%4)*8));
    } else {
        for (uint i=0; i<BYTES; ++i)
            result[i] = base[((offset+i)/4)*stride*4 + (offset+i)%4];
    }
}
"""


_MASK4_HEADER = r"""
// Load four native ushort packs as two aligned words. Keep the original
// four-term sum and affine rounding, while reusing masks across matched rows.
inline uint4 glm_load_masks4(const device uchar* base, size_t stride, size_t offset) {
    const device uint* words = (const device uint*)base;
    const uint a = words[(offset / 4) * stride];
    const uint b = words[(offset / 4 + 1) * stride];
    return uint4(a & 65535, a >> 16, b & 65535, b >> 16);
}
inline float glm_mask_qdot4(uint4 packed, const thread float* x,
                            float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < 4; ++i)
        accum += (x[4*i] * (packed[i] & 0x000f) +
                  x[4*i+1] * (packed[i] & 0x00f0) +
                  x[4*i+2] * (packed[i] & 0x0f00) +
                  x[4*i+3] * (packed[i] & 0xf000));
    return scale * accum + sum * bias;
}
"""


_ROUTED_GATE_UP = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = (threadgroup_position_in_grid.x * 4 + sg) * 4;
    const uint pick = threadgroup_position_in_grid.y;
    const uint tok = pick / TOPK;
    const uint expert = indices[pick];
    float gate[4] = {0}, up[4] = {0};
    constexpr int PACK = glm_pack_factor<BITS>();
    constexpr int BYTES = glm_bytes_per_pack<BITS>();
    constexpr int VPT = 2 * PACK;
    constexpr int BLOCK = 32 * VPT;
    // Both producers consume the same lane-local input and affine sum. Load
    // it once, retaining each projection's qdot and accumulation order.
    for (int k = int(lane) * VPT; k < K; k += BLOCK) {
        float input[VPT];
        float sum = glm_load_vector<T, VPT, BITS>(x + size_t(tok) * K + k, input);
        for (int j = 0; j < 4; ++j) {
            const size_t gr = size_t(expert)*gw_strides[0]*4 + size_t(row+j)*gw_strides[1]*4;
            const size_t ur = size_t(expert)*uw_strides[0]*4 + size_t(row+j)*uw_strides[1]*4;
            const size_t packed = k * BYTES / PACK;
            uchar gp[2*BYTES], upack[2*BYTES];
            glm_load_packed<2*BYTES>((const device uchar*)gw+gr,gw_strides[2],packed,gp);
            glm_load_packed<2*BYTES>((const device uchar*)uw+ur,uw_strides[2],packed,upack);
            gate[j] += glm_qdot_shared<VPT, BITS>(gp, input,
                gs[size_t(expert)*gs_strides[0] + size_t(row+j)*gs_strides[1] + size_t(k/GS)*gs_strides[2]],
                gb[size_t(expert)*gb_strides[0] + size_t(row+j)*gb_strides[1] + size_t(k/GS)*gb_strides[2]], sum);
            up[j] += glm_qdot_shared<VPT, BITS>(upack, input,
                us[size_t(expert)*us_strides[0] + size_t(row+j)*us_strides[1] + size_t(k/GS)*us_strides[2]],
                ub[size_t(expert)*ub_strides[0] + size_t(row+j)*ub_strides[1] + size_t(k/GS)*ub_strides[2]], sum);
        }
    }
    for (uint i = 0; i < 4; ++i) {
        T g = T(simd_sum(gate[i]));
        T u = T(simd_sum(up[i]));
        if (HAS_LIMIT) {
            g = T(min(float(g), float(T(limit[0]))));
            u = T(clamp(float(u), float(T(-limit[0])), float(T(limit[0]))));
        }
        T sigmoid = glm_sigmoid<T>(g);
        T activated = g * sigmoid;
        T result = activated * u;
        if (lane == 0) out[size_t(pick) * N + row + i] = result;
    }
"""


# Group picks across the verification chain on-device. The first pick owns
# the expert; bounded three-row arithmetic tiles retain singleton rounding.
_GROUPED_GATE_UP_PREFIX = r"""
const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = (threadgroup_position_in_grid.x * 4 + sg) * RPS;
    const uint first = threadgroup_position_in_grid.y;
    const uint expert=indices[first];
    const uint matches=counts[first];
    if(matches==0) return;
    const device uint* picks=grouped+first*GROUP_WIDTH;
    constexpr int PACK = glm_pack_factor<BITS>();
    constexpr int BYTES = glm_bytes_per_pack<BITS>();
    constexpr int VPT = 2 * PACK;
    constexpr int BLOCK = 32 * VPT;
    const size_t base = size_t(expert) * N + row;
    for(uint start=0;start<matches;start+=TILE) {
const uint rows=min(uint(TILE),matches-start);
"""

_GROUPED_GATE_UP_SINGLE = r"""
float gate[RPS] = {0}, up[RPS] = {0};
    constexpr int PACK = glm_pack_factor<BITS>();
    constexpr int BYTES = glm_bytes_per_pack<BITS>();
    constexpr int VPT = 2 * PACK;
    constexpr int BLOCK = 32 * VPT;
    // Both producers consume the same lane-local input and affine sum. Load
    // it once, retaining each projection's qdot and accumulation order.
    for (int k = int(lane) * VPT; k < K; k += BLOCK) {
        float input[VPT];
        float sum = glm_load_vector<T, VPT, BITS>(x + size_t(picks[start] / TOPK) * K + k, input);
        for (int j = 0; j < RPS; ++j) {
            const size_t gr = size_t(expert)*gw_strides[0]*4 + size_t(row+j)*gw_strides[1]*4;
            const size_t ur = size_t(expert)*uw_strides[0]*4 + size_t(row+j)*uw_strides[1]*4;
            const size_t packed = k * BYTES / PACK;
            uchar gp[2*BYTES], upack[2*BYTES];
            glm_load_packed<2*BYTES>((const device uchar*)gw+gr,gw_strides[2],packed,gp);
            glm_load_packed<2*BYTES>((const device uchar*)uw+ur,uw_strides[2],packed,upack);
            gate[j] += glm_qdot_shared<VPT, BITS>(gp, input,
                gs[size_t(expert)*gs_strides[0] + size_t(row+j)*gs_strides[1] + size_t(k/GS)*gs_strides[2]],
                gb[size_t(expert)*gb_strides[0] + size_t(row+j)*gb_strides[1] + size_t(k/GS)*gb_strides[2]], sum);
            up[j] += glm_qdot_shared<VPT, BITS>(upack, input,
                us[size_t(expert)*us_strides[0] + size_t(row+j)*us_strides[1] + size_t(k/GS)*us_strides[2]],
                ub[size_t(expert)*ub_strides[0] + size_t(row+j)*ub_strides[1] + size_t(k/GS)*ub_strides[2]], sum);
        }
    }
    for (uint i = 0; i < RPS; ++i) {
        T g = T(simd_sum(gate[i]));
        T u = T(simd_sum(up[i]));
        if (HAS_LIMIT) {
            g = T(min(float(g), float(T(limit[0]))));
            u = T(clamp(float(u), float(T(-limit[0])), float(T(limit[0]))));
        }
        T sigmoid = glm_sigmoid<T>(g);
        T activated = g * sigmoid;
        T result = activated * u;
        if (lane == 0) out[size_t(picks[start]) * N + row + i] = result;
    }
"""

_GROUPED_GATE_UP_ROWS = r"""
float gate[ROWS][RPS] = {0}, up[ROWS][RPS] = {0};
        for (int k = int(lane) * VPT; k < K; k += BLOCK) {
            float inputs[ROWS][VPT], sums[ROWS];

#pragma clang loop unroll(full)
for (uint t = 0; t < ROWS; ++t)
                sums[t] = glm_load_vector<T,VPT,BITS>(
                    x + size_t(picks[start+t] / TOPK) * K + k, inputs[t]);

#pragma clang loop unroll(full)
for (uint j = 0; j < RPS; ++j) {
                const size_t gr = size_t(expert)*gw_strides[0]*4 + size_t(row+j)*gw_strides[1]*4;
                const size_t ur = size_t(expert)*uw_strides[0]*4 + size_t(row+j)*uw_strides[1]*4;
                const size_t packed = k * BYTES / PACK;
                uchar gp[2*BYTES], upack[2*BYTES];
                glm_load_packed<2*BYTES>((const device uchar*)gw+gr,gw_strides[2],packed,gp);
                glm_load_packed<2*BYTES>((const device uchar*)uw+ur,uw_strides[2],packed,upack);
                float gscale=gs[size_t(expert)*gs_strides[0] + size_t(row+j)*gs_strides[1] + size_t(k/GS)*gs_strides[2]], gbias=gb[size_t(expert)*gb_strides[0] + size_t(row+j)*gb_strides[1] + size_t(k/GS)*gb_strides[2]];
                float uscale=us[size_t(expert)*us_strides[0] + size_t(row+j)*us_strides[1] + size_t(k/GS)*us_strides[2]], ubias=ub[size_t(expert)*ub_strides[0] + size_t(row+j)*ub_strides[1] + size_t(k/GS)*ub_strides[2]];

#pragma clang loop unroll(full)
for (uint t=0; t<ROWS; ++t) {
                    gate[t][j] += glm_qdot_shared<VPT,BITS>(gp,inputs[t],gscale,gbias,sums[t]);
                    up[t][j] += glm_qdot_shared<VPT,BITS>(upack,inputs[t],uscale,ubias,sums[t]);
                }
            }
        }

#pragma clang loop unroll(full)
for (uint t=0; t<ROWS; ++t)

#pragma clang loop unroll(full)
for (uint j=0; j<RPS; ++j) {
                T g=T(simd_sum(gate[t][j])), u=T(simd_sum(up[t][j]));
                if (HAS_LIMIT) {
                    g=T(min(float(g),float(T(limit[0]))));
                    u=T(clamp(float(u),float(T(-limit[0])),float(T(limit[0]))));
                }
                T sigmoid=glm_sigmoid<T>(g);
                T activated=g*sigmoid;
                T result=activated*u;
                if(lane==0) out[size_t(picks[start+t])*N+row+j]=result;
            }
"""

_GROUPED_GATE_UP_SUFFIX = r"""
}
"""

_GROUPED_GATE_UP = (
    _GROUPED_GATE_UP_PREFIX
    + "if (rows == 1) {\n"
    + _GROUPED_GATE_UP_SINGLE
    + "}\n"
    + "".join(
        "else if (rows == "
        + str(rows)
        + ") {\n"
        + _GROUPED_GATE_UP_ROWS.replace("ROWS", str(rows))
        + "}\n"
        for rows in (2, 3)
    )
    + _GROUPED_GATE_UP_SUFFIX
)


def _masked_gate_up_source():
    source = _GROUPED_GATE_UP
    source = source.replace(
        "uchar gp[2*BYTES], upack[2*BYTES];",
        "uint4 gp = glm_load_masks4((const device uchar*)gw+gr, gw_strides[2], packed);"
        "uint4 upack = glm_load_masks4((const device uchar*)uw+ur, uw_strides[2], packed);",
    )
    for name, weight in (("gp", "gw"), ("upack", "uw")):
        source = source.replace(
            f"glm_load_packed<2*BYTES>((const device uchar*){weight}+"
            f"{'gr' if weight == 'gw' else 'ur'},{weight}_strides[2],packed,{name});",
            "",
        )
        for signature in ("VPT, BITS", "VPT,BITS"):
            source = source.replace(
                f"glm_qdot_shared<{signature}>({name}", f"glm_mask_qdot4({name}"
            )
    return source


@cache
def _masked_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="sglang_routed_masked_swiglu",
        input_names=[
            "x",
            "gw",
            "gs",
            "gb",
            "uw",
            "us",
            "ub",
            "indices",
            "limit",
            "grouped",
            "counts",
        ],
        output_names=["out"],
        source=_masked_gate_up_source(),
        header=_SHARED_HEADER + _MASK4_HEADER,
        ensure_row_contiguous=False,
    )


@cache
def _grouped_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="sglang_routed_grouped_swiglu",
        input_names=[
            "x",
            "gw",
            "gs",
            "gb",
            "uw",
            "us",
            "ub",
            "indices",
            "limit",
            "grouped",
            "counts",
        ],
        output_names=["out"],
        source=_GROUPED_GATE_UP,
        header=_SHARED_HEADER + _PACKED_LOAD,
        ensure_row_contiguous=False,
    )


@cache
def _routed_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="sglang_routed_gate_up_swiglu",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "indices", "limit"],
        output_names=["out"],
        source=_ROUTED_GATE_UP,
        header=_SHARED_HEADER + _PACKED_LOAD,
        ensure_row_contiguous=False,
    )


def mlx_affine_routed_swiglu(
    x, gate, up, indices, *, bits, group_size, limit=None, grouped=None
):
    """Fuse routed gate/up and rounded activation without duplicating weights.

    Each SIMD owns bounded output rows. Matching experts are grouped across the
    chain and processed in three-row tiles sharing packed weights and metadata.
    Outputs scatter to the original token/top-k slots, including duplicates.
    """
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or indices.ndim != 3
        or indices.shape[:2] != x.shape[:2]
        or indices.dtype not in (mx.int32, mx.uint32)
        or not 1 <= indices.shape[-1] <= 32
        or x.dtype not in (mx.float16, mx.bfloat16)
        or bits not in (4, 5, 6, 8)
        or group_size not in (32, 64, 128)
    ):
        return None
    k = x.shape[-1]
    pack = 8 if bits == 5 else 4 if bits == 6 else 32 // bits
    if k % (64 * pack) or k % group_size:
        return None
    for w, sc, bi in (gate, up):
        if (
            w.ndim != 3
            or w.dtype != mx.uint32
            or w.shape[-1] * 32 != k * bits
            or w.shape[-2] % 16
            or sc is None
            or bi is None
            or sc.dtype != x.dtype
            or bi.dtype != x.dtype
            or sc.shape != w.shape[:-1] + (k // group_size,)
            or bi.shape != sc.shape
        ):
            return None
    if any(a.shape != b.shape for a, b in zip(gate, up)):
        return None
    n, topk = gate[0].shape[-2], indices.shape[-1]
    # Activations and ids are small; prepared expert weights must retain their
    # strides rather than materializing complete gate/up matrices each decode.
    x, indices = mx.contiguous(x), mx.contiguous(indices)
    inputs = [x, *gate, *up, indices, mx.array([limit or 0], mx.float32)]
    if x.shape[1] > 1:
        inputs.extend(
            _group_picks(indices, x.shape[1])
            if grouped is None
            else tuple(mx.contiguous(v) for v in grouped)
        )
    kernel = (
        _masked_gate_up_kernel()
        if bits == 4
        and x.shape[1] > 1
        and mx.device_info()["architecture"].startswith("applegpu_g17")
        else (_grouped_gate_up_kernel() if x.shape[1] > 1 else _routed_gate_up_kernel())
    )
    return kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("GS", group_size),
            ("BITS", bits),
            ("TOPK", topk),
            ("COUNT", x.shape[1]),
            ("TILE", min(3, x.shape[1])),
            ("GROUP_WIDTH", inputs[-2].shape[1] if x.shape[1] > 1 else topk),
            ("RPS", 2),
            ("HAS_LIMIT", limit is not None),
        ],
        grid=(128 * (n // (8 if x.shape[1] > 1 else 16)), x.shape[1] * topk, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, x.shape[1], topk, n)],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _multi_mixed_kernel(parts):
    names = ["x"]
    outputs, body = [], []
    offset = 0
    for i, (bits, group_size, width) in enumerate(parts):
        names.extend([f"w{i}", f"s{i}", f"b{i}"])
        outputs.append(f"out{i}")
        text = _MIXED_SOURCE
        text = text.replace(
            "const int row = int(threadgroup_position_in_grid.x) * 4 + int(sg);",
            f"const int row = (int(threadgroup_position_in_grid.x)-{offset})*4+int(sg); if(row>={width}) return;",
        )
        text = text.replace(
            "x+size_t(t)*K+k+lane*VPT",
            "x+(size_t(threadgroup_position_in_grid.y)*3+t)*K+k+lane*VPT",
        )
        text = text.replace(
            "const device uchar* src=(const device uchar*)w+size_t(row)*(K*BITS/8)+(k/PACK+lane*2)*BYTES;\n        for(int i=0;i<2*BYTES;++i) packed[i]=src[i];",
            "const device uchar* src=(const device uchar*)w+size_t(row)*w_strides[0]*4;\n        glm_load_packed<2*BYTES>(src,w_strides[1],(k/PACK+lane*2)*BYTES,packed);",
        )
        text = text.replace(
            "s[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]",
            "s[size_t(row)*s_strides[0]+(k/GS+lane/(GS/VPT))*s_strides[1]]",
        )
        text = text.replace(
            "b[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]",
            "b[size_t(row)*b_strides[0]+(k/GS+lane/(GS/VPT))*b_strides[1]]",
        )
        text = text.replace(
            "out[size_t(t)*N+row]",
            "out[(size_t(threadgroup_position_in_grid.y)*3+t)*N+row]",
        )
        text = text.replace(
            "for(int t=0;t<ROWS;++t) {",
            "for(int t=0;t<ROWS;++t) { if(threadgroup_position_in_grid.y*3+t>=COUNT) break;",
        )
        text = text.replace(
            "for (int t = 0; t < ROWS; ++t)\n            sums[t]",
            "for (int t = 0; t < ROWS; ++t)\n            sums[t]",
        )
        # Tail inputs alias the final valid row; guarded stores exclude them.
        text = text.replace(
            "(size_t(threadgroup_position_in_grid.y)*3+t)*K",
            "size_t(min(uint(threadgroup_position_in_grid.y*3+t),uint(COUNT-1)))*K",
        )
        import re

        for symbol in ("w", "s", "b", "out"):
            text = re.sub(r"\b" + symbol + r"\b", symbol + str(i), text)
            text = text.replace(symbol + "_strides", symbol + str(i) + "_strides")
        for symbol, value in (("BITS", bits), ("GS", group_size), ("N", width)):
            text = re.sub(r"\b" + symbol + r"\b", str(value), text)
        text = re.sub(r"\bO\b", f"O{i}", text)
        tiles = (width + 3) // 4
        body.append(
            f"if(threadgroup_position_in_grid.x>={offset} && threadgroup_position_in_grid.x<{offset + tiles}) {{\n{text}\n}}"
        )
        offset += tiles
    return mx.fast.metal_kernel(
        name="sglang_affine_multi_rows",
        input_names=names,
        output_names=outputs,
        source="\n".join(body),
        header=_SHARED_HEADER + _PACKED_LOAD,
        ensure_row_contiguous=False,
    )


def mlx_affine_multi_rows(x, parts, *, prepared_strides=None):
    """Independent affine projections in one dispatch, preserving local promotion.

    Each part is (weight, scales, biases, bits, group_size). Weight and metadata
    strides remain independent; only the bounded activation is made contiguous.
    """
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 17
        or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or not parts
    ):
        return None
    count, dim = x.shape[1:]
    specs, inputs, dtypes = [], [mx.contiguous(x)], []
    for w, s, b, bits, gs in parts:
        pack = (
            8 if bits == 5 else 4 if bits == 6 else 32 // bits if bits in (4, 8) else 0
        )
        if (
            not pack
            or gs not in (32, 64, 128)
            or dim % (64 * pack)
            or dim % gs
            or w.ndim != 2
            or w.dtype != mx.uint32
            or w.shape[1] * 32 != dim * bits
            or s is None
            or b is None
            or s.shape != (w.shape[0], dim // gs)
            or b.shape != s.shape
            or s.dtype != b.dtype
            or s.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        ):
            return None
        specs.append((bits, gs, w.shape[0]))
        inputs.extend((w, s, b))
        dtypes.append(mx.result_type(x.dtype, s.dtype, b.dtype))
    return (
        _multi_mixed_kernel(tuple(specs))
        if prepared_strides is None
        else _e_prepared_kernel(tuple(specs), tuple(tuple(v) for v in prepared_strides))
    )(
        inputs=inputs,
        template=[("X", x.dtype), ("K", dim), ("ROWS", min(3, count)), ("COUNT", count)]
        + [(f"O{i}", dtype) for i, dtype in enumerate(dtypes)],
        grid=(128 * sum((n + 3) // 4 for _, _, n in specs), (count + 2) // 3, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, count, n) for _, _, n in specs],
        output_dtypes=dtypes,
    )


@cache
def _e_prepared_kernel(parts, strides):
    if len(strides) != len(parts) or any(
        len(v) != 6 or any(type(x) is not int or x < 0 for x in v) for v in strides
    ):
        raise ValueError("explicit prepared element strides required")
    names = ["x"]
    outputs, body = [], []
    offset = 0
    for i, (bits, group_size, width) in enumerate(parts):
        names.extend([f"w{i}", f"s{i}", f"b{i}"])
        outputs.append(f"out{i}")
        text = _MIXED_SOURCE
        text = text.replace(
            "const int row = int(threadgroup_position_in_grid.x) * 4 + int(sg);",
            f"const int row = (int(threadgroup_position_in_grid.x)-{offset})*4+int(sg); if(row>={width}) return;",
        )
        text = text.replace(
            "x+size_t(t)*K+k+lane*VPT",
            "x+(size_t(threadgroup_position_in_grid.y)*3+t)*K+k+lane*VPT",
        )
        text = text.replace(
            "const device uchar* src=(const device uchar*)w+size_t(row)*(K*BITS/8)+(k/PACK+lane*2)*BYTES;\n        for(int i=0;i<2*BYTES;++i) packed[i]=src[i];",
            "const device uchar* src=(const device uchar*)w+size_t(row)*w_strides[0]*4;\n        glm_load_packed<2*BYTES>(src,w_strides[1],(k/PACK+lane*2)*BYTES,packed);",
        )
        text = text.replace(
            "s[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]",
            "s[size_t(row)*s_strides[0]+(k/GS+lane/(GS/VPT))*s_strides[1]]",
        )
        text = text.replace(
            "b[size_t(row)*(K/GS)+k/GS+lane/(GS/VPT)]",
            "b[size_t(row)*b_strides[0]+(k/GS+lane/(GS/VPT))*b_strides[1]]",
        )
        text = text.replace(
            "out[size_t(t)*N+row]",
            "out[(size_t(threadgroup_position_in_grid.y)*3+t)*N+row]",
        )
        text = text.replace(
            "for(int t=0;t<ROWS;++t) {",
            "for(int t=0;t<ROWS;++t) { if(threadgroup_position_in_grid.y*3+t>=COUNT) break;",
        )
        # Tail inputs alias the final valid row; guarded stores exclude them.
        text = text.replace(
            "(size_t(threadgroup_position_in_grid.y)*3+t)*K",
            "size_t(min(uint(threadgroup_position_in_grid.y*3+t),uint(COUNT-1)))*K",
        )
        # Same geometries and accumulation expressions; only metadata accesses
        # become constants. Specialize actual prepared viewstrides, not lengths.
        for symbol, pair in zip(
            ("w", "s", "b"), (strides[i][:2], strides[i][2:4], strides[i][4:])
        ):
            for axis, value in enumerate(pair):
                text = text.replace(
                    symbol + "_strides[" + str(axis) + "]",
                    "size_t(" + str(value) + "ul)",
                )
        import re

        for symbol in ("w", "s", "b", "out"):
            text = re.sub(r"\b" + symbol + r"\b", symbol + str(i), text)
            text = text.replace(symbol + "_strides", symbol + str(i) + "_strides")
        for symbol, value in (("BITS", bits), ("GS", group_size), ("N", width)):
            text = re.sub(r"\b" + symbol + r"\b", str(value), text)
        text = re.sub(r"\bO\b", f"O{i}", text)
        tiles = (width + 3) // 4
        body.append(
            f"if(threadgroup_position_in_grid.x>={offset} && threadgroup_position_in_grid.x<{offset + tiles}) {{\n{text}\n}}"
        )
        offset += tiles
    return mx.fast.metal_kernel(
        name="sglang_affine_e_static_strides",
        input_names=names,
        output_names=outputs,
        source="\n".join(body),
        header=_SHARED_HEADER + _PACKED_LOAD,
        ensure_row_contiguous=False,
    )


@cache
def _multi_stride_metadata_kernel(count):
    names = [f"a{i}" for i in range(count)]
    return mx.fast.metal_kernel(
        name=f"sglang_affine_multi_stride_metadata_{count}",
        input_names=names,
        output_names=["table"],
        ensure_row_contiguous=False,
        source="\n".join(
            f"table[{2 * i}]=a{i}_strides[0];table[{2 * i + 1}]=a{i}_strides[1];"
            for i in range(count)
        ),
    )


def mlx_affine_multi_prepare_strides(parts):
    """Extract actual element strides outside compiled calls and serving timing."""
    arrays = [a for part in parts for a in part[:3]]
    if not arrays or any(a is None or a.ndim != 2 for a in arrays):
        raise ValueError("nonempty 2D affine operands required")
    table = _multi_stride_metadata_kernel(len(arrays))(
        inputs=arrays,
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(2 * len(arrays),)],
        output_dtypes=[mx.uint64],
    )[0]
    mx.eval(table)
    values = table.tolist()
    return tuple(tuple(values[i : i + 6]) for i in range(0, len(values), 6))
