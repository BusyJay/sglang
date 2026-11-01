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
            for (uint i=0; i<2*BYTES; ++i) {
                gp[i] = ((const device uchar*)gw)[gr + ((packed+i)/4)*gw_strides[2]*4 + (packed+i)%4];
                upack[i] = ((const device uchar*)uw)[ur + ((packed+i)/4)*uw_strides[2]*4 + (packed+i)%4];
            }
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


# Three-token tiles group picks by expert entirely on-device. The first pick
# owns the group, and scatters each result back to its original top-k slot.
_GROUPED_GATE_UP_PREFIX = r"""
const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = (threadgroup_position_in_grid.x * 4 + sg) * RPS;
    const uint first = threadgroup_position_in_grid.y;
    const uint expert=indices[first];
    const uint matches=counts[first];
    if(matches==0) return;
    const device uint* picks=grouped+first*(TOPK*TILE);
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
            for (uint i=0; i<2*BYTES; ++i) {
                gp[i] = ((const device uchar*)gw)[gr + ((packed+i)/4)*gw_strides[2]*4 + (packed+i)%4];
                upack[i] = ((const device uchar*)uw)[ur + ((packed+i)/4)*uw_strides[2]*4 + (packed+i)%4];
            }
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
                for (uint i=0; i<2*BYTES; ++i) {
                    gp[i] = ((const device uchar*)gw)[gr + ((packed+i)/4)*gw_strides[2]*4 + (packed+i)%4];
                    upack[i] = ((const device uchar*)uw)[ur + ((packed+i)/4)*uw_strides[2]*4 + (packed+i)%4];
                }
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
        header=_SHARED_HEADER,
        ensure_row_contiguous=False,
    )


@cache
def _routed_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="sglang_routed_gate_up_swiglu",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "indices", "limit"],
        output_names=["out"],
        source=_ROUTED_GATE_UP,
        header=_SHARED_HEADER,
        ensure_row_contiguous=False,
    )


def mlx_affine_routed_swiglu(
    x, gate, up, indices, *, bits, group_size, limit=None, grouped=None
):
    """Fuse routed gate/up and rounded activation without duplicating weights.

    Each SIMD owns four outputs. Three-token tiles group actual matching
    experts on-device, sharing packed weights and metadata across their rows.
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
            _group_picks(indices, min(3, x.shape[1]))
            if grouped is None
            else tuple(mx.contiguous(v) for v in grouped)
        )
    return (_grouped_gate_up_kernel() if x.shape[1] > 1 else _routed_gate_up_kernel())(
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
            ("RPS", 2),
            ("HAS_LIMIT", limit is not None),
        ],
        grid=(128 * (n // (8 if x.shape[1] > 1 else 16)), x.shape[1] * topk, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, x.shape[1], topk, n)],
        output_dtypes=[x.dtype],
    )[0]
