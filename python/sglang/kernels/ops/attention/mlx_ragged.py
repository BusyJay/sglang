# SPDX-License-Identifier: Apache-2.0
"""Indexed latent attention, with the MLX singleton matrix reduction order.

Compact score/probability arrays and, for FP8, selected BF16 rows are allocated.
Historical KV remains in shared prefix/append buffers; invalid index-table tails
are never read. The reduction policy
matches MLX 0.32.4 SIMD or M5 MPP matmul, including its TF32 policy.
"""

import os
from functools import cache

import mlx.core as mx


@cache
def _matmul_policy():
    # MLX utils.h latches this integer environment setting, default 1. Like
    # MLX, treat it as a process startup option, not a per-forward toggle.
    architecture = mx.device_info()["architecture"]
    return (
        architecture.startswith("applegpu_g17"),
        int(os.environ.get("MLX_ENABLE_TF32", "1")) != 0,
        architecture.endswith("d"),
    )


# M5 MLX uses MPP tensor operations for low-precision matmul. Match its
# interleaved thin-QK and contiguous split-PV accumulators while loading legal
# cache slots directly into cooperative registers.
_NAX_SCORES = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.y / 2;
    const uint h0 = threadgroup_position_in_grid.y % 2 * 32;
    const uint length = lengths[r];
    uint col = threadgroup_position_in_grid.x * 32;
    if (col >= length) {
        for (uint i = sg * 32 + lane; i < 32 * 32; i += 128) {
            uint p = col + i % 32;
            if (p < WIDTH) scores[(size_t(r) * H + h0 + i / 32) * WIDTH + p] = T(-INFINITY);
        }
        return;
    }
    if (col + 32 > length && length >= 32) col = length - 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
    for (uint i = 0; i < c.get_capacity(); ++i) c[i] = 0;
    for (uint base = sg * 64; base < D; base += 256) {
      for (uint k = base; k < base + 64; k += 32) {
        for (uint i = 0; i < a.get_capacity(); ++i) {
            auto coord = a.get_multidimensional_index(i);
            a[i] = queries[(size_t(r) * H + h0 + coord[1]) * D + k + coord[0]];
        }
        for (uint i = 0; i < b.get_capacity(); ++i) {
            auto coord = b.get_multidimensional_index(i);
            uint p = col + coord[1];
            uint slot = p < length ? indices[size_t(r) * WIDTH + p] : 0;
            b[i] = p < length ? T(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[k + coord[0]]) : T(0);
        }
        op.run(a, b, c);
      }
    }
    threadgroup float partials[3 * 32 * 32];
    if (sg) {
        for (uint i = 0; i < c.get_capacity(); ++i) {
            auto coord = c.get_multidimensional_index(i);
            partials[(sg - 1) * 1024 + coord[1] * 32 + coord[0]] = c[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) {
        for (uint i = 0; i < c.get_capacity(); ++i) {
            auto coord = c.get_multidimensional_index(i);
            float total = c[i];
            for (uint part = 0; part < 3; ++part)
                total += partials[part * 1024 + coord[1] * 32 + coord[0]];
            uint p = col + coord[0];
            // A shifted tail loads a complete MPP tile, but only owns its
            // original output interval. Avoid overlapping threadgroup stores.
            if (p >= threadgroup_position_in_grid.x * 32 && p < length)
                scores[(size_t(r) * H + h0 + coord[1]) * WIDTH + p] = T(total);
        }
    }
"""

_NAX_VALUES = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y / 2;
    const uint h0 = threadgroup_position_in_grid.y % 2 * 32 + (sg / 2) * 16;
    const uint col = threadgroup_position_in_grid.x * 64 + (sg % 2) * 32;
    const uint length = lengths[r];
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto total = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
    auto c = total;
    for (uint i = 0; i < total.get_capacity(); ++i) total[i] = 0;
    uint partition_size = length > 2048 ? 2048 : 1024;
    uint partitions = length > 1024 ? (length + partition_size - 1) / partition_size : 1;
    if (partitions == 1) partition_size = length;
    for (uint part = 0; part < partitions; ++part) {
        for (uint i = 0; i < c.get_capacity(); ++i) c[i] = 0;
        uint end = min(length, (part + 1) * partition_size);
        for (uint k = part * partition_size; k < end; k += 16) {
            for (uint i = 0; i < a.get_capacity(); ++i) {
                auto coord = a.get_multidimensional_index(i);
                uint p = k + coord[0];
                a[i] = p < end ? probs[(size_t(r) * H + h0 + coord[1]) * WIDTH + p] : T(0);
            }
            for (uint i = 0; i < b.get_capacity(); ++i) {
                auto coord = b.get_multidimensional_index(i);
                uint p = k + coord[1];
                uint slot = p < end ? indices[size_t(r) * WIDTH + p] : 0;
                b[i] = p < end ? T(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[col + coord[0]]) : T(0);
            }
            op.run(a, b, c);
        }
        for (uint i = 0; i < c.get_capacity(); ++i) total[i] += c[i];
    }
    for (uint i = 0; i < total.get_capacity(); ++i) {
        auto coord = total.get_multidimensional_index(i);
        out[(size_t(r) * H + h0 + coord[1]) * D + col + coord[0]] = T(total[i]);
    }
"""

_NAX_SCORES_REGULAR = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y / 2;
    const uint h0 = threadgroup_position_in_grid.y % 2 * 32 + (sg / 2) * 16;
    const uint col = threadgroup_position_in_grid.x * 64 + (sg % 2) * 32;
    const uint length = lengths[r];
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto total = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
    auto c = total;
    for (uint i = 0; i < total.get_capacity(); ++i) total[i] = 0;
    const uint partitions = length < 256 ? 2 : 1;
    for (uint part = 0; part < partitions; ++part) {
        for (uint i = 0; i < c.get_capacity(); ++i) c[i] = 0;
        for (uint k = part * (D / partitions); k < (part + 1) * (D / partitions); k += 16) {
            for (uint i = 0; i < a.get_capacity(); ++i) {
                auto coord = a.get_multidimensional_index(i);
                a[i] = queries[(size_t(r) * H + h0 + coord[1]) * D + k + coord[0]];
            }
            for (uint i = 0; i < b.get_capacity(); ++i) {
                auto coord = b.get_multidimensional_index(i);
                uint p = col + coord[1];
                uint slot = p < length ? indices[size_t(r) * WIDTH + p] : 0;
                b[i] = p < length ? T(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[k + coord[0]]) : T(0);
            }
            op.run(a, b, c);
        }
        for (uint i = 0; i < c.get_capacity(); ++i) total[i] += c[i];
    }
    for (uint i = 0; i < total.get_capacity(); ++i) {
        auto coord = total.get_multidimensional_index(i);
        const uint p = col + coord[0];
        if (p < WIDTH)
            scores[(size_t(r) * H + h0 + coord[1]) * WIDTH + p] = p < length ? T(total[i]) : T(-INFINITY);
    }
"""

# MLX D512/H64 singleton collapses heads into matrix rows: SIMD GEMM,
# precise softmax, then regular or split-K SIMD GEMM.
# rather than the vector SDPA kernel. Preserve all three dtype boundaries.
_SCORES = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.y / (H / 32);
    const uint h0 = threadgroup_position_in_grid.y % (H / 32) * 32;
    const uint col = threadgroup_position_in_grid.x * 32;
    const uint length = lengths[r];
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    threadgroup T left[32 * 32];
    threadgroup V right[32 * 32];
    simdgroup_float8x8 total[4], acc[4];
    for (uint i = 0; i < 4; ++i) total[i] = simdgroup_float8x8(0.0f);
    const uint key0 = col + lid / 8;
    const uint key1 = key0 + 16;
    const uint slot0 = key0 < length ? indices[size_t(r) * WIDTH + key0] : 0;
    const uint slot1 = key1 < length ? indices[size_t(r) * WIDTH + key1] : 0;
    const uint raw = (D / 16) / (((H + 31) / 32) * max(1u, (length + 31) / 32));
    uint partitions = 2;
    while (partitions < raw && partitions < 32) partitions *= 2;
    if (length > D) partitions = 1;
    const uint partition_size = D / partitions;
    for (uint part = 0; part < partitions; ++part) {
        for (uint i = 0; i < 4; ++i) acc[i] = simdgroup_float8x8(0.0f);
        const uint end = (part + 1) * partition_size;
        for (uint k = part * partition_size; k < end; k += 32) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < 32 * 8; i += 128) {
                const uint h = i / 8, col_k = (i % 8) * 4;
                const vec<T, 4> v = *((const device vec<T, 4>*)(queries + (size_t(r) * H + h0 + h) * D + k + col_k));
                for (uint j = 0; j < 4; ++j) left[h * 32 + col_k + j] = v[j];
                const uint pos = col + h;
                const uint slot = i < 128 ? slot0 : slot1;
                const vec<V, 4> key = pos < length ? *((const device vec<V, 4>*)(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D) + k + col_k)) : vec<V, 4>(0);
                for (uint j = 0; j < 4; ++j) right[(col_k + j) * 32 + h] = key[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < 32; kk += 8) {
                #pragma clang loop unroll(full)
                for (uint mi = 0; mi < 2; ++mi) {
                    simdgroup_float8x8 aq;
                    for (uint j = 0; j < 2; ++j)
                        aq.thread_elements()[j] = left[((sg / 2) * 8 + mi * 16 + fm) * 32 + kk + fn + j];
                    #pragma clang loop unroll(full)
                    for (uint ni = 0; ni < 2; ++ni) {
                        simdgroup_float8x8 bk;
                        for (uint j = 0; j < 2; ++j)
                            bk.thread_elements()[j] = right[(kk + fm) * 32 + (sg % 2) * 8 + ni * 16 + fn + j];
                        simdgroup_multiply_accumulate(acc[mi * 2 + ni], aq, bk, acc[mi * 2 + ni]);
                    }
                }
            }
        }
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 2; ++j)
                total[i].thread_elements()[j] += acc[i].thread_elements()[j];
    }
    for (uint mi = 0; mi < 2; ++mi)
        for (uint ni = 0; ni < 2; ++ni)
            for (uint j = 0; j < 2; ++j) {
                const uint h = h0 + (sg / 2) * 8 + mi * 16 + fm;
                const uint pos = col + (sg % 2) * 8 + ni * 16 + fn + j;
                if (pos < WIDTH) scores[(size_t(r) * H + h) * WIDTH + pos] = pos < length ? T(total[mi * 2 + ni].thread_elements()[j]) : T(-INFINITY);
            }
"""

_SOFTMAX = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.x / H;
    const uint h = threadgroup_position_in_grid.x % H;
    const uint length = lengths[r];
    if (length <= 4096) {
    const uint simds = (length + 127) / 128;
    threadgroup float maxima[32], totals[32];
    if (sg == 0) { maxima[lane] = -INFINITY; totals[lane] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vals[4], maximum = -MAXFLOAT;
    for (uint i = 0; i < 4; ++i) {
        const uint pos = lid * 4 + i;
        vals[i] = pos < length ? float(scores[(size_t(r) * H + h) * WIDTH + pos]) : -INFINITY;
        maximum = max(maximum, vals[i]);
    }
    maximum = simd_max(maximum);
    if (lane == 0 && sg < simds) maxima[sg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        maximum = simd_max(maxima[lane]);
        if (lane == 0) maxima[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = maxima[0];
    float total = 0;
    for (uint i = 0; i < 4; ++i) {
        vals[i] = fast::exp(vals[i] - maximum);
        total += vals[i];
    }
    total = simd_sum(total);
    if (lane == 0 && sg < simds) totals[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(totals[lane]);
        if (lane == 0) totals[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = length ? 1 / totals[0] : 0;
    for (uint i = 0; i < 4; ++i) {
        const uint pos = lid * 4 + i;
        if (pos < WIDTH)
            probs[(size_t(r) * H + h) * WIDTH + pos] = pos < length ? T(vals[i] * inv) : T(0);
    }
    } else {
        // Match MLX's looped precise softmax for a logical row above 4096.
        // The width bucket is only a storage stride; reduction follows length.
        threadgroup float maxima[32], totals[32];
        float maximum = -MAXFLOAT, total = 0;
        for (uint base = 0; base < length; base += 4096) {
            float values[4];
            for (uint i = 0; i < 4; ++i) {
                uint pos = base + lid * 4 + i;
                values[i] = pos < length ? float(scores[(size_t(r) * H + h) * WIDTH + pos]) : -INFINITY;
            }
            float previous = maximum;
            for (uint i = 0; i < 4; ++i) maximum = maximum < values[i] ? values[i] : maximum;
            total *= fast::exp(previous - maximum);
            for (uint i = 0; i < 4; ++i) total += fast::exp(values[i] - maximum);
        }
        float previous = maximum;
        maximum = simd_max(maximum);
        total *= fast::exp(previous - maximum);
        total = simd_sum(total);
        previous = maximum;
        if (lane == 0) maxima[sg] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = simd_max(maxima[lane]);
        total *= fast::exp(previous - maximum);
        if (lane == 0) totals[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = 1 / simd_sum(totals[lane]);
        for (uint base = 0; base < WIDTH; base += 4096) {
            for (uint i = 0; i < 4; ++i) {
                uint pos = base + lid * 4 + i;
                if (pos < WIDTH)
                    probs[(size_t(r) * H + h) * WIDTH + pos] = pos < length ?
                        T(fast::exp(float(scores[(size_t(r) * H + h) * WIDTH + pos]) - maximum) * total) : T(0);
            }
        }
    }

"""

_VALUES = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.y / (H / 32);
    const uint h0 = threadgroup_position_in_grid.y % (H / 32) * 32;
    const uint col = threadgroup_position_in_grid.x * 32;
    const uint length = lengths[r];
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    threadgroup T left[32 * 32];
    threadgroup V right[32 * 32];
    simdgroup_float8x8 total[4], acc[4];
    for (uint i = 0; i < 4; ++i) total[i] = simdgroup_float8x8(0.0f);
    uint raw = (length / 16) / ((H / 32) * (D / 32));
    uint partitions = 2;
    while (partitions < raw && partitions < 32) partitions *= 2;
    if (length < D) partitions = 1;
    const uint step = partitions == 1 ? REGULAR_BK : 16;
    const uint partition_size = ((length / 16) / partitions) * 16;
    for (uint part = 0; part < partitions; ++part) {
        for (uint i = 0; i < 4; ++i) acc[i] = simdgroup_float8x8(0.0f);
        const uint end = part + 1 == partitions ? ((length + step - 1) / step) * step : (part + 1) * partition_size;
        const uint tail = length % step;
        const uint tiles = (end - part * partition_size + step - 1) / step;
        for (uint it = 0; it < tiles; ++it) {
            // Native regular fused GEMM processes its partial K tile first;
            // split-K processes the remainder after the aligned partitions.
            const uint k = partitions == 1 && tail ?
                (it == 0 ? (length / step) * step : (it - 1) * step) :
                part * partition_size + it * step;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint head = lid / 4, col_k = (lid % 4) * 4;
            for (uint kk = col_k; kk < step; kk += 16) {
                const uint pos = k + kk;
                for (uint j = 0; j < 4; ++j)
                    left[head * 32 + kk + j] = pos + j < length && pos + j < end ? probs[(size_t(r) * H + h0 + head) * WIDTH + pos + j] : T(0);
            }
            for (uint i = lid; i < step * 8; i += 128) {
                const uint key = k + i / 8;
                const uint cv = (i % 8) * 4;
                const uint slot = key < length && key < end ? indices[size_t(r) * WIDTH + key] : 0;
                const vec<V, 4> v = key < length && key < end ? *((const device vec<V, 4>*)(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D) + col + cv)) : vec<V, 4>(0);
                for (uint j = 0; j < 4; ++j) right[(i / 8) * 32 + cv + j] = v[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < step && k + kk < end; kk += 8) {
                #pragma clang loop unroll(full)
                for (uint mi = 0; mi < 2; ++mi) {
                    simdgroup_float8x8 aq;
                    for (uint j = 0; j < 2; ++j)
                        aq.thread_elements()[j] = left[((sg / 2) * 8 + mi * 16 + fm) * 32 + kk + fn + j];
                    #pragma clang loop unroll(full)
                    for (uint ni = 0; ni < 2; ++ni) {
                        simdgroup_float8x8 bk;
                        for (uint j = 0; j < 2; ++j)
                            bk.thread_elements()[j] = right[(kk + fm) * 32 + (sg % 2) * 8 + ni * 16 + fn + j];
                        simdgroup_multiply_accumulate(acc[mi * 2 + ni], aq, bk, acc[mi * 2 + ni]);
                    }
                }
            }
        }
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 2; ++j)
                total[i].thread_elements()[j] += acc[i].thread_elements()[j];
    }
    for (uint mi = 0; mi < 2; ++mi)
        for (uint ni = 0; ni < 2; ++ni)
            for (uint j = 0; j < 2; ++j) {
                const uint h = h0 + (sg / 2) * 8 + mi * 16 + fm;
                const uint pos = col + (sg % 2) * 8 + ni * 16 + fn + j;
                out[(size_t(r) * H + h) * D + pos] = T(total[mi * 2 + ni].thread_elements()[j]);
            }
"""


@cache
def _kernels(nax=False, fp32=False, fp8=False, implicit=False):
    header = (
        "#include <metal_simdgroup_matrix>\n"
        + r"""
        template <typename V>
        inline const device V* glm_latent_row(
            const device V* prefix, const device V* tail,
            uint slot, uint split, uint dim) {
            return slot < split ? prefix + size_t(slot) * dim
                                : tail + size_t(slot - split) * dim;
        }
    """
    )
    if fp8:
        header += r"""
        inline bfloat glm_fp8_decode(uchar code, uchar exponent) {
            ushort value_bits = code & 127;
            ushort bits = (value_bits << 7) | (((value_bits + 1) >> 7) << 14)
                        | (ushort(code & 128) << 8);
            half value = as_type<half>(bits);
            float scale = as_type<float>(
                exponent == 0 ? 0x400000u : uint(exponent) << 23);
            return bfloat((float(value) * 256.0f) * scale);
        }
        inline bfloat glm_fp8_value(
            const device uchar* prefix, const device uchar* tail,
            uint slot, uint split, uint dim, uint col) {
            const device uchar* row = glm_latent_row(
                prefix, tail, slot, split, dim + dim / 32);
            return glm_fp8_decode(row[col], row[dim + col / 32]);
        }
        inline vec<bfloat,4> glm_fp8_vector(
            const device uchar* prefix, const device uchar* tail,
            uint slot, uint split, uint dim, uint col) {
            const device uchar* row = glm_latent_row(
                prefix, tail, slot, split, dim + dim / 32);
            const uchar4 codes = *((const device uchar4*)(row + col));
            const uchar exponent = row[dim + col / 32];
            return vec<bfloat,4>(
                glm_fp8_decode(codes[0],exponent),
                glm_fp8_decode(codes[1],exponent),
                glm_fp8_decode(codes[2],exponent),
                glm_fp8_decode(codes[3],exponent));
        }
        """

    def cache_source(source):
        if implicit:
            for position in ("p", "key0", "key1", "key"):
                source = source.replace(
                    f"indices[size_t(r) * WIDTH + {position}]", position
                )
        if not fp8:
            return source
        source = source.replace(
            "glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[k + coord[0]]",
            "glm_fp8_value(kv,kv_tail,slot,KV_SPLIT,D,k+coord[0])",
        )
        source = source.replace(
            "glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[col + coord[0]]",
            "glm_fp8_value(kv,kv_tail,slot,KV_SPLIT,D,col+coord[0])",
        )
        source = source.replace(
            "*((const device vec<V, 4>*)(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D) + k + col_k))",
            "glm_fp8_vector(kv,kv_tail,slot,KV_SPLIT,D,k+col_k)",
        )
        source = source.replace(
            "*((const device vec<V, 4>*)(glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D) + col + cv))",
            "glm_fp8_vector(kv,kv_tail,slot,KV_SPLIT,D,col+cv)",
        )
        return source

    if nax:
        header += "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    # Context length changes every decode. Keep strides in a tiny device
    # argument instead of compiling a new program for each cache boundary.
    layout = (
        "const uint WIDTH = lengths[ROWS];\nconst uint KV_SPLIT = lengths[ROWS + 1];\n"
    )
    return (
        mx.fast.metal_kernel(
            name=("sglang_ragged_scores_nax" if nax else "sglang_ragged_scores")
            + ("_fp8" if fp8 else "")
            + ("_implicit" if implicit else ""),
            input_names=["queries", "kv", "kv_tail", "indices", "lengths"],
            output_names=["scores", "decoded_kv"] if fp8 and nax else ["scores"],
            source=layout
            + cache_source(
                (_NAX_SCORES_REGULAR if fp32 else _NAX_SCORES) if nax else _SCORES
            ).replace(
                "b[i] = p < length ? T(glm_fp8_value(kv,kv_tail,slot,KV_SPLIT,D,k+coord[0])) : T(0);",
                "b[i] = p < length ? T(glm_fp8_value(kv,kv_tail,slot,KV_SPLIT,D,k+coord[0])) : T(0);\n"
                "if (h0 == 0 && p >= threadgroup_position_in_grid.x*32 && p < length) decoded_kv[(size_t(r)*WIDTH+p)*D+k+coord[0]] = b[i];",
            ),
            header=header,
        ),
        mx.fast.metal_kernel(
            name="sglang_ragged_softmax",
            input_names=["scores", "lengths"],
            output_names=["probs"],
            source=layout + _SOFTMAX,
        ),
        mx.fast.metal_kernel(
            name=("sglang_ragged_values_nax" if nax else "sglang_ragged_values")
            + ("_fp8" if fp8 else "")
            + ("_implicit" if implicit else ""),
            input_names=["probs", "kv", "kv_tail", "indices", "lengths"],
            output_names=["out"],
            source=layout
            + (
                cache_source(
                    _NAX_VALUES.replace(
                        "glm_latent_row(kv, kv_tail, slot, KV_SPLIT, D)[col + coord[0]]",
                        "kv[(size_t(r)*WIDTH+p)*D+col+coord[0]]",
                    )
                )
                if fp8 and nax
                else cache_source(_NAX_VALUES if nax else _VALUES)
            ),
            header=header,
        ),
    )


def mlx_ragged_latent_attention(
    queries, kv, indices, lengths, *, scale, tail=None, split=None, implicit_width=None
):
    """Q [1,64,R,512], shared KV [1,L,512], index table [1,R,W].

    Optional tail follows the sealed prefix without materializing a concat.
    Lengths may be a uint32 device array; split may be a scalar device array.
    Valid indices address both KV segments. Padded table widths keep compiled
    graph layouts stable while the visible lengths and segment split advance.
    With indices=None, implicit_width sets the capacity and positions [0,length)
    are addressed directly, without constructing a redundant index table.
    Invalid index padding is never read. Retains singleton promotion and reduction
    for all table widths, including empty rows. Logical lengths above 4096
    use the native looped softmax reduction with bounded threadgroup size. Other geometries use the
    caller's general batch attention. No per-token SDPA or KV copies occur.
    """
    implicit = indices is None
    if queries.ndim != 4 or kv.ndim != 3 or (not implicit and indices.ndim != 3):
        return None
    batch, heads, rows, dim = queries.shape
    width = implicit_width if implicit else indices.shape[-1]
    if implicit and (not isinstance(width, int) or width < 0):
        raise ValueError("implicit attention requires a nonnegative capacity width")
    dtypes = (mx.float16, mx.bfloat16, mx.float32)
    fp8 = kv.dtype == mx.uint8
    stored_dim = dim + dim // 32 if fp8 else dim
    if (
        batch != 1
        or dim != 512
        or heads != 64
        or not rows
        or kv.shape[0] != batch
        or kv.shape[2] != stored_dim
        or (not implicit and indices.shape[:2] != (batch, rows))
        or len(lengths) != rows
        or (not fp8 and mx.result_type(queries.dtype, kv.dtype) != queries.dtype)
        or queries.dtype not in dtypes
        or (not fp8 and kv.dtype not in dtypes)
        or (fp8 and queries.dtype != mx.bfloat16)
        or (not implicit and indices.dtype not in (mx.int32, mx.uint32))
    ):
        return None
    if tail is None:
        tail = kv
    elif (
        tail.ndim != 3
        or tail.shape[0] != batch
        or tail.shape[2] != stored_dim
        or tail.dtype != kv.dtype
    ):
        return None
    device_lengths = isinstance(lengths, mx.array)
    if device_lengths:
        if lengths.shape != (rows,) or lengths.dtype != mx.uint32:
            raise ValueError("device ragged lengths must be uint32 [rows]")
    elif any(length < 0 or length > width for length in lengths):
        raise ValueError("ragged lengths must be within the index table width")
    if not width or (not device_lengths and not any(lengths)):
        return mx.zeros_like(queries)
    prefix_size = kv.shape[1]
    split = prefix_size if split is None else split
    # Empty MLX inputs use constant pointers, but both row-loader arguments
    # require device pointers. Alias the nonempty segment for the unreachable
    # argument, preserving the original split and every valid slot address.
    if not prefix_size:
        kv = tail
    if not tail.shape[1]:
        tail = kv
    template = [
        ("T", queries.dtype),
        ("V", queries.dtype if fp8 else kv.dtype),
        ("H", heads),
        ("D", dim),
        ("ROWS", rows),
        (
            "REGULAR_BK",
            32 if queries.dtype == mx.float32 and _matmul_policy()[2] else 16,
        ),
    ]
    lens = mx.concatenate(
        [
            lengths if device_lengths else mx.array(lengths, mx.uint32),
            mx.array([width], mx.uint32),
            split.reshape(1)
            if isinstance(split, mx.array)
            else mx.array([split], mx.uint32),
        ]
    )
    has_nax, tf32, _ = _matmul_policy()
    nax = has_nax and (queries.dtype != mx.float32 or tf32)
    scores_kernel, softmax_kernel, values_kernel = _kernels(
        nax, nax and queries.dtype == mx.float32, fp8, implicit
    )
    if implicit:
        indices = mx.array([0], mx.int32)
    scaled_q = (queries * mx.array(scale, queries.dtype)).transpose(0, 2, 1, 3)
    score_outputs = scores_kernel(
        inputs=[scaled_q, kv, tail, indices, lens],
        template=template,
        grid=(
            128
            * (
                (width + (64 if nax and queries.dtype == mx.float32 else 32) - 1)
                // (64 if nax and queries.dtype == mx.float32 else 32)
            ),
            rows * (heads // 32),
            1,
        ),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, heads, width), (rows, width, dim)]
        if fp8 and nax
        else [(rows, heads, width)],
        output_dtypes=[queries.dtype, queries.dtype]
        if fp8 and nax
        else [queries.dtype],
    )
    scores = score_outputs[0]
    value_kv = score_outputs[1] if fp8 and nax else kv
    threads = min(1024, 32 * ((width + 127) // 128))
    probs = softmax_kernel(
        inputs=[scores, lens],
        template=template,
        grid=(threads * rows * heads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[scores.shape],
        output_dtypes=[queries.dtype],
    )[0]
    out = values_kernel(
        inputs=[probs, value_kv, value_kv if fp8 and nax else tail, indices, lens],
        template=template,
        grid=(128 * (dim // (64 if nax else 32)), rows * (heads // 32), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, rows, heads, dim)],
        output_dtypes=[queries.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)
