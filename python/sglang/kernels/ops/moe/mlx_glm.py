# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx decode_kernels.py, commit 5c6cbd6314ca.
# Kernel arithmetic originally follows Apple's MLX Metal implementations.
"""MLX sigmoid routing and affine expert-down / weighted-combine kernels."""

from functools import cache

import mlx.core as mx

_HEADER = r"""

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int bits>
constexpr int glm_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
constexpr int glm_bytes_per_pack() {
  return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
}

// Verbatim copy of MLX quantized.h load_vector (U = float).
template <typename T, int values_per_thread, int bits>
inline float glm_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  } else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Verbatim copy of MLX quantized.h qdot (U = float).
template <int values_per_thread, int bits>
inline float glm_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;
      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;
      accum += (w[0] & 0x3f) * x_thread[0];
      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);
      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// qmv_fast_impl for RPS consecutive rows of one [N, K] affine matrix
// (row pointers already offset to the first row), one simdgroup.  Leaves the
// per-lane partial sums in `result`; the caller simd_sums them.
template <typename T, int K, int group_size, int bits, int RPS>
inline void glm_qmv_rows(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = glm_pack_factor<bits>();
  constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  ws += simd_lid * packs_per_thread * bytes_per_pack;
  scales += simd_lid / scale_step_per_thread;
  biases += simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < K; k += block_size) {
    float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
}

template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

"""

_LOGITS = r"""

  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  // One row per simdgroup: MLX's gemv gives each thread TM = 4 rows, but
  // every row's per-lane products and shuffle ladder are independent of TM.
  constexpr int RPS = ROWS_PER_SIMD;
  const int out_row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RPS;
  if (out_row >= E) {
    return;
  }
  const device auto* mat = w + size_t(out_row) * K;
  const device T* xv = x + size_t(tok) * K;
  float result[RPS];
  for (int tm = 0; tm < RPS; tm++) {
    result[tm] = 0.0f;
  }
  int bn = int(lane) * 4;
  for (int i = 0; i < K / 128; ++i) {
    float v_coeff[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(xv[bn + tn]);
    }
    int mat_offset = 0;
    for (int tm = 0; tm < RPS; tm++) {
      float inter[4];
      for (int tn = 0; tn < 4; tn++) {
        inter[tn] = static_cast<float>(mat[mat_offset + bn + tn]);
      }
      for (int tn = 0; tn < 4; tn++) {
        result[tm] += inter[tn] * v_coeff[tn];
      }
      mat_offset += K;
    }
    bn += 128;
  }
  for (int tm = 0; tm < RPS; tm++) {
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result[tm] += simd_shuffle_down(result[tm], sn);
    }
  }
  if (lane == 0) {
    for (int tm = 0; tm < RPS; tm++) {
      const int e = out_row + tm;
      float sgm = glm_sigmoid<float>(result[tm]);
      float biased = sgm + bias[e];
      sig[size_t(tok) * E + e] = sgm;
      biased_out[size_t(tok) * E + e] = biased;
    }
  }

"""

_SELECT = r"""

  const uint lane = thread_index_in_simdgroup;
  const int tok = int(threadgroup_position_in_grid.x);
  constexpr int PER = (E + 31) / 32;
  const device float* bz = biased + size_t(tok) * E;
  const device float* sz = sig + size_t(tok) * E;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  int picked[TOPK];
  for (int r = 0; r < TOPK; r++) {
    // Best remaining candidate of this lane: highest value, lowest index.
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      // Only NaNs remain (uniform branch): argpartition's sort places them
      // after every number, lowest index first.
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
  if (lane == 0) {
    float total = 0.0f;
    float gathered[TOPK];
    for (int r = 0; r < TOPK; r++) {
      gathered[r] = sz[picked[r]];
      total = gathered[r] + total;
    }
    for (int r = 0; r < TOPK; r++) {
      float q = NORM ? gathered[r] / (total + 1e-20f) : gathered[r];
      float s = q * scaling[0];
      indices[tok * TOPK + r] = uint(picked[r]);
      scores[tok * TOPK + r] = s;
    }
  }

"""

# Keep the native FP32 logits reduction; fuse its routing consumers only.
_SELECT_LOGITS = r"""
  const uint lane = thread_index_in_simdgroup;
  const int tok = int(threadgroup_position_in_grid.x);
  constexpr int PER = (E + 31) / 32;
  const device float* gz = logits + size_t(tok) * E;
  float vals[PER], probabilities[PER];
  bool taken[PER];
  float local_best = -INFINITY;
  int local_e = 0x7fffffff;
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    probabilities[j] = e < E ? glm_sigmoid<float>(gz[e]) : 0.0f;
    vals[j] = e < E ? probabilities[j] + bias[e] : -INFINITY;
    taken[j] = e >= E;
    if (!taken[j] && !isnan(vals[j]) &&
        (local_e == 0x7fffffff || vals[j] > local_best ||
         (vals[j] == local_best && e < local_e))) {
      local_best = vals[j];
      local_e = e;
    }
  }
  int picked[TOPK];
  float gathered[TOPK];
  for (int r = 0; r < TOPK; r++) {
    float best = local_best;
    int best_e = local_e;
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      if (oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e))) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) best_e = e;
      }
      for (ushort off = 16; off >= 1; off >>= 1)
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
    }
    picked[r] = best_e;
    // The selected owner already has the exact, unbiased probability.
    // Broadcast it without a second sigmoid or read from the logits buffer.
    float probability = probabilities[best_e / 32];
    gathered[r] = simd_shuffle(probability, ushort(best_e % 32));
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
      local_best = -INFINITY;
      local_e = 0x7fffffff;
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && !isnan(vals[j]) &&
            (local_e == 0x7fffffff || vals[j] > local_best ||
             (vals[j] == local_best && e < local_e))) {
          local_best = vals[j];
          local_e = e;
        }
      }
    }
  }
  if (lane == 0) {
    float total = 0.0f;
    for (int r = 0; r < TOPK; r++) total = gathered[r] + total;
    for (int r = 0; r < TOPK; r++) {
      float q = NORM ? gathered[r] / (total + 1e-20f) : gathered[r];
      indices[tok * TOPK + r] = uint(picked[r]);
      scores[tok * TOPK + r] = q * scaling[0];
    }
  }

"""


_DOWN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * 4;
    const int tok = int(threadgroup_position_in_grid.y);
    float acc[4] = {0};
    for (int r = 0; r < TOPK; r++) {
        const size_t base = size_t(indices[tok * TOPK + r]) * N + row;
        float value[4] = {0};
        glm_qmv_rows<T, K, GS, BITS, 4>(
            (const device uint8_t*)w + base * (K * BITS / 8),
            s + base * (K / GS), b + base * (K / GS),
            act + (size_t(tok) * TOPK + r) * K, lane, value);
        for (int j = 0; j < 4; j++) {
            // The local FFN output is rounded to T, restored to R, then
            // multiplied in FP32 before the separate routing reduction.
            float v = float(R(T(simd_sum(value[j]))));
            volatile float weighted = v * scores[tok * TOPK + r];
            acc[j] += weighted;
        }
    }
    if (lane == 0) {
        for (int j = 0; j < 4; j++) {
            out[size_t(tok) * N + row + j] = R(acc[j]) + shared[size_t(tok) * N + row + j];
        }
    }
"""

_DOWN_PARALLEL = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint expert = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x * 4;
    const uint tok = threadgroup_position_in_grid.y;
    const size_t base = size_t(indices[tok * TOPK + expert]) * N + row;
    float value[4] = {0};
    glm_qmv_rows<T, K, GS, BITS, 4>(
        (const device uint8_t*)w + base * (K * BITS / 8),
        s + base * (K / GS), b + base * (K / GS),
        act + (size_t(tok) * TOPK + expert) * K, lane, value);
    threadgroup float weighted[TOPK * 4];
    for (uint j = 0; j < 4; ++j) {
        float v = float(R(T(simd_sum(value[j]))));
        volatile float product = v * scores[tok * TOPK + expert];
        if (lane == 0) weighted[expert * 4 + j] = product;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (expert == 0 && lane < 4) {
        float total = 0;
        // Keep the original routing order and both local-output casts.
        for (uint r = 0; r < TOPK; ++r) total += weighted[r * 4 + lane];
        out[size_t(tok) * N + row + lane] =
            R(total) + shared[size_t(tok) * N + row + lane];
    }
"""


# Group experts across up to three requests/verification rows. Every expert
# computes independently; matching experts share the memory-system cache
# without serializing per-row qdots or allocating intermediate projections.
_DOWN_BATCH_PARALLEL = (
    _DOWN_PARALLEL.replace(
        "const uint expert = simdgroup_index_in_threadgroup;",
        "const uint group = simdgroup_index_in_threadgroup;\n"
        "    const uint expert = group % TOPK;",
    )
    .replace(
        "const uint tok = threadgroup_position_in_grid.y;",
        "const uint tok = threadgroup_position_in_grid.y * TILE + group / TOPK;",
    )
    .replace(
        "indices[tok * TOPK + expert]",
        "indices[min(tok, uint(COUNT-1)) * TOPK + expert]",
    )
    .replace(
        "(size_t(tok) * TOPK + expert) * K",
        "(size_t(min(tok, uint(COUNT-1))) * TOPK + expert) * K",
    )
    .replace(
        "threadgroup float weighted[TOPK * 4]",
        "threadgroup float weighted[TILE * TOPK * 4]",
    )
    .replace(
        "scores[tok * TOPK + expert]",
        "scores[min(tok, uint(COUNT-1)) * TOPK + expert]",
    )
    .replace("weighted[expert * 4 + j]", "weighted[group * 4 + j]")
    .replace(
        "if (expert == 0 && lane < 4)",
        "if (expert == 0 && lane < 4 && tok < COUNT)",
    )
    .replace(
        "weighted[r * 4 + lane]", "weighted[((group / TOPK) * TOPK + r) * 4 + lane]"
    )
)


@cache
def _batch_parallel_down_kernel():
    return mx.fast.metal_kernel(
        name="sglang_moe_batch_parallel_down",
        input_names=["act", "indices", "scores", "w", "s", "b", "shared"],
        output_names=["out"],
        source=_DOWN_BATCH_PARALLEL,
        header=_HEADER,
    )


_GROUP_PICKS = r"""
    const uint first=thread_position_in_grid.x;
    if(first>=COUNT*TOPK) return;
    const uint begin=(first/(TOPK*TILE))*(TOPK*TILE);
    const uint end=min(begin+uint(TOPK*TILE),uint(COUNT*TOPK));
    uint count=0;
    const uint expert=indices[first];
    bool owner=true;
    for(uint p=begin;p<first;++p) if(indices[p]==expert) owner=false;
    if(owner) {
        for(uint p=first;p<end;++p)
            if(indices[p]==expert) picks[first*(TOPK*TILE)+count++]=p;
    }
    counts[first]=count;
"""


@cache
def _group_picks_kernel():
    return mx.fast.metal_kernel(
        name="sglang_moe_group_picks",
        input_names=["indices"],
        output_names=["picks", "counts"],
        source=_GROUP_PICKS,
    )


def _group_picks(indices, tile):
    count, topk = indices.shape[1:]
    return _group_picks_kernel()(
        inputs=[indices],
        template=[("COUNT", count), ("TOPK", topk), ("TILE", tile)],
        grid=(count * topk, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(count * topk, topk * tile), (count * topk,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )


@cache
def _kernel(kind):
    if kind == "logits":
        inputs, outputs, source = ["x", "w", "bias"], ["sig", "biased_out"], _LOGITS
    elif kind == "select_logits":
        inputs = ["logits", "bias", "scaling"]
        outputs, source = ["indices", "scores"], _SELECT_LOGITS
    elif kind == "select":
        inputs = ["sig", "biased", "scaling"]
        outputs, source = ["indices", "scores"], _SELECT
    else:
        inputs = ["act", "indices", "scores", "w", "s", "b", "shared"]
        outputs, source = ["out"], _DOWN_PARALLEL if kind == "down_parallel" else _DOWN
    return mx.fast.metal_kernel(
        name="sglang_moe_" + kind,
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_HEADER,
    )


def mlx_glm_router_select(logits, bias, *, top_k, scaling, renormalize):
    """Corrected sigmoid selection and FP32 routing weights in one dispatch.

    Leading dimensions are independent rows. Keep logits and bias in FP32,
    bias only the selection scores, and normalize the original probabilities.
    """
    if (
        logits.ndim < 1
        or logits.dtype != mx.float32
        or bias.dtype != mx.float32
        or bias.shape != (logits.shape[-1],)
        or not 1 <= top_k <= min(32, logits.shape[-1])
        or not 1 <= logits.shape[-1] <= 1024
    ):
        return None
    experts = logits.shape[-1]
    count = logits.size // experts
    if count == 0:
        return None
    shape = (*logits.shape[:-1], top_k)
    return tuple(
        _kernel("select_logits")(
            inputs=[logits, bias, mx.array([scaling], mx.float32)],
            template=[
                ("E", experts),
                ("TOPK", top_k),
                ("NORM", renormalize and top_k > 1),
            ],
            grid=(count * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[mx.uint32, mx.float32],
        )
    )


def mlx_glm_router(x, weight, bias, *, top_k, scaling, renormalize):
    """Sigmoid GEMV, corrected top-k and FP32 weights for 1..17 rows."""
    if x.ndim != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 17 or weight.ndim != 2:
        return None
    count = x.shape[1]
    experts, dim = weight.shape
    if (
        x.shape[-1] != dim
        or x.dtype not in (mx.bfloat16, mx.float16)
        or weight.dtype != mx.float32
        or bias.dtype != mx.float32
        or bias.shape != (experts,)
        or not 16 <= experts <= 1024
        or experts % 16
        or dim % 128
        or dim <= 64
        or dim >= 16 * experts
        or not 1 <= top_k <= min(32, experts)
    ):
        return None
    sig, biased = _kernel("logits")(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("K", dim), ("E", experts), ("ROWS_PER_SIMD", 1)],
        grid=(128 * (experts // 4), count, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, experts)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    indices, scores = _kernel("select")(
        inputs=[sig, biased, mx.array([scaling], mx.float32)],
        template=[
            ("E", experts),
            ("TOPK", top_k),
            ("NORM", int(renormalize and top_k > 1)),
        ],
        grid=(32 * count, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, count, top_k)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, scores


def mlx_glm_down_combine(
    act, indices, scores, weight, scales, biases, shared, *, bits, group_size
):
    """Affine expert down + FP32 weighted reduction + shared output; else None."""
    if (
        act.ndim != 4
        or act.shape[0] != 1
        or not 1 <= act.shape[1] <= 17
        or weight.ndim != 3
    ):
        return None
    _, count, top_k, dim = act.shape
    experts, out_dim, packed = weight.shape
    if (
        bits not in (4, 5, 6, 8)
        or group_size not in (32, 64, 128)
        or act.dtype not in (mx.bfloat16, mx.float16)
        or shared.dtype not in (mx.bfloat16, mx.float16)
        or packed * 32 != dim * bits
        or dim % (32 * (8 if bits == 5 else 4 if bits == 6 else 32 // bits) * 2)
        or dim % group_size
        or out_dim % 16
        or not 1 <= top_k <= 32
        or scales.shape != (experts, out_dim, dim // group_size)
        or biases.shape != scales.shape
        or scales.dtype != act.dtype
        or biases.dtype != act.dtype
        or weight.dtype != mx.uint32
        or shared.shape != (1, count, out_dim)
        or scores.dtype != mx.float32
        or indices.dtype not in (mx.uint32, mx.int32)
        or indices.shape != (1, count, top_k)
        or scores.shape != indices.shape
    ):
        return None
    if count > 1 and top_k <= 8:
        # One request per threadgroup keeps eight expert SIMD groups resident
        # without the register/occupancy cost of a 24-SIMD three-row group.
        # Requests remain parallel in grid.y; combine order is unchanged.
        tile = 1
        threads = 32 * top_k * tile
        return _batch_parallel_down_kernel()(
            inputs=[act, indices, scores, weight, scales, biases, shared],
            template=[
                ("T", act.dtype),
                ("R", shared.dtype),
                ("K", dim),
                ("N", out_dim),
                ("TOPK", top_k),
                ("BITS", bits),
                ("GS", group_size),
                ("COUNT", count),
                ("TILE", tile),
            ],
            grid=(threads * (out_dim // 4), (count + tile - 1) // tile, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[shared.shape],
            output_dtypes=[shared.dtype],
        )[0]
    # Independent experts compute concurrently; the final four lanes retain
    # the original routing reduction order. Small top-k keeps native geometry.
    parallel = count == 1 and top_k >= 4 and out_dim >= 1024
    threads = 32 * top_k if parallel else 128
    rows = 4 if parallel else 16
    return _kernel("down_parallel" if parallel else "down")(
        inputs=[act, indices, scores, weight, scales, biases, shared],
        template=[
            ("T", act.dtype),
            ("R", shared.dtype),
            ("K", dim),
            ("N", out_dim),
            ("TOPK", top_k),
            ("BITS", bits),
            ("GS", group_size),
        ],
        grid=(threads * (out_dim // rows), count, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[shared.dtype],
    )[0]
