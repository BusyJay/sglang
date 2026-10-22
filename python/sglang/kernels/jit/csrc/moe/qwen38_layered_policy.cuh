// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "tvm_ffi_utils.h"

namespace sglang {
// One row: 64-bit signature (split into two int32 words), candidate count,
// selected prefix length, per-token valid/hit counts, then 30 candidate profiles.
constexpr int kLayeredColumns = 47;
struct LayeredHybridArgs {
  const int32_t* original_ids = nullptr;
  const uint16_t* x = nullptr;
  const float* weights = nullptr;
  const int32_t* table = nullptr;
  int rows = 0;
  float* gpu_weights = nullptr;
  int32_t* cpu_ids = nullptr;
  int32_t* descriptor = nullptr;  // [CPU expert count, request sequence]
  int64_t* stats = nullptr;       // [original misses, CPU experts, GPU fills, CPU calls, GPU fill calls]
  uint8_t* page = nullptr;
  int hidden = 0, routes = 0, topk = 0, layer = 0, sequence = 0, forced = -1, slots = 0, overlap = 0;
  bool safe_mask = false;
  float* expert_frequency = nullptr;  // Expert-indexed heat, including CPU routes.
  int max_cpu_experts = 0;
};

/// Select cold nonresident experts without changing this layer's slot ownership.
/// Free slots admit the hottest misses first; resident routes are never victims.
/// Compare against the coldest eligible victim conservatively. If too many cold
/// candidates remain, use the all-GPU branch, rather than extending CPU latency.
__device__ inline int layered_lfu_select(const int32_t* ids, const int32_t* lookup,
    const int32_t* active_ids, int active, int misses, int free, float victim,
    const LayeredHybridArgs& a, int32_t* filtered) {
  for (int j = 0; j < a.routes; ++j) filtered[j] = ids[j];
  if (!a.max_cpu_experts || !misses || misses <= free) return 0;
  int cold[30], count = 0;
  for (int j = 0; j < active; ++j) {
    int e = active_ids[j];
    if (lookup[e] < 0 && a.expert_frequency[e] < victim) cold[count++] = e;
  }
  // Reserving the hottest `free` misses leaves this many cold candidates.
  // Most calls select none or exceed the limit: neither needs a sort.
  int selected = count < misses - free ? count : misses - free;
  if (selected > a.max_cpu_experts || active > a.slots ||
      (a.safe_mask && selected == active)) return 0;
  if (selected && selected < count) {
    for (int j = 1; j < count; ++j) {
      int e = cold[j], p = j;
      while (p && (a.expert_frequency[cold[p-1]] > a.expert_frequency[e] ||
          (a.expert_frequency[cold[p-1]] == a.expert_frequency[e] && cold[p-1] > e))) {
        cold[p] = cold[p-1]; --p;
      }
      cold[p] = e;
    }
  }
  for (int j = 0; j < a.routes; ++j)
    for (int k = 0; k < selected; ++k)
      if (ids[j] == cold[k]) filtered[j] = -1;
  return selected;
}

__device__ inline uint64_t layered_signature(int tokens, int topk, int misses,
                                             int active, int valid_routes, int layer, int slots, int overlap,
                                             const int* valid_counts, const int* hit_counts, const int* profiles) {
  uint64_t h = 1469598103934665603ull;
  int header[] = {tokens, topk, misses, active, valid_routes, layer, slots, overlap};
  for (int v : header) { h ^= static_cast<uint32_t>(v); h *= 1099511628211ull; }
  for (int i = 0; i < tokens; ++i) { h ^= static_cast<uint32_t>(valid_counts[i]); h *= 1099511628211ull; }
  for (int i = 0; i < tokens; ++i) { h ^= static_cast<uint32_t>(hit_counts[i]); h *= 1099511628211ull; }
  for (int i = 0; i < misses; ++i) {
    h ^= static_cast<uint32_t>(profiles[i]); h *= 1099511628211ull;
  }
  return h;
}

__device__ inline int layered_select(const int32_t* ids, const int32_t* lookup,
                                     int experts, int slots, const LayeredHybridArgs& a,
                                     int32_t* filtered) {
  int candidate[30], profiles[30], valid_counts[3] = {}, hit_counts[3] = {};
  const bool bucketed = a.rows && a.table[4] == -1;
  int count = 0, active = 0, valid_routes = 0;
  for (int i = 0; i < a.routes; ++i) {
    int e = ids[i];
    if (e < 0 || e >= experts) { filtered[i] = e; continue; }
    filtered[i] = e;
    ++valid_routes;
    ++valid_counts[i / a.topk];
    if (lookup[e] >= 0) ++hit_counts[i / a.topk];
    bool seen = false;
    for (int j = 0; j < i; ++j) seen |= ids[j] == e;
    if (seen) continue;
    ++active;
    if (lookup[e] >= 0) continue;
    int profile = 0, token_count = 0, route_count = 0;
    for (int t = 0; t < a.routes / a.topk; ++t) {
      int routed = 0;
      for (int j = 0; j < a.topk; ++j)
        routed += ids[t * a.topk + j] == e;
      profile |= routed << (4 * t);
      token_count += routed != 0;
      route_count += routed;
    }
    // Workload buckets use (distinct tokens, routes, expert ID), preserving
    // the legacy exact-mask order for schema 3 and forced diagnostic calls.
    int p = count;
    while (p) {
      int old_tokens = 0, old_routes = 0;
      for (int t = 0; t < a.routes / a.topk; ++t) {
        old_tokens += ((profiles[p-1] >> (4*t)) & 15) != 0;
        old_routes += (profiles[p-1] >> (4*t)) & 15;
      }
      if (old_tokens < token_count ||
          (old_tokens == token_count &&
           ((!bucketed || old_routes == route_count) ? candidate[p-1] < e
                                                    : old_routes < route_count))) break;
      candidate[p] = candidate[p-1]; profiles[p] = profiles[p-1]; --p;
    }
    candidate[p] = e; profiles[p] = profile; ++count;
  }
  int selected = 0;
  if (a.forced >= 0) {
    selected = a.forced < count ? a.forced : count;
  } else if (a.rows) {
    if (bucketed) {
      for (int j = 0; j < count; ++j) {
        int distinct_tokens = 0, routes = 0;
        for (int t = 0; t < a.routes / a.topk; ++t) {
          int n = (profiles[j] >> (4*t)) & 15;
          distinct_tokens += n != 0; routes += n;
        }
        profiles[j] = (distinct_tokens << 5) | routes;
      }
      for (int t = 0; t < 3; ++t) valid_counts[t] = hit_counts[t] = 0;
    }
    const int cost_layer = bucketed ? -1 : a.layer;
    const int cost_slots = bucketed ? -1 : slots;
    uint64_t sig = layered_signature(a.routes / a.topk, a.topk, count, active, valid_routes,
                                     cost_layer, cost_slots, a.overlap, valid_counts, hit_counts, profiles);
    int lo = 0, hi = a.rows;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      const int32_t* row = a.table + mid * kLayeredColumns;
      uint64_t key = static_cast<uint32_t>(row[0]) | (uint64_t(static_cast<uint32_t>(row[1])) << 32);
      if (key < sig) lo = mid + 1; else hi = mid;
    }
    // Hash collisions never authorize a CPU route: compare the full mask key.
    for (int i = lo; i < a.rows; ++i) {
      const int32_t* row = a.table + i * kLayeredColumns;
      uint64_t key = static_cast<uint32_t>(row[0]) | (uint64_t(static_cast<uint32_t>(row[1])) << 32);
      if (key != sig) break;
      if (row[2] != count || row[4] != cost_layer || row[5] != cost_slots ||
          row[42] != a.routes / a.topk || row[43] != a.topk ||
          row[44] != active || row[45] != valid_routes || row[46] != a.overlap) continue;
      bool exact = true;
      for (int j = 0; j < a.routes / a.topk; ++j)
        exact &= row[6+j] == valid_counts[j] && row[9+j] == hit_counts[j];
      for (int j = 0; j < count; ++j) exact &= row[12+j] == profiles[j];
      if (exact) { selected = row[3]; break; }
    }
  }
  if (selected < 0 || selected > count || active > slots) selected = 0;
  // Without a remaining GPU route there is no safe slot for B12X masked
  // routes. Fall back to C=0; this also avoids changing a calibrated prefix.
  if (a.safe_mask && selected == active) selected = 0;
  for (int i = 0; i < a.routes; ++i) {
    int e = ids[i];
    for (int j = 0; j < selected; ++j)
      if (e == candidate[j]) { filtered[i] = -1; break; }
  }
  return selected;
}

__device__ inline void layered_publish(const int32_t* ids, const int32_t* filtered,
                                        const LayeredHybridArgs& a) {
  // Mapped page: head/layer/tokens, BF16 x, int32 CPU IDs, FP32 route weights,
  // FP32 result, done sequence. Same payload layout as the global batch page.
  auto* head = reinterpret_cast<int32_t*>(a.page);
  auto* host_x = reinterpret_cast<uint16_t*>(a.page + 12);
  auto* host_ids = reinterpret_cast<int32_t*>(host_x + 3 * a.hidden);
  auto* host_weights = reinterpret_cast<float*>(host_ids + 3 * a.topk);
  for (int i = threadIdx.x; i < a.hidden * (a.routes / a.topk); i += blockDim.x)
    host_x[i] = a.x[i];
  for (int i = threadIdx.x; i < a.routes; i += blockDim.x) {
    host_ids[i] = filtered[i] < 0 && ids[i] >= 0 ? ids[i] : -1;
    host_weights[i] = a.weights[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    head[1] = a.layer; head[2] = a.routes / a.topk;
    __threadfence_system();
    head[0] = a.sequence;
  }
}

inline LayeredHybridArgs layered_args(tvm::ffi::TensorView ids,
    tvm::ffi::TensorView x, tvm::ffi::TensorView weights,
    tvm::ffi::TensorView table, tvm::ffi::TensorView gpu_weights,
    tvm::ffi::TensorView cpu_ids, tvm::ffi::TensorView descriptor,
    tvm::ffi::TensorView stats, int64_t host_page, int64_t layer,
    int64_t sequence, int64_t forced, int64_t safe_mask, int64_t overlap, int64_t slots) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(weights, dl_float32);
  CHECK_INPUT_AND_TYPE(table, dl_int32);
  CHECK_INPUT_AND_TYPE(gpu_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(cpu_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(descriptor, dl_int32);
  CHECK_INPUT_AND_TYPE(stats, dl_int64);
  CHECK_DIM(2, x); CHECK_DIM(2, weights); CHECK_DIM(2, table);
  CHECK_DIM(2, gpu_weights); CHECK_DIM(2, cpu_ids);
  CHECK_DIM(1, descriptor); CHECK_DIM(1, stats);
  CHECK_CONTIGUOUS(ids); CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(table); CHECK_CONTIGUOUS(gpu_weights);
  CHECK_CONTIGUOUS(cpu_ids); CHECK_CONTIGUOUS(descriptor); CHECK_CONTIGUOUS(stats);
  CHECK_DEVICE(ids, x); CHECK_DEVICE(ids, weights); CHECK_DEVICE(ids, table);
  CHECK_DEVICE(ids, descriptor); CHECK_DEVICE(ids, stats);
  CHECK_DEVICE(ids, gpu_weights); CHECK_DEVICE(ids, cpu_ids);
  TVM_FFI_ICHECK_GE(x.size(0), 1); TVM_FFI_ICHECK_LE(x.size(0), 3);
  TVM_FFI_ICHECK_GT(weights.size(1), 0); TVM_FFI_ICHECK_LE(weights.size(1), 10);
  TVM_FFI_ICHECK_EQ(ids.size(1), weights.size(0) * weights.size(1));
  TVM_FFI_ICHECK_EQ(x.size(0), weights.size(0));
  TVM_FFI_ICHECK_GT(x.size(1), 0);
  TVM_FFI_ICHECK_EQ(cpu_ids.size(0), weights.size(0));
  TVM_FFI_ICHECK_EQ(cpu_ids.size(1), weights.size(1));
  TVM_FFI_ICHECK_EQ(gpu_weights.size(0), weights.size(0));
  TVM_FFI_ICHECK_EQ(gpu_weights.size(1), weights.size(1));
  TVM_FFI_ICHECK_EQ(table.size(1), kLayeredColumns);
  TVM_FFI_ICHECK_LE(table.size(0), 4096);
  TVM_FFI_ICHECK_GE(descriptor.size(0), 2); TVM_FFI_ICHECK_GE(stats.size(0), 5);
  TVM_FFI_ICHECK_GT(host_page, 0);
  TVM_FFI_ICHECK_GE(sequence, 1); TVM_FFI_ICHECK_LT(sequence, 2147483647);
  TVM_FFI_ICHECK_GE(forced, -1); TVM_FFI_ICHECK_LE(forced, 30);
  TVM_FFI_ICHECK(safe_mask == 0 || safe_mask == 1);
  TVM_FFI_ICHECK(overlap == 0 || overlap == 1);
  TVM_FFI_ICHECK_GT(slots, 0);
  return {static_cast<const int32_t*>(ids.data_ptr()),
          static_cast<const uint16_t*>(x.data_ptr()),
          static_cast<const float*>(weights.data_ptr()),
          static_cast<const int32_t*>(table.data_ptr()), static_cast<int>(table.size(0)),
          static_cast<float*>(gpu_weights.data_ptr()),
          static_cast<int32_t*>(cpu_ids.data_ptr()),
          static_cast<int32_t*>(descriptor.data_ptr()), static_cast<int64_t*>(stats.data_ptr()),
          reinterpret_cast<uint8_t*>(host_page), static_cast<int>(x.size(1)),
          static_cast<int>(ids.size(1)), static_cast<int>(weights.size(1)),
          static_cast<int>(layer), static_cast<int>(sequence), static_cast<int>(forced),
          static_cast<int>(slots), static_cast<int>(overlap), static_cast<bool>(safe_mask)};
}
}  // namespace sglang
