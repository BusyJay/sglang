// SPDX-License-Identifier: Apache-2.0
#include "qwen38_layered_policy.cuh"
#include "tvm_ffi_utils.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {
// NVFP4/SM120 policy, indexed by miss count. Use three CTAs across 5-7 misses;
// counts above this table retain eight CTAs.
__device__ __constant__ int kMailboxCtasByMiss[11] = {8, 8, 5, 4, 4, 3, 3, 3, 2, 2, 1};
template <bool Adaptive = false, bool Tiered = false, bool Candidate = false>
__global__ void
mailbox_fill_kernel(const int32_t* plan, const int64_t* layout, int tensors, const volatile int32_t* failed) {
  // The planner and fill run on the same stream. Immutable registered source
  // addresses need no CPU publication or acknowledgement before they are read.
  if (plan[0] || plan[3]) {
    if (threadIdx.x == 0) asm volatile("trap;");
    return;
  }
  int miss = blockIdx.x;
  if (miss >= plan[1]) return;
  int ctas = gridDim.y;
  if constexpr (Adaptive) {
    ctas = plan[1] <= 10 ? kMailboxCtasByMiss[plan[1]] : 8;
    // Only change independently scanned buckets. Unmeasured verify union
    // counts retain the baseline, including 11..19 and 21..29.
    if constexpr (Candidate) {
      if (plan[1] == 3)
        ctas = 5;
      else if (plan[1] == 4)
        ctas = 3;
      else if (plan[1] == 20 || plan[1] == 30)
        ctas = 1;
    }
    if (blockIdx.y >= ctas) return;
  }
  // Keep all-hit fills entirely on device. Only one thread per miss block
  // reads the mapped failure flag; the barrier guards every source read.
  if (threadIdx.x == 0 && *failed) asm volatile("trap;");
  __syncthreads();
  int slot = plan[5 + 2 * miss];
  int expert = plan[4 + 2 * miss];
  constexpr int columns = Tiered ? 4 : 3;
  for (int tensor = 0; tensor < tensors; ++tensor) {
    int64_t bytes = layout[columns * tensor + 1];
    auto* destination = reinterpret_cast<uint8_t*>(layout[columns * tensor]) + slot * bytes;
    const uint8_t* source =
        reinterpret_cast<const uint8_t*>(layout[columns * tensor + 2]) + static_cast<int64_t>(expert) * bytes;
    int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * ctas;
    if ((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes) % 16 == 0) {
      for (; i < bytes / 16; i += stride) {
        uint4 value;
        asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                     : "l"(source + 16 * i)
                     : "memory");
        if constexpr (Tiered) {
          if (i == 62 && layout[columns * tensor + 3]) {
            const uint64_t offset = *reinterpret_cast<const volatile uint64_t*>(source + 976);
            const uint64_t pointer = offset ? offset + layout[columns * tensor + 3] : 0;
            value.x = static_cast<uint32_t>(pointer);
            value.y = static_cast<uint32_t>(pointer >> 32);
          }
        }
        reinterpret_cast<uint4*>(destination)[i] = value;
      }
    } else {
      for (; i < bytes; i += stride)
        destination[i] = reinterpret_cast<const volatile uint8_t*>(source)[i];
    }
  }
}
}  // namespace

// Register the pack's single private source mapping; all transports share it.
struct Qwen38MailboxRegister {
  static int64_t run(TensorView bytes) {
    TVM_FFI_ICHECK_EQ(bytes.device().device_type, kDLCPU);
    TVM_FFI_ICHECK_EQ(bytes.ndim(), 1);
    CHECK_CONTIGUOUS(bytes);
    CHECK_INPUT_TYPE(bytes, dl_uint8);
    TVM_FFI_ICHECK_GT(bytes.size(0), 0);
    unsigned int flags = cudaHostRegisterMapped;
    const auto register_start = std::chrono::steady_clock::now();
    auto status = cudaHostRegister(bytes.data_ptr(), bytes.size(0), flags);
    const auto register_end = std::chrono::steady_clock::now();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "register expert mmap: " << cudaGetErrorString(status);
    void* mapped = nullptr;
    status = cudaHostGetDevicePointer(&mapped, bytes.data_ptr(), 0);
    const auto pointer_end = std::chrono::steady_clock::now();
    const char* observation = std::getenv("SGLANG_QWEN38_OBSERVE");
    if (!observation || observation[0] != '0') {
      std::fprintf(
          stderr,
          "Qwen38Register bytes=%lld register_ms=%.3f device_pointer_ms=%.3f status=%d\n",
          static_cast<long long>(bytes.size(0)),
          std::chrono::duration<double, std::milli>(register_end - register_start).count(),
          std::chrono::duration<double, std::milli>(pointer_end - register_end).count(),
          static_cast<int>(status));
    }
    if (status != cudaSuccess) cudaHostUnregister(bytes.data_ptr());
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    return reinterpret_cast<int64_t>(mapped);
  }
};
struct Qwen38MailboxUnregister {
  static void run(TensorView bytes) {
    TVM_FFI_ICHECK_EQ(bytes.device().device_type, kDLCPU);
    auto status = cudaHostUnregister(bytes.data_ptr());
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

struct Qwen38MailboxFill {
  static void run(TensorView plan, TensorView layout, int64_t top_k, int64_t failed) {
    run_config(plan, layout, top_k, failed, 8);
  }
  static void run_config(TensorView plan, TensorView layout, int64_t top_k, int64_t failed, int64_t grid_y) {
    // -1 selects the opt-in historical candidate; 0 is baseline adaptive.
    TVM_FFI_ICHECK(grid_y >= -1 && grid_y <= 64);
    CHECK_INPUT_AND_TYPE(plan, dl_int32);
    CHECK_INPUT_AND_TYPE(layout, dl_int64);
    CHECK_DIM(1, plan);
    CHECK_DIM(2, layout);
    CHECK_DEVICE(plan, layout);
    TVM_FFI_ICHECK(layout.size(1) == 3 || layout.size(1) == 4);
    TVM_FFI_ICHECK_GT(failed, 0);
    TVM_FFI_ICHECK_GE(plan.size(0), 4);
    TVM_FFI_ICHECK_GT(top_k, 0);
    TVM_FFI_ICHECK_LE(top_k, 32);
    if (layout.size(1) == 4 && grid_y == -1) {
      mailbox_fill_kernel<true, true, true><<<dim3(top_k, 8), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    } else if (grid_y == -1) {
      mailbox_fill_kernel<true, false, true><<<dim3(top_k, 8), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    } else if (layout.size(1) == 4 && grid_y == 0) {
      mailbox_fill_kernel<true, true><<<dim3(top_k, 8), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    } else if (layout.size(1) == 4) {
      mailbox_fill_kernel<false, true><<<dim3(top_k, grid_y), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    } else if (grid_y == 0) {
      mailbox_fill_kernel<true><<<dim3(top_k, 8), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    } else {
      mailbox_fill_kernel<false><<<dim3(top_k, grid_y), 256, 0, get_stream(plan.device())>>>(
          static_cast<const int32_t*>(plan.data_ptr()),
          static_cast<const int64_t*>(layout.data_ptr()),
          layout.size(0),
          reinterpret_cast<const volatile int32_t*>(failed));
    }
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
};

namespace sglang {
namespace {
__global__ void
csf_headers_kernel(uint8_t* heads, int32_t count, int64_t stride, int64_t delta, uint64_t lower, uint64_t upper) {
  const int32_t expert = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert >= count) return;
  uint8_t* head = heads + static_cast<int64_t>(expert) * stride;
  const uint64_t offset = *reinterpret_cast<const uint64_t*>(head + 976);
  if (offset >= lower && offset < upper) *reinterpret_cast<uint64_t*>(head + 992) = offset + delta;
}
}  // namespace

/// Bind immutable exception banks after prefill uploads; decode fills bind in-place.
struct Qwen38CsfHeaders {
  static void run(TensorView heads, int64_t delta, int64_t lower, int64_t upper) {
    CHECK_INPUT_AND_TYPE(heads, dl_uint8);
    CHECK_DIM(2, heads);
    CHECK_CONTIGUOUS(heads);
    TVM_FFI_ICHECK_GE(heads.size(1), 1024);
    TVM_FFI_ICHECK_EQ(heads.size(1) % 16, 0);
    TVM_FFI_ICHECK_GE(lower, 0);
    TVM_FFI_ICHECK_GE(upper, lower);
    // delta is device_address - file_offset. CUDA VMM allocators may return
    // low virtual addresses, so a valid bank can have a negative delta.
    TVM_FFI_ICHECK_GE(delta, -lower);
    if (!heads.size(0) || lower == upper) return;
    csf_headers_kernel<<<(heads.size(0) + 127) / 128, 128, 0, get_stream(heads.device())>>>(
        static_cast<uint8_t*>(heads.data_ptr()),
        static_cast<int32_t>(heads.size(0)),
        heads.size(1),
        delta,
        lower,
        upper);
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
};
}  // namespace sglang

namespace {
__global__ void alpha_scatter_kernel(
    const float* g1, const float* g2, float* slot_g1, float* slot_g2, const int32_t* plan, int experts, int slots) {
  if (plan[0] || plan[3]) return;
  for (int i = threadIdx.x; i < plan[1] && i < slots; i += blockDim.x) {
    int expert = plan[4 + 2 * i], slot = plan[5 + 2 * i];
    if (expert >= 0 && expert < experts && slot >= 0 && slot < slots) {
      slot_g1[slot] = g1[expert];
      slot_g2[slot] = g2[expert];
    }
  }
}
}  // namespace

struct Qwen38CacheAlphaScatter {
  static void run(TensorView g1, TensorView g2, TensorView slot_g1, TensorView slot_g2, TensorView plan) {
    CHECK_INPUT_AND_TYPE(g1, dl_float32);
    CHECK_INPUT_AND_TYPE(g2, dl_float32);
    CHECK_INPUT_AND_TYPE(slot_g1, dl_float32);
    CHECK_INPUT_AND_TYPE(slot_g2, dl_float32);
    CHECK_INPUT_AND_TYPE(plan, dl_int32);
    CHECK_DIM(1, g1);
    CHECK_DIM(1, g2);
    CHECK_DIM(1, slot_g1);
    CHECK_DIM(1, slot_g2);
    CHECK_DIM(1, plan);
    CHECK_DEVICE(g1, g2);
    CHECK_DEVICE(g1, slot_g1);
    CHECK_DEVICE(g1, slot_g2);
    CHECK_DEVICE(g1, plan);
    TVM_FFI_ICHECK_GT(g1.size(0), 0);
    TVM_FFI_ICHECK_GT(slot_g1.size(0), 0);
    TVM_FFI_ICHECK_LE(slot_g1.size(0), g1.size(0));
    TVM_FFI_ICHECK_EQ(g1.size(0), g2.size(0));
    TVM_FFI_ICHECK_EQ(slot_g1.size(0), slot_g2.size(0));
    TVM_FFI_ICHECK(plan.size(0) == 4 + 3 * slot_g1.size(0) || plan.size(0) == 4 + 4 * slot_g1.size(0));
    alpha_scatter_kernel<<<1, 128, 0, get_stream(g1.device())>>>(
        static_cast<const float*>(g1.data_ptr()),
        static_cast<const float*>(g2.data_ptr()),
        static_cast<float*>(slot_g1.data_ptr()),
        static_cast<float*>(slot_g2.data_ptr()),
        static_cast<const int32_t*>(plan.data_ptr()),
        g1.size(0),
        slot_g1.size(0));
    auto status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

namespace {
template <typename T>
__global__ void scatter_kernel(
    const T* cache, T* arena, const int32_t* keys, const int32_t* counts, int slots, int experts, int64_t row_bytes) {
  int64_t size = static_cast<int64_t>(slots) * row_bytes;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int slot = i / row_bytes;
    int expert = keys[slot];
    if (expert >= 0 && expert < experts && counts[expert])
      arena[static_cast<int64_t>(expert) * row_bytes + i % row_bytes] = cache[i];
  }
}
}  // namespace

struct Qwen38CacheScatter {
  static void run(TensorView cache, TensorView arena, TensorView keys, TensorView counts) {
    CHECK_INPUT_AND_TYPE(cache, dl_uint8);
    CHECK_INPUT_AND_TYPE(arena, dl_uint8);
    CHECK_INPUT_AND_TYPE(keys, dl_int32);
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_DIM(2, cache);
    CHECK_DIM(2, arena);
    CHECK_DIM(1, keys);
    CHECK_DIM(1, counts);
    CHECK_DEVICE(cache, arena);
    CHECK_DEVICE(cache, keys);
    CHECK_DEVICE(cache, counts);
    TVM_FFI_ICHECK_EQ(cache.size(0), keys.size(0));
    TVM_FFI_ICHECK_EQ(cache.size(1), arena.size(1));
    TVM_FFI_ICHECK_EQ(counts.size(0), arena.size(0) + 1);
    int64_t size = cache.size(0) * cache.size(1);
    if (!size) return;
    int blocks = static_cast<int>(std::min<int64_t>((size + 255) / 256, 65535));
    if (cache.size(1) % 16 == 0) {
      size /= 16;
      blocks = static_cast<int>(std::min<int64_t>((size + 255) / 256, 65535));
      scatter_kernel<<<blocks, 256, 0, get_stream(cache.device())>>>(
          static_cast<const uint4*>(cache.data_ptr()),
          static_cast<uint4*>(arena.data_ptr()),
          static_cast<const int32_t*>(keys.data_ptr()),
          static_cast<const int32_t*>(counts.data_ptr()),
          cache.size(0),
          arena.size(0),
          cache.size(1) / 16);
    } else {
      scatter_kernel<<<blocks, 256, 0, get_stream(cache.device())>>>(
          static_cast<const uint8_t*>(cache.data_ptr()),
          static_cast<uint8_t*>(arena.data_ptr()),
          static_cast<const int32_t*>(keys.data_ptr()),
          static_cast<const int32_t*>(counts.data_ptr()),
          cache.size(0),
          arena.size(0),
          cache.size(1));
    }
    auto status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

namespace {
template <bool WriteOutput = true>
__global__ void route_kernel(
    const int32_t* ids, const int32_t* lookup, int32_t* output, int32_t* counts, int64_t size, int32_t experts) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    int32_t expert = ids[i];
    if (expert == -1) {
      if (WriteOutput) output[i] = -1;
    } else if (expert < 0 || expert >= experts) {
      if (WriteOutput) output[i] = -1;
      atomicAdd(counts + experts, 1);  // invalid-route counter
    } else {
      if (WriteOutput) output[i] = lookup[expert];
      atomicAdd(counts + expert, 1);
    }
  }
}
}  // namespace

struct Qwen38CacheRoute {
  static void run_counts(TensorView ids, TensorView counts) {
    CHECK_INPUT_AND_TYPE(ids, dl_int32);
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_DIM(2, ids);
    CHECK_DIM(1, counts);
    CHECK_DEVICE(ids, counts);
    TVM_FFI_ICHECK_GT(counts.size(0), 1);
    auto stream = get_stream(ids.device());
    auto status = cudaMemsetAsync(counts.data_ptr(), 0, counts.size(0) * sizeof(int32_t), stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    auto size = ids.size(0) * ids.size(1);
    if (size) {
      int blocks = static_cast<int>(std::min<int64_t>((size + 255) / 256, 65535));
      route_kernel<false><<<blocks, 256, 0, stream>>>(
          static_cast<const int32_t*>(ids.data_ptr()),
          nullptr,
          nullptr,
          static_cast<int32_t*>(counts.data_ptr()),
          size,
          counts.size(0) - 1);
      status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }
  }

  static void run(TensorView ids, TensorView lookup, TensorView output, TensorView counts) {
    CHECK_INPUT_AND_TYPE(ids, dl_int32);
    CHECK_INPUT_AND_TYPE(lookup, dl_int32);
    CHECK_INPUT_AND_TYPE(output, dl_int32);
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_DIM(2, ids);
    CHECK_DIM(1, lookup);
    CHECK_DIM(1, counts);
    CHECK_DIM(2, output);
    CHECK_DEVICE(ids, lookup);
    CHECK_DEVICE(ids, output);
    CHECK_DEVICE(ids, counts);
    TVM_FFI_ICHECK_EQ(ids.size(0), output.size(0));
    TVM_FFI_ICHECK_EQ(ids.size(1), output.size(1));
    TVM_FFI_ICHECK_EQ(counts.size(0), lookup.size(0) + 1);
    TVM_FFI_ICHECK_GT(lookup.size(0), 0);
    auto stream = get_stream(ids.device());
    auto status = cudaMemsetAsync(counts.data_ptr(), 0, counts.size(0) * sizeof(int32_t), stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    auto size = ids.size(0) * ids.size(1);
    if (size) {
      int blocks = static_cast<int>(std::min<int64_t>((size + 255) / 256, 65535));
      route_kernel<<<blocks, 256, 0, stream>>>(
          static_cast<const int32_t*>(ids.data_ptr()),
          static_cast<const int32_t*>(lookup.data_ptr()),
          static_cast<int32_t*>(output.data_ptr()),
          static_cast<int32_t*>(counts.data_ptr()),
          size,
          lookup.size(0));
      status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }
  }
};

namespace {
// E <= 512, S <= E: a single control thread avoids atomics/partial decisions.
// Routing histogram is parallel; policy cost is bounded independently of tokens.
__device__ void apply_plan(
    const int32_t* counts,
    int32_t* lookup,
    int32_t* keys,
    float* frequency,
    int64_t* ages,
    int64_t* generation,
    int64_t* clock,
    int32_t* plan,
    int experts,
    int slots,
    float decay,
    int policy,
    int protected_slots,
    bool export_routes,
    const int32_t* active_ids = nullptr,
    int active_size = -1) {
  // plan: status, miss_count, active_count, arena; S (expert, slot) pairs;
  // S active slot IDs. status=1 for invalid routes, 0 for success.
  plan[0] = counts[experts] ? 1 : 0;
  plan[1] = 0;
  plan[2] = 0;
  plan[3] = 0;
  if (plan[0]) return;
  int active = active_size;
  if (active < 0) {
    active = 0;
    for (int e = 0; e < experts; ++e)
      active += counts[e] > 0;
  }
  if (active > slots) {
    plan[3] = 1;
    return;
  }
  if (!policy)
    for (int s = 0; s < slots; ++s)
      frequency[s] *= decay;
  for (int index = 0; index < (active_ids ? active : experts); ++index) {
    int e = active_ids ? active_ids[index] : index;
    if (!counts[e]) continue;
    int slot = lookup[e];
    bool hit = slot >= 0;
    if (slot < 0) {
      int victim = -1;
      for (int s = 0; s < slots; ++s) {
        int old = keys[s];
        if (old < 0) {
          victim = s;
          break;
        }
        if (counts[old]) continue;  // Never evict this invocation's active set.
        if (victim < 0 || frequency[s] < frequency[victim] ||
            (frequency[s] == frequency[victim] && ages[s] < ages[victim]))
          victim = s;
      }
      // There is always capacity since active <= slots and no persistent pins.
      if (victim < 0) {
        plan[0] = 2;
        return;
      }
      slot = victim;
      if (keys[slot] >= 0) lookup[keys[slot]] = -1;
      keys[slot] = e;
      lookup[e] = slot;
      generation[slot] += 1;
      frequency[slot] = 0;
      int miss = plan[1]++;
      plan[4 + 2 * miss] = e;
      plan[5 + 2 * miss] = slot;
    }
    ages[slot] = ++clock[0];
    if (!policy) {
      frequency[slot] += 1;
    } else if (hit && frequency[slot] != 1) {
      frequency[slot] = 1;
      int count = 0, oldest = -1;
      for (int s = 0; s < slots; ++s) {
        if (frequency[s] != 1) continue;
        ++count;
        if (oldest < 0 || ages[s] < ages[oldest]) oldest = s;
      }
      if (count > protected_slots) frequency[oldest] = 0;
    }
    if (export_routes) plan[4 + 3 * slots + plan[2]] = e;
    plan[4 + 2 * slots + plan[2]++] = slot;
  }
}

__global__ void plan_kernel(
    const int32_t* counts,
    int32_t* lookup,
    int32_t* keys,
    float* frequency,
    int64_t* ages,
    int64_t* generation,
    int64_t* clock,
    int32_t* plan,
    int experts,
    int slots,
    float decay,
    int policy,
    int protected_slots,
    bool export_routes) {
  if (threadIdx.x || blockIdx.x) return;
  apply_plan(
      counts,
      lookup,
      keys,
      frequency,
      ages,
      generation,
      clock,
      plan,
      experts,
      slots,
      decay,
      policy,
      protected_slots,
      export_routes);
}

// One token, top-k <= 32. Only the small active list is sorted, preserving
// the original planner's ascending expert order and its LFU/SLRU tie-breaks.
// LFU only: parallel decay/victim scan, sequential sorted expert updates.
__device__ bool better_victim(int a, int b, const int32_t* keys, const float* frequency, const int64_t* ages) {
  if (a < 0) return false;
  if (b < 0) return true;
  bool empty_a = keys[a] < 0, empty_b = keys[b] < 0;
  if (empty_a != empty_b) return empty_a;
  if (empty_a) return a < b;
  if (frequency[a] != frequency[b]) return frequency[a] < frequency[b];
  if (ages[a] != ages[b]) return ages[a] < ages[b];
  return a < b;
}

// A reduction carries the complete LFU ordering key, including 64-bit ages.
struct VictimKey {
  int slot;
  int empty;
  float heat;
  int64_t age;
};

__device__ bool better_victim_key(const VictimKey& a, const VictimKey& b) {
  if (a.slot < 0) return false;
  if (b.slot < 0) return true;
  if (a.empty != b.empty) return a.empty;
  if (a.empty) return a.slot < b.slot;
  if (a.heat != b.heat) return a.heat < b.heat;
  if (a.age != b.age) return a.age < b.age;
  return a.slot < b.slot;
}

__device__ void parallel_lfu_plan(
    const int32_t* counts,
    const int32_t* active_ids,
    int active,
    int experts,
    int slots,
    int32_t* lookup,
    int32_t* keys,
    float* frequency,
    int64_t* ages,
    int64_t* generation,
    int64_t* clock,
    int32_t* plan,
    float decay,
    bool export_routes,
    bool warp_reduce) {
  __shared__ int candidates[128];
  __shared__ VictimKey warp_candidates[4];
  __shared__ int current_slot;
  if (threadIdx.x == 0) {
    plan[0] = counts[experts] ? 1 : 0;
    plan[1] = plan[2] = plan[3] = 0;
    if (active > slots) plan[3] = 1;
  }
  __syncthreads();
  if (plan[0] || plan[3]) return;
  for (int slot = threadIdx.x; slot < slots; slot += blockDim.x)
    frequency[slot] *= decay;
  __syncthreads();
  for (int index = 0; index < active; ++index) {
    int expert = active_ids[index];
    // Active IDs are unique and protected from eviction. Their lookup rows
    // cannot change in earlier iterations, so hits need no block rendezvous.
    int slot = lookup[expert];
    if (slot < 0) {
      int best = -1;
      for (int slot = threadIdx.x; slot < slots; slot += blockDim.x) {
        if (keys[slot] >= 0 && counts[keys[slot]]) continue;
        if (better_victim(slot, best, keys, frequency, ages)) best = slot;
      }
      if (warp_reduce) {
        // Carry comparison fields in registers: shuffling only slot IDs would
        // reload dependent global metadata at every reduction stage.
        VictimKey key{
            best, best >= 0 && keys[best] < 0, best >= 0 ? frequency[best] : 0.0f, best >= 0 ? ages[best] : 0};
        for (int delta = 16; delta; delta >>= 1) {
          VictimKey other{
              __shfl_down_sync(0xffffffff, key.slot, delta),
              __shfl_down_sync(0xffffffff, key.empty, delta),
              __shfl_down_sync(0xffffffff, key.heat, delta),
              __shfl_down_sync(0xffffffff, key.age, delta)};
          if ((threadIdx.x & 31) + delta < 32 && better_victim_key(other, key)) key = other;
        }
        if ((threadIdx.x & 31) == 0) warp_candidates[threadIdx.x >> 5] = key;
        __syncthreads();
        if (threadIdx.x == 0) {
          key = warp_candidates[0];
          for (int warp = 1; warp < 4; ++warp)
            if (better_victim_key(warp_candidates[warp], key)) key = warp_candidates[warp];
          candidates[0] = key.slot;
        }
      } else {
        candidates[threadIdx.x] = best;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
          if (threadIdx.x < stride &&
              better_victim(candidates[threadIdx.x + stride], candidates[threadIdx.x], keys, frequency, ages))
            candidates[threadIdx.x] = candidates[threadIdx.x + stride];
          __syncthreads();
        }
      }
      if (threadIdx.x == 0) {
        current_slot = candidates[0];
        if (current_slot < 0)
          plan[0] = 2;
        else {
          int slot = current_slot;
          if (keys[slot] >= 0) lookup[keys[slot]] = -1;
          keys[slot] = expert;
          lookup[expert] = slot;
          generation[slot] += 1;
          frequency[slot] = 0;
          int miss = plan[1]++;
          plan[4 + 2 * miss] = expert;
          plan[5 + 2 * miss] = slot;
        }
      }
      __syncthreads();
      if (plan[0]) return;
      slot = current_slot;
    }
    if (threadIdx.x == 0) {
      ages[slot] = ++clock[0];
      frequency[slot] += 1;
      if (export_routes) plan[4 + 3 * slots + plan[2]] = expert;
      plan[4 + 2 * slots + plan[2]++] = slot;
    }
    // Only active metadata changed here; victim scans exclude those slots.
    // The miss-install barrier above publishes keys before the next scan.
  }
}

__global__ void decode_plan_kernel(
    const int32_t* ids,
    int32_t* output,
    int32_t* counts,
    int32_t* lookup,
    int32_t* keys,
    float* frequency,
    int64_t* ages,
    int64_t* generation,
    int64_t* clock,
    int32_t* plan,
    int experts,
    int slots,
    int top_k,
    float decay,
    int policy,
    int protected_slots,
    bool export_routes,
    int64_t* hit_stats,
    int32_t* sample,
    int parallel_miss,
    int32_t* early_ids = nullptr,
    int32_t* late_ids = nullptr,
    sglang::LayeredHybridArgs hybrid_args = {},
    bool keep_all_hits = false) {
  const auto* hybrid = hybrid_args.descriptor ? &hybrid_args : nullptr;
  const int route_count = hybrid ? hybrid->routes : top_k;
  __shared__ int32_t histogram[513];
  __shared__ int32_t active_ids[32];
  __shared__ int active_count;
  __shared__ bool all_hit;
  __shared__ int64_t clock_before;
  __shared__ int32_t filtered_ids[32];
  const int32_t* original_ids = ids;
  __shared__ int original_active;
  __shared__ int original_misses;
  __shared__ int32_t original_active_ids[32];
  __shared__ int safe_slot;
  __shared__ int cpu_count;
  __shared__ float victim_heat[4];
  __shared__ int resident_count[4];
  for (int e = threadIdx.x; e <= experts; e += blockDim.x)
    histogram[e] = 0;
  __syncthreads();
  if (threadIdx.x < route_count) {
    int expert = ids[threadIdx.x];
    filtered_ids[threadIdx.x] = expert;
    if (early_ids) {
      int32_t hit = (expert >= 0 && expert < experts) ? lookup[expert] : -1;
      early_ids[threadIdx.x] = keep_all_hits && hit >= slots ? -1 : hit;
    }
    if (expert >= 0 && expert < experts)
      atomicAdd(histogram + expert, 1);
    else if (expert != -1)
      atomicAdd(histogram + experts, 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    // The host bounds this union to one warp. Sort in registers, then compact
    // unique valid IDs in ascending order, preserving LFU ages and tie breaks.
    const int lane = threadIdx.x;
    int expert = lane < route_count ? ids[lane] : experts;
    if (expert < 0 || expert >= experts) expert = experts;
#pragma unroll
    for (int size = 2; size <= 32; size *= 2) {
#pragma unroll
      for (int stride = size / 2; stride > 0; stride /= 2) {
        int other = __shfl_xor_sync(0xffffffff, expert, stride);
        bool take_min = ((lane & stride) == 0) == ((lane & size) == 0);
        expert = take_min ? min(expert, other) : max(expert, other);
      }
    }
    int previous = __shfl_up_sync(0xffffffff, expert, 1);
    bool unique = expert < experts && (lane == 0 || expert != previous);
    unsigned mask = __ballot_sync(0xffffffff, unique);
    bool missing = false;
    if (unique) {
      int position = __popc(mask & ((1u << lane) - 1));
      active_ids[position] = expert;
      original_active_ids[position] = expert;
      missing = lookup[expert] < 0;
    }
    unsigned misses = __ballot_sync(0xffffffff, missing);
    if (lane == 0) {
      original_active = active_count = __popc(mask);
      original_misses = __popc(misses);
    }
  }
  __syncthreads();
  if (hybrid && hybrid->expert_frequency && !histogram[experts] && original_active <= slots) {
    // Read resident heat through the lookup in the same pass as nonresident
    // history. This preserves prefill/resize restoration without a seed pass.
    float victim = __int_as_float(0x7f800000);
    int residents = 0;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
      int slot = lookup[e];
      float heat = slot >= 0 ? frequency[slot] : hybrid->expert_frequency[e];
      heat *= decay;
      if (histogram[e]) heat += 1.0f;
      hybrid->expert_frequency[e] = heat;
      if (slot >= 0) {
        ++residents;
        if (!histogram[e] && heat < victim) victim = heat;
      }
    }
    // The launch has four full warps. Reduce admission metadata alongside
    // heat, avoiding thread 0's serial scan over all resident slots.
    if (original_misses && hybrid->max_cpu_experts) {
      for (int delta = 16; delta; delta >>= 1) {
        float other = __shfl_down_sync(0xffffffff, victim, delta);
        if (other < victim) victim = other;
        residents += __shfl_down_sync(0xffffffff, residents, delta);
      }
      if ((threadIdx.x & 31) == 0) {
        victim_heat[threadIdx.x >> 5] = victim;
        resident_count[threadIdx.x >> 5] = residents;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int active = original_active;
    cpu_count = 0;
    if (hybrid && !histogram[experts] && active <= slots) {
      if (hybrid->expert_frequency) {
        if (original_misses && hybrid->max_cpu_experts) {
          float victim = victim_heat[0];
          int residents = resident_count[0];
          for (int warp = 1; warp < 4; ++warp) {
            if (victim_heat[warp] < victim) victim = victim_heat[warp];
            residents += resident_count[warp];
          }
          cpu_count = sglang::layered_lfu_select(
              ids, lookup, active_ids, active, original_misses, slots - residents, victim, *hybrid, filtered_ids);
        }
      } else {
        cpu_count = sglang::layered_select(ids, lookup, experts, slots, *hybrid, filtered_ids);
      }
      // Only CPU selection changes the GPU route union. Most calls keep it.
      if (cpu_count) {
        for (int j = 0; j < route_count; ++j) {
          int e = ids[j];
          if (e >= 0 && e < experts && filtered_ids[j] < 0) histogram[e] = 0;
        }
        int gpu_active = 0;
        for (int j = 0; j < active; ++j)
          if (histogram[active_ids[j]]) active_ids[gpu_active++] = active_ids[j];
        active = gpu_active;
      }
      hybrid->descriptor[0] = cpu_count;
      hybrid->descriptor[1] = hybrid->sequence;
      hybrid->stats[0] += original_misses;
      hybrid->stats[1] += cpu_count;
      // GPU fill count is recorded after the LFU plan is complete.
      hybrid->stats[3] += cpu_count != 0;
    } else if (hybrid) {
      hybrid->stats[0] += original_misses;
      hybrid->descriptor[0] = 0;
      hybrid->descriptor[1] = hybrid->sequence;
    }
    active_count = active;
    safe_slot = -1;
  }
  __syncthreads();
  if (hybrid && cpu_count) sglang::layered_publish(original_ids, filtered_ids, *hybrid);
  if (hybrid) ids = filtered_ids;
  if (threadIdx.x == 0) {
    all_hit = !policy && !histogram[experts] && active_count <= slots;
    // CPU selection only removes unique nonresident experts from the union.
    all_hit = all_hit && original_misses == cpu_count;
    if (all_hit) {
      clock_before = clock[0];
      clock[0] += active_count;
      plan[0] = plan[1] = plan[3] = 0;
      plan[2] = active_count;
    } else if (!parallel_miss || policy) {
      apply_plan(
          histogram,
          lookup,
          keys,
          frequency,
          ages,
          generation,
          clock,
          plan,
          experts,
          slots,
          decay,
          policy,
          protected_slots,
          export_routes,
          active_ids,
          active_count);
    }
  }
  __syncthreads();
  if (!all_hit && parallel_miss && !policy) {
    parallel_lfu_plan(
        histogram,
        active_ids,
        active_count,
        experts,
        slots,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        export_routes,
        parallel_miss == 2);
  }
  if (all_hit) {
    // No victim selection or lookup mutation. Keep decay and increment as
    // separate operations to match the reference LFU arithmetic exactly.
    for (int s = threadIdx.x; s < slots; s += blockDim.x)
      frequency[s] *= decay;
    __syncthreads();
    if (threadIdx.x < active_count) {
      int expert = active_ids[threadIdx.x], slot = lookup[expert];
      frequency[slot] += 1;
      ages[slot] = clock_before + threadIdx.x + 1;
      plan[4 + 2 * slots + threadIdx.x] = slot;
      if (export_routes) plan[4 + 3 * slots + threadIdx.x] = expert;
    }
  }
  if (hybrid && hybrid->expert_frequency) {
    __syncthreads();
    for (int j = threadIdx.x; j < active_count; j += blockDim.x) {
      int e = active_ids[j], slot = lookup[e];
      if (slot >= 0) frequency[slot] = hybrid->expert_frequency[e];
    }
  }
  for (int e = threadIdx.x; e <= experts; e += blockDim.x)
    counts[e] = histogram[e];
  if (hybrid && hybrid->safe_mask) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < route_count; ++i) {
        int expert = ids[i];
        if (expert >= 0 && expert < experts && lookup[expert] >= 0) {
          safe_slot = lookup[expert];
          break;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < route_count) {
    int expert = ids[threadIdx.x];
    int slot = (expert >= 0 && expert < experts) ? lookup[expert] : -1;
    if (hybrid) {
      hybrid->cpu_ids[threadIdx.x] = original_ids[threadIdx.x] >= 0 && original_ids[threadIdx.x] < experts && expert < 0
                                         ? original_ids[threadIdx.x]
                                         : -1;
      hybrid->gpu_weights[threadIdx.x] = expert < 0 ? 0.0f : hybrid->weights[threadIdx.x];
      if (hybrid->safe_mask && slot < 0 && safe_slot >= 0) slot = safe_slot;
    }
    output[threadIdx.x] = slot;
    if (early_ids) {
      // Full-MoE splitting needs two contributions only for mixed plans.
      // Native FC1 staging also retains all-hit snapshots so they run early.
      // No hit status crosses to the host.
      bool mixed = keep_all_hits || (plan[1] > 0 && plan[1] < plan[2]);
      int hit = early_ids[threadIdx.x];
      early_ids[threadIdx.x] = mixed ? hit : -1;
      late_ids[threadIdx.x] = mixed && hit >= 0 ? -1 : slot;
    }
  }
  if (hit_stats || sample || hybrid) {
    __syncthreads();
    if (plan[0] || plan[3]) {
      if (threadIdx.x == 0) asm volatile("trap;");
      return;
    }
    if (threadIdx.x == 0) {
      // Prior readers and updates are ordered by the caller's stream/events.
      // Accumulate every plan on device; a host snapshot cannot lose misses
      // when several invocations of the same layer are already queued.
      if (hybrid) {
        hybrid->stats[2] += plan[1];
        hybrid->stats[4] += plan[1] != 0;
      }
      if (hit_stats) {
        hit_stats[0] += hybrid ? original_active - original_misses : plan[2] - plan[1];
        hit_stats[1] += (hybrid ? original_misses : plan[1]) == 0;
        hit_stats[2] += hybrid ? original_misses : plan[1];
        // Copy-call accounting follows actual GPU fills, including CPU-only
        // misses whose original miss count is nonzero but need no DMA.
        hit_stats[3] += plan[1] != 0;
      }
      if (sample) {
        // Original expert IDs, sorted and unique, including hits. Never emit
        // remapped cache slots or stale padding from the reusable plan tail.
        if (hybrid) {
          sample[0] = original_active;
          for (int i = 0; i < original_active; ++i)
            sample[1 + i] = original_active_ids[i];
        } else {
          sample[0] = active_count;
          for (int i = 0; i < active_count; ++i)
            sample[1 + i] = active_ids[i];
        }
      }
    }
  }
}
}  // namespace

struct Qwen38CachePlan {
  static void
  run(TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction) {
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_INPUT_AND_TYPE(lookup, dl_int32);
    CHECK_INPUT_AND_TYPE(keys, dl_int32);
    CHECK_INPUT_AND_TYPE(frequency, dl_float32);
    CHECK_INPUT_AND_TYPE(ages, dl_int64);
    CHECK_INPUT_AND_TYPE(generation, dl_int64);
    CHECK_INPUT_AND_TYPE(clock, dl_int64);
    CHECK_INPUT_AND_TYPE(plan, dl_int32);
    CHECK_DIM(1, counts);
    CHECK_DIM(1, lookup);
    CHECK_DIM(1, keys);
    CHECK_DIM(1, frequency);
    CHECK_DIM(1, ages);
    CHECK_DIM(1, generation);
    CHECK_DIM(1, clock);
    CHECK_DIM(1, plan);
    CHECK_DEVICE(counts, lookup);
    CHECK_DEVICE(counts, keys);
    CHECK_DEVICE(counts, frequency);
    CHECK_DEVICE(counts, ages);
    CHECK_DEVICE(counts, generation);
    CHECK_DEVICE(counts, clock);
    CHECK_DEVICE(counts, plan);
    int experts = lookup.size(0), slots = keys.size(0);
    TVM_FFI_ICHECK_GT(experts, 0);
    TVM_FFI_ICHECK_LE(experts, 512);
    TVM_FFI_ICHECK_GT(slots, 0);
    TVM_FFI_ICHECK_LE(slots, experts);
    TVM_FFI_ICHECK_EQ(counts.size(0), experts + 1);
    TVM_FFI_ICHECK_EQ(frequency.size(0), slots);
    TVM_FFI_ICHECK_EQ(ages.size(0), slots);
    TVM_FFI_ICHECK_EQ(generation.size(0), slots);
    TVM_FFI_ICHECK_EQ(clock.size(0), 1);
    TVM_FFI_ICHECK(plan.size(0) == 4 + 3 * slots || plan.size(0) == 4 + 4 * slots);
    TVM_FFI_ICHECK(policy == 0 || policy == 1);
    TVM_FFI_ICHECK_GE(protected_fraction, 0.0);
    TVM_FFI_ICHECK_LT(protected_fraction, 1.0);
    TVM_FFI_ICHECK_GT(decay, 0.0);
    TVM_FFI_ICHECK_LE(decay, 1.0);
    plan_kernel<<<1, 1, 0, get_stream(counts.device())>>>(
        static_cast<const int32_t*>(counts.data_ptr()),
        static_cast<int32_t*>(lookup.data_ptr()),
        static_cast<int32_t*>(keys.data_ptr()),
        static_cast<float*>(frequency.data_ptr()),
        static_cast<int64_t*>(ages.data_ptr()),
        static_cast<int64_t*>(generation.data_ptr()),
        static_cast<int64_t*>(clock.data_ptr()),
        static_cast<int32_t*>(plan.data_ptr()),
        experts,
        slots,
        static_cast<float>(decay),
        static_cast<int>(policy),
        static_cast<int>(slots * protected_fraction),
        plan.size(0) == 4 + 4 * slots);
    auto status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

struct Qwen38CacheDecodePlan {
  static void
  run(TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction) {
    run_mailbox(
        ids,
        output,
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
        0,
        0);
  }
  static void run_mailbox(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample) {
    CHECK_INPUT_AND_TYPE(ids, dl_int32);
    CHECK_INPUT_AND_TYPE(output, dl_int32);
    CHECK_DIM(2, ids);
    CHECK_DIM(2, output);
    CHECK_DEVICE(ids, counts);
    CHECK_DEVICE(ids, output);
    TVM_FFI_ICHECK_EQ(ids.size(0), 1);
    TVM_FFI_ICHECK_GT(ids.size(1), 0);
    TVM_FFI_ICHECK_LE(ids.size(1), 32);
    TVM_FFI_ICHECK_EQ(output.size(0), 1);
    TVM_FFI_ICHECK_EQ(output.size(1), ids.size(1));
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_INPUT_AND_TYPE(lookup, dl_int32);
    CHECK_INPUT_AND_TYPE(keys, dl_int32);
    CHECK_INPUT_AND_TYPE(frequency, dl_float32);
    CHECK_INPUT_AND_TYPE(ages, dl_int64);
    CHECK_INPUT_AND_TYPE(generation, dl_int64);
    CHECK_INPUT_AND_TYPE(clock, dl_int64);
    CHECK_INPUT_AND_TYPE(plan, dl_int32);
    CHECK_DIM(1, counts);
    CHECK_DIM(1, lookup);
    CHECK_DIM(1, keys);
    CHECK_DIM(1, frequency);
    CHECK_DIM(1, ages);
    CHECK_DIM(1, generation);
    CHECK_DIM(1, clock);
    CHECK_DIM(1, plan);
    CHECK_DEVICE(counts, lookup);
    CHECK_DEVICE(counts, keys);
    CHECK_DEVICE(counts, frequency);
    CHECK_DEVICE(counts, ages);
    CHECK_DEVICE(counts, generation);
    CHECK_DEVICE(counts, clock);
    CHECK_DEVICE(counts, plan);
    int experts = lookup.size(0), slots = keys.size(0);
    TVM_FFI_ICHECK_GT(experts, 0);
    TVM_FFI_ICHECK_LE(experts, 512);
    TVM_FFI_ICHECK_GT(slots, 0);
    TVM_FFI_ICHECK_LE(slots, experts);
    TVM_FFI_ICHECK_EQ(counts.size(0), experts + 1);
    TVM_FFI_ICHECK_EQ(frequency.size(0), slots);
    TVM_FFI_ICHECK_EQ(ages.size(0), slots);
    TVM_FFI_ICHECK_EQ(generation.size(0), slots);
    TVM_FFI_ICHECK_EQ(clock.size(0), 1);
    TVM_FFI_ICHECK(plan.size(0) == 4 + 3 * slots || plan.size(0) == 4 + 4 * slots);
    TVM_FFI_ICHECK(policy == 0 || policy == 1);
    TVM_FFI_ICHECK_GE(protected_fraction, 0.0);
    TVM_FFI_ICHECK_LT(protected_fraction, 1.0);
    TVM_FFI_ICHECK_GT(decay, 0.0);
    TVM_FFI_ICHECK_LE(decay, 1.0);
    decode_plan_kernel<<<1, 128, 0, get_stream(ids.device())>>>(
        static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<int32_t*>(counts.data_ptr()),
        static_cast<int32_t*>(lookup.data_ptr()),
        static_cast<int32_t*>(keys.data_ptr()),
        static_cast<float*>(frequency.data_ptr()),
        static_cast<int64_t*>(ages.data_ptr()),
        static_cast<int64_t*>(generation.data_ptr()),
        static_cast<int64_t*>(clock.data_ptr()),
        static_cast<int32_t*>(plan.data_ptr()),
        experts,
        slots,
        ids.size(1),
        static_cast<float>(decay),
        static_cast<int>(policy),
        static_cast<int>(slots * protected_fraction),
        plan.size(0) == 4 + 4 * slots,
        reinterpret_cast<int64_t*>(hit_stats),
        reinterpret_cast<int32_t*>(sample),
        false);
    auto status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
  static void run_parallel(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample,
      int64_t parallel_miss) {
    run_parallel_impl(
        ids,
        output,
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
        hit_stats,
        sample,
        parallel_miss,
        0,
        0);
  }
  template <bool KeepAllHits = false>
  static void run_parallel_early(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample,
      int64_t parallel_miss,
      TensorView early_ids,
      TensorView late_ids) {
    CHECK_INPUT_AND_TYPE(early_ids, dl_int32);
    CHECK_INPUT_AND_TYPE(late_ids, dl_int32);
    CHECK_DIM(2, early_ids);
    CHECK_DIM(2, late_ids);
    CHECK_DEVICE(ids, early_ids);
    CHECK_DEVICE(ids, late_ids);
    TVM_FFI_ICHECK_EQ(early_ids.size(0), ids.size(0));
    TVM_FFI_ICHECK_EQ(early_ids.size(1), ids.size(1));
    TVM_FFI_ICHECK_EQ(late_ids.size(0), ids.size(0));
    TVM_FFI_ICHECK_EQ(late_ids.size(1), ids.size(1));
    run_parallel_impl(
        ids,
        output,
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
        hit_stats,
        sample,
        parallel_miss,
        reinterpret_cast<int64_t>(early_ids.data_ptr()),
        reinterpret_cast<int64_t>(late_ids.data_ptr()), nullptr, KeepAllHits);
  }
  static void run_layered(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample,
      int64_t parallel_miss,
      TensorView x,
      TensorView weights,
      TensorView table,
      TensorView gpu_weights,
      TensorView cpu_ids,
      TensorView descriptor,
      TensorView stats,
      int64_t host_page,
      int64_t layer,
      int64_t sequence,
      int64_t forced,
      int64_t safe_mask,
      int64_t overlap) {
    auto args = sglang::layered_args(
        ids,
        x,
        weights,
        table,
        gpu_weights,
        cpu_ids,
        descriptor,
        stats,
        host_page,
        layer,
        sequence,
        forced,
        safe_mask,
        overlap,
        keys.size(0));
    TVM_FFI_ICHECK_EQ(policy, 0);
    run_parallel_impl(
        ids,
        output,
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
        hit_stats,
        sample,
        parallel_miss,
        0,
        0,
        &args);
  }
  static void run_layered_lfu(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample,
      int64_t parallel_miss,
      TensorView x,
      TensorView weights,
      TensorView table,
      TensorView gpu_weights,
      TensorView cpu_ids,
      TensorView descriptor,
      TensorView stats,
      int64_t host_page,
      int64_t layer,
      int64_t sequence,
      int64_t forced,
      int64_t safe_mask,
      int64_t overlap,
      TensorView expert_frequency,
      int64_t max_cpu_experts) {
    auto args = sglang::layered_args(
        ids,
        x,
        weights,
        table,
        gpu_weights,
        cpu_ids,
        descriptor,
        stats,
        host_page,
        layer,
        sequence,
        forced,
        safe_mask,
        overlap,
        keys.size(0));
    CHECK_INPUT_AND_TYPE(expert_frequency, dl_float32);
    CHECK_CONTIGUOUS(expert_frequency);
    CHECK_DIM(1, expert_frequency);
    CHECK_DEVICE(ids, expert_frequency);
    TVM_FFI_ICHECK_EQ(expert_frequency.size(0), lookup.size(0));
    TVM_FFI_ICHECK_GE(max_cpu_experts, 0);
    TVM_FFI_ICHECK_LE(max_cpu_experts, 30);
    args.expert_frequency = static_cast<float*>(expert_frequency.data_ptr());
    args.max_cpu_experts = static_cast<int>(max_cpu_experts);
    TVM_FFI_ICHECK_EQ(policy, 0);
    run_parallel_impl(
        ids,
        output,
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
        hit_stats,
        sample,
        parallel_miss,
        0,
        0,
        &args);
  }
  static void run_parallel_impl(
      TensorView ids,
      TensorView output,
      TensorView counts,
      TensorView lookup,
      TensorView keys,
      TensorView frequency,
      TensorView ages,
      TensorView generation,
      TensorView clock,
      TensorView plan,
      double decay,
      int64_t policy,
      double protected_fraction,
      int64_t hit_stats,
      int64_t sample,
      int64_t parallel_miss,
      int64_t early_ids,
      int64_t late_ids,
      const sglang::LayeredHybridArgs* hybrid = nullptr,
      bool keep_all_hits = false) {
    TVM_FFI_ICHECK_GE(parallel_miss, 0);
    TVM_FFI_ICHECK_LE(parallel_miss, 2);
    CHECK_INPUT_AND_TYPE(ids, dl_int32);
    CHECK_INPUT_AND_TYPE(output, dl_int32);
    CHECK_DIM(2, ids);
    CHECK_DIM(2, output);
    CHECK_DEVICE(ids, counts);
    CHECK_DEVICE(ids, output);
    TVM_FFI_ICHECK_EQ(ids.size(0), 1);
    TVM_FFI_ICHECK_GT(ids.size(1), 0);
    TVM_FFI_ICHECK_LE(ids.size(1), 32);
    TVM_FFI_ICHECK_EQ(output.size(0), 1);
    TVM_FFI_ICHECK_EQ(output.size(1), ids.size(1));
    CHECK_INPUT_AND_TYPE(counts, dl_int32);
    CHECK_INPUT_AND_TYPE(lookup, dl_int32);
    CHECK_INPUT_AND_TYPE(keys, dl_int32);
    CHECK_INPUT_AND_TYPE(frequency, dl_float32);
    CHECK_INPUT_AND_TYPE(ages, dl_int64);
    CHECK_INPUT_AND_TYPE(generation, dl_int64);
    CHECK_INPUT_AND_TYPE(clock, dl_int64);
    CHECK_INPUT_AND_TYPE(plan, dl_int32);
    CHECK_DIM(1, counts);
    CHECK_DIM(1, lookup);
    CHECK_DIM(1, keys);
    CHECK_DIM(1, frequency);
    CHECK_DIM(1, ages);
    CHECK_DIM(1, generation);
    CHECK_DIM(1, clock);
    CHECK_DIM(1, plan);
    CHECK_DEVICE(counts, lookup);
    CHECK_DEVICE(counts, keys);
    CHECK_DEVICE(counts, frequency);
    CHECK_DEVICE(counts, ages);
    CHECK_DEVICE(counts, generation);
    CHECK_DEVICE(counts, clock);
    CHECK_DEVICE(counts, plan);
    int experts = lookup.size(0), slots = keys.size(0);
    TVM_FFI_ICHECK_GT(experts, 0);
    TVM_FFI_ICHECK_LE(experts, 512);
    TVM_FFI_ICHECK_GT(slots, 0);
    TVM_FFI_ICHECK_LE(slots, experts);
    TVM_FFI_ICHECK_EQ(counts.size(0), experts + 1);
    TVM_FFI_ICHECK_EQ(frequency.size(0), slots);
    TVM_FFI_ICHECK_EQ(ages.size(0), slots);
    TVM_FFI_ICHECK_EQ(generation.size(0), slots);
    TVM_FFI_ICHECK_EQ(clock.size(0), 1);
    TVM_FFI_ICHECK(plan.size(0) == 4 + 3 * slots || plan.size(0) == 4 + 4 * slots);
    TVM_FFI_ICHECK(policy == 0 || policy == 1);
    TVM_FFI_ICHECK_GE(protected_fraction, 0.0);
    TVM_FFI_ICHECK_LT(protected_fraction, 1.0);
    TVM_FFI_ICHECK_GT(decay, 0.0);
    TVM_FFI_ICHECK_LE(decay, 1.0);
    decode_plan_kernel<<<1, 128, 0, get_stream(ids.device())>>>(
        static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<int32_t*>(counts.data_ptr()),
        static_cast<int32_t*>(lookup.data_ptr()),
        static_cast<int32_t*>(keys.data_ptr()),
        static_cast<float*>(frequency.data_ptr()),
        static_cast<int64_t*>(ages.data_ptr()),
        static_cast<int64_t*>(generation.data_ptr()),
        static_cast<int64_t*>(clock.data_ptr()),
        static_cast<int32_t*>(plan.data_ptr()),
        experts,
        slots,
        ids.size(1),
        static_cast<float>(decay),
        static_cast<int>(policy),
        static_cast<int>(slots * protected_fraction),
        plan.size(0) == 4 + 4 * slots,
        reinterpret_cast<int64_t*>(hit_stats),
        reinterpret_cast<int32_t*>(sample),
        parallel_miss,
        reinterpret_cast<int32_t*>(early_ids),
        reinterpret_cast<int32_t*>(late_ids),
        hybrid ? *hybrid : sglang::LayeredHybridArgs{}, keep_all_hits);
    auto status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

namespace {
__global__ void packed_scatter_kernel(const uint8_t* scratch, const int64_t* desc) {
  const int64_t* row = desc + 3 * blockIdx.x;
  auto* target = reinterpret_cast<uint8_t*>(row[0]);
  const auto* source = scratch + row[1];
  const int64_t size = row[2];
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.y;
  int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if ((reinterpret_cast<uintptr_t>(target) | reinterpret_cast<uintptr_t>(source) | static_cast<uintptr_t>(size)) % 16 ==
      0) {
    for (; i < size / 16; i += stride)
      reinterpret_cast<uint4*>(target)[i] = reinterpret_cast<const uint4*>(source)[i];
  } else {
    for (; i < size; i += stride)
      target[i] = source[i];
  }
}
}  // namespace

struct Qwen38PackedScatter {
  static void run(TensorView scratch, TensorView descriptors) {
    CHECK_INPUT_AND_TYPE(scratch, dl_uint8);
    CHECK_INPUT_AND_TYPE(descriptors, dl_int64);
    CHECK_DIM(1, scratch);
    CHECK_DIM(2, descriptors);
    CHECK_DEVICE(scratch, descriptors);
    TVM_FFI_ICHECK_EQ(descriptors.size(1), 3);
    TVM_FFI_ICHECK_LE(descriptors.size(0), 1024);
    if (!descriptors.size(0)) return;
    // Internal descriptors are immutable validated layout/address tables.
    packed_scatter_kernel<<<dim3(descriptors.size(0), 16), 256, 0, get_stream(scratch.device())>>>(
        static_cast<const uint8_t*>(scratch.data_ptr()), static_cast<const int64_t*>(descriptors.data_ptr()));
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
};

// Explicit registered-host DMA: does not depend on PyTorch pinned allocator tracking.
struct Qwen38RegisteredCopy {
  static void run(TensorView destination, TensorView source) {
    TVM_FFI_ICHECK_EQ(source.device().device_type, kDLCPU);
    CHECK_INPUT_TYPE(source, dl_uint8);
    CHECK_INPUT_AND_TYPE(destination, dl_uint8);
    CHECK_DIM(1, source);
    CHECK_DIM(1, destination);
    TVM_FFI_ICHECK_EQ(source.size(0), destination.size(0));
    auto status = cudaMemcpyAsync(
        destination.data_ptr(),
        source.data_ptr(),
        source.size(0),
        cudaMemcpyHostToDevice,
        get_stream(destination.device()));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};
