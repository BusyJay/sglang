// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <limits>

namespace sglang {
namespace details {
constexpr int kRoutes = 32;
constexpr int kCostColumns = 7 + 2 * kRoutes;

// Keep the cheapest at most kRoutes victims in a max heap. An invocation can
// insert no more than kRoutes distinct experts. Admission and forced placement
// each scan once because cold reclamation can change the eligible victims.
struct HybridVictim {
  double heat;
  int32_t layer, expert, slot;
};
#include "qwen38_heap_index.cuh"
__device__ inline bool victim_less(const HybridVictim& a, const HybridVictim& b) {
  if (a.heat != b.heat) return a.heat < b.heat;
  if (a.layer != b.layer) return a.layer < b.layer;
  return a.expert < b.expert;
}
__device__ inline void victim_push(HybridVictim* heap, int32_t* size, HybridVictim value) {
  int32_t i = (*size)++;
  while (i > 0) {
    int32_t parent = (i - 1) / 2;
    if (!victim_less(heap[parent], value)) break;
    heap[i] = heap[parent];
    i = parent;
  }
  heap[i] = value;
}
__device__ inline void victim_replace_root(HybridVictim* heap, int32_t size, HybridVictim value) {
  int32_t i = 0;
  while (2 * i + 1 < size) {
    int32_t child = 2 * i + 1;
    if (child + 1 < size && victim_less(heap[child], heap[child + 1])) ++child;
    if (!victim_less(value, heap[child])) break;
    heap[i] = heap[child];
    i = child;
  }
  heap[i] = value;
}
__device__ inline HybridVictim victim_pop_max(HybridVictim* heap, int32_t* size) {
  HybridVictim result = heap[0];
  HybridVictim last = heap[--*size];
  if (*size) victim_replace_root(heap, *size, last);
  return result;
}

// Global planners and all preceding GPU readers run on one ordered stream.
// Ownership identifies payloads; it does not restrict cross-layer victims.
template <bool Indexed>
__global__ void hybrid_plan_kernel(
    const int32_t* __restrict__ ids,
    int32_t* lookup,
    double* frequency,
    int32_t* owners,
    int32_t* keys,
    const double* __restrict__ costs,
    int32_t* cache_ids,
    int32_t* cpu_ids,
    int32_t* cache_plan,
    int64_t* stats,
    int32_t tokens,
    int32_t routes,
    int32_t experts,
    int32_t slots,
    int32_t samples,
    int32_t layer,
    double decay,
    double admission,
    const uint16_t* x = nullptr,
    const float* weights = nullptr,
    uint16_t* host_x = nullptr,
    int32_t* host_ids = nullptr,
    float* host_weights = nullptr,
    volatile int32_t* host_head = nullptr,
    int32_t hidden = 0,
    int32_t sequence = 0,
    int32_t* index_data = nullptr,
    int32_t layers = 0,
    int32_t index_slots = 0) {
  int32_t* global_layer_lookup = lookup + static_cast<int64_t>(layer) * experts;
  double* global_layer_frequency = frequency + static_cast<int64_t>(layer) * experts;
  // Serial policy scans must not issue a dependent global load per expert.
  // Publish the completed lookup cooperatively before the next ordered reader.
  __shared__ int32_t layer_lookup[512];
  __shared__ double layer_frequency[512];
  __shared__ int32_t layer_heap[512], layer_position[512], layer_heap_size;
  // Leave room for the global layer minima as well as the current layer.
  constexpr int kCachedCostRows = 32;
  __shared__ double cached_costs[kCachedCostRows * kCostColumns];
  constexpr int kCachedLayers = 64;
  __shared__ int32_t cached_top[2 * kCachedLayers + 1];
  __shared__ double cached_top_heat[kCachedLayers];
  int32_t* local_top = Indexed && layers <= kCachedLayers ? cached_top : nullptr;
  double* local_top_heat = local_top ? cached_top_heat : nullptr;
  const double* policy_costs = samples <= kCachedCostRows ? cached_costs : costs;
  __shared__ int32_t histogram[512];
  __shared__ int32_t requested[kRoutes], requested_count;
  __shared__ uint32_t cold_mask[16], route_mask[16];
  __shared__ int32_t route_offset[16];
  __shared__ int32_t active[kRoutes], rejected[kRoutes], selected;
  __shared__ int32_t active_count, rejected_count;
  __shared__ bool layer_was_present;
  for (int32_t e = threadIdx.x; e < experts; e += blockDim.x) {
    histogram[e] = 0;
    layer_lookup[e] = global_layer_lookup[e];
    layer_frequency[e] = global_layer_frequency[e];
    if (Indexed) {
      layer_heap[e] = index_data[layer * experts + e];
      layer_position[e] = index_data[layers * experts + layer * experts + e];
    }
  }
  if (threadIdx.x == 0) {
    cache_plan[0] = cache_plan[1] = cache_plan[2] = cache_plan[3] = 0;
    active_count = rejected_count = selected = 0;
    if (Indexed) layer_heap_size = index_data[2 * layers * experts + layer];
  }
  __syncthreads();
  for (int32_t route = threadIdx.x; route < routes; route += blockDim.x) {
    int32_t e = ids[route];
    cache_ids[route] = cpu_ids[route] = -1;
    if (e >= 0 && e < experts)
      atomicAdd(histogram + e, 1);
    else if (e != -1)
      atomicExch(cache_plan, 1);
  }
  __syncthreads();
  if (cache_plan[0]) {
    if (threadIdx.x == 0) asm volatile("trap;");
    return;
  }
  bool needs_policy = false;
  for (int32_t e = threadIdx.x; e < experts; e += blockDim.x) {
    layer_frequency[e] = __dadd_rn(__dmul_rn(layer_frequency[e], decay), histogram[e] > 0 ? 1.0 : 0.0);
    if (!Indexed) global_layer_frequency[e] = layer_frequency[e];
    needs_policy |= (histogram[e] > 0 && layer_lookup[e] < 0) ||
                    (histogram[e] == 0 && layer_lookup[e] >= 0 && layer_frequency[e] < 0.5);
  }
  // CTA-uniform decision also publishes the updated heats. All-hit must still
  // reclaim cold residents; only the indexed path can bypass ownership scans.
  int32_t policy_required = __syncthreads_or(needs_policy);
  bool hit_only = Indexed && !policy_required;
  if (!hit_only) {
    if (samples <= kCachedCostRows)
      for (int32_t i = threadIdx.x; i < samples * kCostColumns; i += blockDim.x)
        cached_costs[i] = costs[i];
    if (local_top) {
      const int32_t* sizes = index_data + 2 * layers * experts;
      for (int32_t i = threadIdx.x; i < 2 * layers + 1; i += blockDim.x)
        local_top[i] = sizes[layers + i];
      for (int32_t l = threadIdx.x; l < layers; l += blockDim.x) {
        if (sizes[l] > 0) local_top_heat[l] = frequency[l * experts + index_data[l * experts]];
      }
    }
    // Every warp compacts one expert word. Prefixes preserve expert-id order,
    // including duplicate and masked routes, without a serial insertion sort.
    int32_t lane = threadIdx.x % 32;
    for (int32_t word = threadIdx.x / 32; word < (experts + 31) / 32; word += blockDim.x / 32) {
      int32_t e = word * 32 + lane;
      bool active = e < experts && histogram[e] > 0;
      bool cold = e < experts && layer_lookup[e] >= 0 && !active && layer_frequency[e] < 0.5;
      uint32_t routes_in_word = __ballot_sync(0xffffffff, active);
      uint32_t cold_in_word = __ballot_sync(0xffffffff, cold);
      if (!lane) {
        route_mask[word] = routes_in_word;
        cold_mask[word] = cold_in_word;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t count = 0;
      for (int32_t word = 0; word < (experts + 31) / 32; ++word) {
        route_offset[word] = count;
        count += __popc(route_mask[word]);
      }
      requested_count = count;
    }
    __syncthreads();
    for (int32_t e = threadIdx.x; e < experts; e += blockDim.x)
      if (histogram[e]) requested[route_offset[e / 32] + __popc(route_mask[e / 32] & ((1u << (e % 32)) - 1))] = e;
  }
  // Updating every heat before heap repair is safe with bottom-up heapify.
  // Nodes on the same level own disjoint subtrees; all lower levels finish
  // before their parents. Heap shape may change, but the minimum tie-break
  // and all LFU admission decisions remain identical.
  if (Indexed) {
    HybridHeapIndex index(
        index_data,
        frequency,
        layers,
        experts,
        index_slots,
        layer,
        layer_heap,
        layer_position,
        &layer_heap_size,
        layer_frequency,
        local_top,
        local_top_heat);
    int32_t width = 1;
    while (width * 2 <= layer_heap_size / 2)
      width *= 2;
    for (; width; width /= 2) {
      for (int32_t i = width - 1 + threadIdx.x; i < 2 * width - 1 && i < layer_heap_size / 2; i += blockDim.x)
        index.sift_expert(layer, i);
      __syncthreads();
    }
  }
  if (hit_only) {
    // Heats can change heap order even without misses. Retain exact bottom-up
    // repair, then update only this layer's global minimum using the live top
    // heap. Avoid loading/storing the entire top index or reading cost samples.
    if (threadIdx.x == 0) {
      HybridHeapIndex index(
          index_data,
          frequency,
          layers,
          experts,
          index_slots,
          layer,
          layer_heap,
          layer_position,
          &layer_heap_size,
          layer_frequency);
      index.refresh_top(layer, layer_heap_size > 0);
      ++stats[1];
    }
    for (int32_t e = threadIdx.x; e < experts; e += blockDim.x) {
      global_layer_frequency[e] = layer_frequency[e];
      index_data[layer * experts + e] = layer_heap[e];
      index_data[layers * experts + layer * experts + e] = layer_position[e];
    }
    for (int32_t route = threadIdx.x; route < routes; route += blockDim.x) {
      int32_t e = ids[route];
      cache_ids[route] = e >= 0 ? layer_lookup[e] : -1;
    }
    // cpu_ids and plan headers were cleared before validation. Never publish
    // a CPU request or change cache payload ownership on this path.
    return;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    HybridHeapIndex index(
        index_data,
        frequency,
        layers,
        experts,
        index_slots,
        Indexed ? layer : -1,
        layer_heap,
        layer_position,
        &layer_heap_size,
        layer_frequency,
        local_top,
        local_top_heat);
    if (Indexed) {
      layer_was_present = index.size_for(layer) > 0;
      index.refresh_top(layer, layer_was_present);
    }
    int32_t initial_misses = 0, total_routes = 0;
    for (int32_t i = 0; i < requested_count; ++i) {
      int32_t e = requested[i];
      total_routes += histogram[e];
      if (layer_lookup[e] >= 0) continue;
      ++initial_misses;
      int32_t pos = active_count;
      while (pos && (layer_frequency[active[pos - 1]] < layer_frequency[e] ||
                     (layer_frequency[active[pos - 1]] == layer_frequency[e] && active[pos - 1] > e))) {
        active[pos] = active[pos - 1];
        --pos;
      }
      active[pos] = e;
      ++active_count;
    }
    int32_t free_slots[kRoutes], free_count = 0, next_free = 0;
    HybridVictim victims[kRoutes];
    int32_t victim_count = 0;
    // Gather free slots in ascending order and the globally coldest victims
    // once. Active residents of this layer cannot be replaced in this window.
    for (int32_t s = 0; !Indexed && active_count && s < slots; ++s) {
      int32_t old_layer = owners[s], old = keys[s];
      if (old_layer == -2) continue;
      if (old < 0) {
        if (free_count < kRoutes) free_slots[free_count++] = s;
        continue;
      }
      if (old_layer == layer && histogram[old]) continue;
      HybridVictim candidate{frequency[static_cast<int64_t>(old_layer) * experts + old], old_layer, old, s};
      if (victim_count < kRoutes)
        victim_push(victims, &victim_count, candidate);
      else if (victim_less(candidate, victims[0]))
        victim_replace_root(victims, victim_count, candidate);
    }
    // Heap extraction yields descending order; reverse to consume the coldest
    // first while preserving the former (heat, layer, expert) tie break.
    HybridVictim coldest[kRoutes];
    int32_t victim_total = victim_count, next_victim = 0;
    for (int32_t i = victim_total - 1; i >= 0; --i)
      coldest[i] = victim_pop_max(victims, &victim_count);

    for (int32_t i = 0; i < active_count; ++i) {
      int32_t e = active[i], slot = -1;
      if (Indexed)
        slot = index.pop_free();
      else if (next_free < free_count)
        slot = free_slots[next_free++];
      if (slot < 0 && layer_frequency[e] >= admission &&
          (Indexed ? index.protect_minimum(layer, histogram) : next_victim < victim_total)) {
        HybridVictim victim = Indexed ? index.minimum() : coldest[next_victim];
        if (layer_frequency[e] > victim.heat) {
          if (Indexed) {
            slot = victim.layer == layer ? layer_lookup[victim.expert]
                                         : lookup[static_cast<int64_t>(victim.layer) * experts + victim.expert];
            index.erase(victim.layer, victim.expert);
          } else {
            ++next_victim;
            slot = victim.slot;
          }
          if (victim.layer == layer)
            layer_lookup[victim.expert] = -1;
          else
            lookup[static_cast<int64_t>(victim.layer) * experts + victim.expert] = -1;
          keys[slot] = -1;
          ++stats[0];
        }
      }
      if (slot < 0) continue;
      owners[slot] = layer;
      keys[slot] = e;
      layer_lookup[e] = slot;
      int32_t n = cache_plan[1]++;
      cache_plan[4 + 2 * n] = e;
      cache_plan[5 + 2 * n] = slot;
    }
    // Match GlobalLfu's post-plan cold reclaim and unused ownership release.
    for (int32_t word = 0; word < (experts + 31) / 32; ++word) {
      uint32_t mask = cold_mask[word];
      for (int32_t bit = 0; mask; ++bit, mask >>= 1) {
        if (!(mask & 1)) continue;
        int32_t e = word * 32 + bit;
        int32_t slot = layer_lookup[e];
        if (slot >= 0 && !histogram[e] && layer_frequency[e] < 0.5f) {
          if (Indexed) index.erase(layer, e);
          layer_lookup[e] = keys[slot] = -1;
          owners[slot] = -1;
          ++stats[0];
          if (Indexed) {
            index.push_free(slot);
            continue;
          }
          // Cold reclaim becomes available to the forced-placement phase in
          // this same call. Keep the remaining free prefix sorted by slot id.
          int32_t pos = next_free;
          while (pos < free_count && free_slots[pos] < slot)
            ++pos;
          if (pos < kRoutes) {
            if (free_count < kRoutes) ++free_count;
            for (int32_t j = free_count - 1; j > pos; --j)
              free_slots[j] = free_slots[j - 1];
            free_slots[pos] = slot;
          }
        }
      }
    }
    if (!Indexed)
      for (int32_t s = 0; s < slots; ++s)
        if (owners[s] == layer && keys[s] < 0) owners[s] = -1;
    for (int32_t i = 0; i < requested_count; ++i) {
      int32_t e = requested[i];
      if (layer_lookup[e] >= 0) continue;
      int32_t pos = rejected_count;
      while (pos && histogram[rejected[pos - 1]] > histogram[e]) {
        rejected[pos] = rejected[pos - 1];
        --pos;
      }
      rejected[pos] = e;
      ++rejected_count;
    }
    double best = std::numeric_limits<double>::infinity();
    int32_t cpu_routes = 0;
    for (int32_t k = 1; k <= rejected_count; ++k) {
      cpu_routes += histogram[rejected[k - 1]];
      // Scalar buckets stay sorted on upload. Search only the matching
      // bucket; table growth no longer multiplies every prefix/profile scan.
      int32_t target[6] = {
          tokens, total_routes - cpu_routes, cpu_routes, k, rejected_count, cache_plan[1] + rejected_count - k};
      int32_t lo = 0, hi = samples;
      while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        const double* row = policy_costs + mid * kCostColumns;
        int32_t cmp = 0;
        for (int32_t i = 0; i < 6; ++i) {
          double value = row[i < 5 ? i : 6];
          if (value != target[i]) {
            cmp = value < target[i] ? -1 : 1;
            break;
          }
        }
        if (cmp < 0)
          lo = mid + 1;
        else
          hi = mid;
      }
      for (int32_t n = lo; n < samples; ++n) {
        const double* row = policy_costs + n * kCostColumns;
        if (row[0] != tokens || row[1] != total_routes - cpu_routes || row[2] != cpu_routes || row[3] != k ||
            row[4] != rejected_count || row[6] != target[5])
          break;
        bool match = true;
        for (int32_t i = 0; i < rejected_count; ++i) {
          if (row[7 + kRoutes + i] != histogram[rejected[i]]) match = false;
          if (i < k && row[7 + i] != histogram[rejected[i]]) match = false;
        }
        if (match && row[5] < best) {
          best = row[5];
          selected = k;
        }
      }
    }
    // The B12X FC2 epilogue reads this device-side count. No host route
    // readback or separate merge launch is needed for GPU-only layers.
    cache_plan[2] = selected;
    // LFU rejection only gates CPU candidacy. Every remaining GPU expert
    // must occupy the same pool as hits and admitted experts. Keep the new
    // cache contents; there is no temporary payload or restoration copy.
    int32_t admitted = cache_plan[1];
    // Rebuild after cold reclamation only when forced GPU placement needs
    // victims. Free slots alone do not require a second pool scan.
    victim_total = 0;
    next_victim = 0;

    if (!Indexed && rejected_count - selected > free_count - next_free) {
      victim_count = 0;
      for (int32_t s = 0; s < slots; ++s) {
        int32_t old_layer = owners[s], old = keys[s];
        if (old_layer < 0 || old < 0) continue;
        if (old_layer == layer && histogram[old]) continue;
        HybridVictim candidate{frequency[static_cast<int64_t>(old_layer) * experts + old], old_layer, old, s};
        if (victim_count < kRoutes)
          victim_push(victims, &victim_count, candidate);
        else if (victim_less(candidate, victims[0]))
          victim_replace_root(victims, victim_count, candidate);
      }
      victim_total = victim_count;
      for (int32_t i = victim_total - 1; i >= 0; --i)
        coldest[i] = victim_pop_max(victims, &victim_count);
    }
    for (int32_t i = selected; i < rejected_count; ++i) {
      int32_t e = rejected[i], slot = -1;
      if (Indexed)
        slot = index.pop_free();
      else if (next_free < free_count)
        slot = free_slots[next_free++];
      if (slot < 0) {
        // The runtime validates capacity before handoff and after resize.
        // Fail closed if a direct operator caller violates that contract.
        if (!Indexed)
          while (next_victim < victim_total && keys[coldest[next_victim].slot] < 0)
            ++next_victim;
        if (Indexed ? !index.protect_minimum(layer, histogram) : next_victim == victim_total) {
          cache_plan[0] = 2;
          asm volatile("trap;");
          return;
        }
        HybridVictim victim = Indexed ? index.minimum() : coldest[next_victim++];
        slot = Indexed ? (victim.layer == layer ? layer_lookup[victim.expert]
                                                : lookup[static_cast<int64_t>(victim.layer) * experts + victim.expert])
                       : victim.slot;
        if (Indexed) index.erase(victim.layer, victim.expert);
        if (victim.layer == layer)
          layer_lookup[victim.expert] = -1;
        else
          lookup[static_cast<int64_t>(victim.layer) * experts + victim.expert] = -1;
        ++stats[0];
      }
      owners[slot] = layer;
      keys[slot] = e;
      layer_lookup[e] = slot;
      int32_t n = cache_plan[1]++;
      cache_plan[4 + 2 * n] = e;
      cache_plan[5 + 2 * n] = slot;
    }
    if (Indexed) {
      // Hits and newly placed experts were excluded from victim selection.
      // No victim query occurs within this batch, so repair the layer top once.
      bool was_present = index.size_for(layer) > 0;
      for (int32_t i = 0; i < requested_count; ++i) {
        int32_t e = requested[i];
        if (layer_lookup[e] >= 0 && index.positions_for(layer)[e] < 0) index.insert(layer, e, false);
      }
      index.refresh_top(layer, was_present);
    }
    stats[1] += active_count ? 0 : 1;  // Calls without an initial miss.
    stats[2] += initial_misses;
    stats[3] += admitted;
    stats[4] += rejected_count;
    stats[5] += selected;
    stats[6] += rejected_count - selected;
  }
  __syncthreads();
  for (int32_t e = threadIdx.x; e < experts; e += blockDim.x) {
    global_layer_lookup[e] = layer_lookup[e];
    global_layer_frequency[e] = layer_frequency[e];
    if (Indexed) {
      index_data[layer * experts + e] = layer_heap[e];
      index_data[layers * experts + layer * experts + e] = layer_position[e];
    }
  }
  if (Indexed && threadIdx.x == 0) index_data[2 * layers * experts + layer] = layer_heap_size;
  if (local_top)
    for (int32_t i = threadIdx.x; i < 2 * layers + 1; i += blockDim.x)
      index_data[2 * layers * experts + layers + i] = local_top[i];
  for (int32_t route = threadIdx.x; route < routes; route += blockDim.x) {
    int32_t e = ids[route];
    if (e < 0) continue;
    bool cpu = false;
    for (int32_t i = 0; i < selected; ++i)
      cpu |= rejected[i] == e;
    int32_t slot = layer_lookup[e];
    cpu_ids[route] = cpu ? e : -1;
    cache_ids[route] = slot >= 0 ? slot : -1;
  }
  if (!host_head) return;
  __syncthreads();
  if (!selected) return;
  for (int32_t i = threadIdx.x; i < routes; i += blockDim.x) {
    host_ids[i] = cpu_ids[i];
    host_weights[i] = weights[i];
  }
  for (int32_t i = threadIdx.x; i < tokens * hidden; i += blockDim.x)
    host_x[i] = x[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    host_head[1] = layer;
    host_head[2] = tokens;
    __threadfence_system();
    host_head[0] = sequence;
  }
}
// Only a selected CPU contribution introduces a device dependency. All-hit
// and all-GPU calls bypass the mailbox worker's completion counter entirely.
__global__ void hybrid_wait_kernel(
    const int32_t* __restrict__ cpu_ids,
    const volatile int32_t* completed,
    int32_t routes,
    int32_t sequence,
    const volatile int32_t* failed) {
  bool selected = false;
  for (int32_t i = 0; i < routes; ++i)
    selected |= cpu_ids[i] >= 0;
  if (!selected) return;
  while (*completed < sequence) {
    if (failed && *failed) {
      asm volatile("trap;");
      return;
    }
    __nanosleep(1000);
  }
}

// The CPU reads the same mapped pages after the routed event. Write the
// bounded verify batch directly to them instead of per-tensor cudaMemcpy calls.
__global__ void hybrid_stage_kernel(
    const uint16_t* x,
    const int32_t* ids,
    const float* weights,
    uint16_t* host_x,
    int32_t* host_ids,
    float* host_weights,
    volatile int32_t* host_head,
    int32_t elements,
    int32_t routes,
    int32_t tokens,
    int32_t layer,
    int32_t sequence) {
  bool selected = false;
  for (int32_t i = 0; i < routes; ++i)
    selected |= ids[i] >= 0;
  if (!selected) return;
  for (int32_t i = threadIdx.x; i < routes; i += blockDim.x)
    host_ids[i] = ids[i];
  for (int32_t i = threadIdx.x; i < elements; i += blockDim.x)
    host_x[i] = x[i];
  for (int32_t i = threadIdx.x; i < routes; i += blockDim.x)
    host_weights[i] = weights[i];
  // Every writer must flush its own payload before lane 0 publishes the head.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    host_head[1] = layer;
    host_head[2] = tokens;
    __threadfence_system();
    host_head[0] = sequence;
  }
}

// Bypass stale results on GPU-only frames. Cache-bypassing loads observe the
// CPU writes after the completion flag has been published and waited on.
__global__ void hybrid_result_kernel(
    const int32_t* cpu_ids, const float* host_result, float* output, int32_t elements, int32_t routes) {
  __shared__ bool selected;
  if (threadIdx.x == 0) {
    selected = false;
    for (int32_t i = 0; i < routes; ++i)
      selected |= cpu_ids[i] >= 0;
  }
  __syncthreads();
  for (int32_t i = threadIdx.x; i < elements; i += blockDim.x) {
    float value = 0;
    if (selected) asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(value) : "l"(host_result + i) : "memory");
    output[i] = value;
  }
}

// One output launch replaces wait, mapped result materialization, FP32 add,
// and BF16 conversion. GPU-only routes read neither completion nor mmap data.
__global__ void hybrid_merge_kernel(
    const int32_t* cpu_ids,
    const float* gpu,
    const float* host_result,
    const volatile int32_t* completed,
    const volatile int32_t* failed,
    bf16_t* output,
    int32_t elements,
    int32_t routes,
    int32_t sequence) {
  bool selected = __syncthreads_or(threadIdx.x < routes && cpu_ids[threadIdx.x] >= 0);
  if (threadIdx.x == 0) {
    if (selected) {
      while (*completed < sequence) {
        if (failed && *failed) {
          asm volatile("trap;");
          return;
        }
        __nanosleep(1000);
      }
      if (failed && *failed) {
        asm volatile("trap;");
        return;
      }
    }
  }
  __syncthreads();
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x) {
    float value = gpu[i];
    if (selected) {
      float contribution;
      asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(contribution) : "l"(host_result + i) : "memory");
      value += contribution;
    }
    output[i] = static_cast<bf16_t>(value);
  }
}
}  // namespace details

/// Perform device LFU admission and exact calibrated CPU subset selection.
template <bool Indexed>
inline void qwen38_hybrid_plan_impl(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView frequency,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView cache_ids,
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView cache_plan,
    tvm::ffi::TensorView stats,
    int64_t layer,
    double decay,
    double admission,
    tvm::ffi::TensorView* x,
    tvm::ffi::TensorView* weights,
    int64_t host_x,
    int64_t host_ids,
    int64_t host_weights,
    int64_t host_head,
    int64_t sequence,
    tvm::ffi::TensorView* heap_index = nullptr) {
  using namespace host;
  using sglang::details::kCostColumns;
  SymbolicSize tokens("tokens"), topk("topk"), layers("layers"), experts("experts"), slots("slots"), samples("samples"),
      plan_size("plan_size");
  SymbolicDevice device;
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
  TensorMatcher({layers, experts}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(lookup);
  TensorMatcher({layers, experts}).with_dtype<double>().with_device<kDLCUDA>(device).verify(frequency);
  TensorMatcher({slots}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(owners);
  TensorMatcher({slots}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(keys);
  TensorMatcher({samples, kCostColumns}).with_dtype<double>().with_device<kDLCUDA>(device).verify(costs);
  for (auto tensor : {cache_ids, cpu_ids})
    TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(tensor);
  TensorMatcher({plan_size}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cache_plan);
  TensorMatcher({7}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(stats);
  int64_t n = tokens.unwrap() * topk.unwrap();
  CHECK_HOST(tokens.unwrap() >= 1 && tokens.unwrap() <= 3 && n <= sglang::details::kRoutes);
  CHECK_HOST(experts.unwrap() > 0 && experts.unwrap() <= 512);
  CHECK_HOST(slots.unwrap() > 0 && slots.unwrap() <= std::numeric_limits<int32_t>::max());
  CHECK_HOST(plan_size.unwrap() >= 4 + 2 * n);
  CHECK_HOST(layer >= 0 && layer < layers.unwrap() && decay > 0 && decay <= 1 && admission >= 0);
  int32_t index_slots = 0;
  if (heap_index) {
    SymbolicSize capacity("index_capacity");
    TensorMatcher({capacity}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(*heap_index);
    int64_t fixed = 2 * layers.unwrap() * experts.unwrap() + 3 * layers.unwrap() + 4;
    CHECK_HOST(capacity.unwrap() >= fixed + slots.unwrap());
    index_slots = static_cast<int32_t>(capacity.unwrap() - fixed);
  }
  int32_t hidden = 0;
  if (x) {
    SymbolicSize width("hidden");
    TensorMatcher({tokens, width}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(*x);
    TensorMatcher({tokens, topk}).with_dtype<float>().with_device<kDLCUDA>(device).verify(*weights);
    CHECK_HOST(host_x > 0 && host_ids > 0 && host_weights > 0 && host_head > 0);
    CHECK_HOST(sequence > 0 && sequence < std::numeric_limits<int32_t>::max());
    hidden = static_cast<int32_t>(width.unwrap());
  }
  LaunchKernel(1, 128, device.unwrap())(
      sglang::details::hybrid_plan_kernel<Indexed>,
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<int32_t*>(lookup.data_ptr()),
      static_cast<double*>(frequency.data_ptr()),
      static_cast<int32_t*>(owners.data_ptr()),
      static_cast<int32_t*>(keys.data_ptr()),
      static_cast<const double*>(costs.data_ptr()),
      static_cast<int32_t*>(cache_ids.data_ptr()),
      static_cast<int32_t*>(cpu_ids.data_ptr()),
      static_cast<int32_t*>(cache_plan.data_ptr()),
      static_cast<int64_t*>(stats.data_ptr()),
      static_cast<int32_t>(tokens.unwrap()),
      static_cast<int32_t>(n),
      static_cast<int32_t>(experts.unwrap()),
      static_cast<int32_t>(slots.unwrap()),
      static_cast<int32_t>(samples.unwrap()),
      static_cast<int32_t>(layer),
      decay,
      admission,
      x ? static_cast<const uint16_t*>(x->data_ptr()) : nullptr,
      weights ? static_cast<const float*>(weights->data_ptr()) : nullptr,
      reinterpret_cast<uint16_t*>(host_x),
      reinterpret_cast<int32_t*>(host_ids),
      reinterpret_cast<float*>(host_weights),
      reinterpret_cast<volatile int32_t*>(host_head),
      hidden,
      static_cast<int32_t>(sequence),
      heap_index ? static_cast<int32_t*>(heap_index->data_ptr()) : nullptr,
      static_cast<int32_t>(layers.unwrap()),
      index_slots);
}

inline void qwen38_hybrid_plan(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView frequency,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView cache_ids,
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView cache_plan,
    tvm::ffi::TensorView stats,
    int64_t layer,
    double decay,
    double admission) {
  qwen38_hybrid_plan_impl<false>(
      ids,
      lookup,
      frequency,
      owners,
      keys,
      costs,
      cache_ids,
      cpu_ids,
      cache_plan,
      stats,
      layer,
      decay,
      admission,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      0);
}

inline void qwen38_hybrid_plan_stage(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView frequency,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView cache_ids,
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView cache_plan,
    tvm::ffi::TensorView stats,
    int64_t layer,
    double decay,
    double admission,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weights,
    int64_t host_x,
    int64_t host_ids,
    int64_t host_weights,
    int64_t host_head,
    int64_t sequence) {
  qwen38_hybrid_plan_impl<false>(
      ids,
      lookup,
      frequency,
      owners,
      keys,
      costs,
      cache_ids,
      cpu_ids,
      cache_plan,
      stats,
      layer,
      decay,
      admission,
      &x,
      &weights,
      host_x,
      host_ids,
      host_weights,
      host_head,
      sequence);
}

inline void qwen38_hybrid_plan_indexed(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView frequency,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView cache_ids,
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView cache_plan,
    tvm::ffi::TensorView stats,
    int64_t layer,
    double decay,
    double admission,
    tvm::ffi::TensorView heap_index) {
  qwen38_hybrid_plan_impl<true>(
      ids,
      lookup,
      frequency,
      owners,
      keys,
      costs,
      cache_ids,
      cpu_ids,
      cache_plan,
      stats,
      layer,
      decay,
      admission,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      &heap_index);
}

inline void qwen38_hybrid_plan_stage_indexed(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView lookup,
    tvm::ffi::TensorView frequency,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView cache_ids,
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView cache_plan,
    tvm::ffi::TensorView stats,
    int64_t layer,
    double decay,
    double admission,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weights,
    int64_t host_x,
    int64_t host_ids,
    int64_t host_weights,
    int64_t host_head,
    int64_t sequence,
    tvm::ffi::TensorView heap_index) {
  qwen38_hybrid_plan_impl<true>(
      ids,
      lookup,
      frequency,
      owners,
      keys,
      costs,
      cache_ids,
      cpu_ids,
      cache_plan,
      stats,
      layer,
      decay,
      admission,
      &x,
      &weights,
      host_x,
      host_ids,
      host_weights,
      host_head,
      sequence,
      &heap_index);
}

}  // namespace sglang

namespace sglang {
/// Wait only when the device planner assigned at least one route to CPU.
inline void qwen38_hybrid_wait(
    tvm::ffi::TensorView cpu_ids, tvm::ffi::TensorView completed, int64_t sequence, int64_t failed_address) {
  using namespace host;
  SymbolicSize tokens("tokens"), topk("topk");
  SymbolicDevice device;
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cpu_ids);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(completed);
  CHECK_HOST(sequence > 0 && sequence < std::numeric_limits<int32_t>::max());
  LaunchKernel(1, 1, device.unwrap())(
      sglang::details::hybrid_wait_kernel,
      static_cast<const int32_t*>(cpu_ids.data_ptr()),
      static_cast<const volatile int32_t*>(completed.data_ptr()),
      static_cast<int32_t>(tokens.unwrap() * topk.unwrap()),
      static_cast<int32_t>(sequence),
      reinterpret_cast<const volatile int32_t*>(failed_address));
}

/// Stage a small verify batch into CUDA-registered mmap pages.
inline void qwen38_hybrid_stage(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView weights,
    int64_t host_x,
    int64_t host_ids,
    int64_t host_weights,
    int64_t host_head,
    int64_t layer,
    int64_t sequence) {
  using namespace host;
  SymbolicSize tokens("tokens"), hidden("hidden"), topk("topk");
  SymbolicDevice device;
  TensorMatcher({tokens, hidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(x);
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids);
  TensorMatcher({tokens, topk}).with_dtype<float>().with_device<kDLCUDA>(device).verify(weights);
  CHECK_HOST(tokens.unwrap() >= 1 && tokens.unwrap() <= 3);
  CHECK_HOST(tokens.unwrap() * topk.unwrap() <= sglang::details::kRoutes);
  CHECK_HOST(host_x > 0 && host_ids > 0 && host_weights > 0 && host_head > 0);
  CHECK_HOST(layer >= 0 && layer < std::numeric_limits<int32_t>::max());
  CHECK_HOST(sequence > 0 && sequence < std::numeric_limits<int32_t>::max());
  LaunchKernel(1, 256, device.unwrap())(
      sglang::details::hybrid_stage_kernel,
      static_cast<const uint16_t*>(x.data_ptr()),
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const float*>(weights.data_ptr()),
      reinterpret_cast<uint16_t*>(host_x),
      reinterpret_cast<int32_t*>(host_ids),
      reinterpret_cast<float*>(host_weights),
      reinterpret_cast<volatile int32_t*>(host_head),
      static_cast<int32_t>(tokens.unwrap() * hidden.unwrap()),
      static_cast<int32_t>(tokens.unwrap() * topk.unwrap()),
      static_cast<int32_t>(tokens.unwrap()),
      static_cast<int32_t>(layer),
      static_cast<int32_t>(sequence));
}

/// Read the mapped CPU result after the completion wait on the same stream.
inline void qwen38_hybrid_result(tvm::ffi::TensorView cpu_ids, tvm::ffi::TensorView output, int64_t host_result) {
  using namespace host;
  SymbolicSize tokens("tokens"), topk("topk"), hidden("hidden");
  SymbolicDevice device;
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cpu_ids);
  TensorMatcher({tokens, hidden}).with_dtype<float>().with_device<kDLCUDA>(device).verify(output);
  CHECK_HOST(tokens.unwrap() >= 1 && tokens.unwrap() <= 3);
  CHECK_HOST(tokens.unwrap() * topk.unwrap() <= sglang::details::kRoutes);
  CHECK_HOST(host_result > 0);
  LaunchKernel(1, 256, device.unwrap())(
      sglang::details::hybrid_result_kernel,
      static_cast<const int32_t*>(cpu_ids.data_ptr()),
      reinterpret_cast<const float*>(host_result),
      static_cast<float*>(output.data_ptr()),
      static_cast<int32_t>(tokens.unwrap() * hidden.unwrap()),
      static_cast<int32_t>(tokens.unwrap() * topk.unwrap()));
}

inline void qwen38_hybrid_merge(
    tvm::ffi::TensorView cpu_ids,
    tvm::ffi::TensorView gpu,
    tvm::ffi::TensorView completed,
    tvm::ffi::TensorView output,
    int64_t host_result,
    int64_t sequence,
    int64_t failed_address,
    int64_t completed_address) {
  using namespace host;
  SymbolicSize tokens("tokens"), topk("topk"), hidden("hidden");
  SymbolicDevice device;
  TensorMatcher({tokens, topk}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cpu_ids);
  TensorMatcher({tokens, hidden}).with_dtype<float>().with_device<kDLCUDA>(device).verify(gpu);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(completed);
  TensorMatcher({tokens, hidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(output);
  CHECK_HOST(tokens.unwrap() >= 1 && tokens.unwrap() <= 3);
  CHECK_HOST(tokens.unwrap() * topk.unwrap() <= sglang::details::kRoutes);
  CHECK_HOST(host_result > 0 && sequence > 0 && sequence < std::numeric_limits<int32_t>::max());
  CHECK_HOST(completed_address >= 0);
  LaunchKernel((tokens.unwrap() * hidden.unwrap() + 1023) / 1024, 256, device.unwrap())(
      sglang::details::hybrid_merge_kernel,
      static_cast<const int32_t*>(cpu_ids.data_ptr()),
      static_cast<const float*>(gpu.data_ptr()),
      reinterpret_cast<const float*>(host_result),
      completed_address ? reinterpret_cast<const volatile int32_t*>(completed_address)
                        : static_cast<const volatile int32_t*>(completed.data_ptr()),
      reinterpret_cast<const volatile int32_t*>(failed_address),
      static_cast<bf16_t*>(output.data_ptr()),
      static_cast<int32_t>(tokens.unwrap() * hidden.unwrap()),
      static_cast<int32_t>(tokens.unwrap() * topk.unwrap()),
      static_cast<int32_t>(sequence));
}
}  // namespace sglang
