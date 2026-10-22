// SPDX-License-Identifier: Apache-2.0

// All arrays are views into one persistent int32 tensor. No decode-time
// allocation is needed. A layer heap stores expert ids, the top heap stores
// one layer id per nonempty layer, and the ring stores unoccupied slot ids.
struct HybridHeapIndex {
  int32_t *heap, *position, *size, *top, *top_position, *top_size;
  int32_t *free, *head, *tail, *free_size;
  const double* frequency;
  int32_t layers, experts, slots;
  int32_t cached_layer;
  int32_t *cached_heap, *cached_position, *cached_size;
  const double* cached_frequency;
  double* top_heat;

  __device__ HybridHeapIndex(
      int32_t* data,
      const double* heat,
      int32_t n_layers,
      int32_t n_experts,
      int32_t n_slots,
      int32_t local_layer = -1,
      int32_t* local_heap = nullptr,
      int32_t* local_position = nullptr,
      int32_t* local_size = nullptr,
      const double* local_frequency = nullptr,
      int32_t* local_top = nullptr,
      double* local_top_heat = nullptr)
      : frequency(heat),
        layers(n_layers),
        experts(n_experts),
        slots(n_slots),
        cached_layer(local_layer),
        cached_heap(local_heap),
        cached_position(local_position),
        cached_size(local_size),
        cached_frequency(local_frequency),
        top_heat(local_top_heat) {
    heap = data;
    if (!data) {
      position = size = top = top_position = top_size = nullptr;
      free = head = tail = free_size = nullptr;
      return;
    }
    position = heap + layers * experts;
    size = position + layers * experts;
    top = size + layers;
    top_position = top + layers;
    top_size = top_position + layers;
    free = top_size + 1;
    head = free + slots;
    tail = head + 1;
    free_size = tail + 1;
    if (local_top) {
      top = local_top;
      top_position = top + layers;
      top_size = top_position + layers;
    }
  }

  __device__ int32_t* heap_for(int32_t layer) const {
    return layer == cached_layer ? cached_heap : heap + layer * experts;
  }
  __device__ int32_t* positions_for(int32_t layer) const {
    return layer == cached_layer ? cached_position : position + layer * experts;
  }
  __device__ int32_t& size_for(int32_t layer) const {
    return layer == cached_layer ? *cached_size : size[layer];
  }
  __device__ const double* heat_for(int32_t layer) const {
    return layer == cached_layer ? cached_frequency : frequency + layer * experts;
  }

  __device__ bool less(int32_t layer, int32_t a, int32_t b) const {
    double x = heat_for(layer)[a], y = heat_for(layer)[b];
    return x < y || (x == y && a < b);
  }
  __device__ bool top_less(int32_t a, int32_t b) const {
    if (top_heat) return top_heat[a] < top_heat[b] || (top_heat[a] == top_heat[b] && a < b);
    int32_t ea = heap_for(a)[0], eb = heap_for(b)[0];
    double x = heat_for(a)[ea], y = heat_for(b)[eb];
    return x < y || (x == y && (a < b || (a == b && ea < eb)));
  }
  __device__ void swap_expert(int32_t layer, int32_t i, int32_t j) {
    int32_t* h = heap_for(layer);
    int32_t* p = positions_for(layer);
    int32_t a = h[i], b = h[j];
    h[i] = b;
    h[j] = a;
    p[a] = j;
    p[b] = i;
  }
  __device__ void fix_expert(int32_t layer, int32_t i) {
    int32_t* h = heap_for(layer);
    while (i > 0 && less(layer, h[i], h[(i - 1) / 2])) {
      int32_t j = (i - 1) / 2;
      swap_expert(layer, i, j);
      i = j;
    }
    sift_expert(layer, i);
  }
  __device__ void sift_expert(int32_t layer, int32_t i) {
    int32_t* h = heap_for(layer);
    int32_t n = size_for(layer);
    while (2 * i + 1 < n) {
      int32_t j = 2 * i + 1;
      if (j + 1 < n && less(layer, h[j + 1], h[j])) ++j;
      if (!less(layer, h[j], h[i])) break;
      swap_expert(layer, i, j);
      i = j;
    }
  }
  __device__ void swap_layer(int32_t i, int32_t j) {
    int32_t a = top[i], b = top[j];
    top[i] = b;
    top[j] = a;
    top_position[a] = j;
    top_position[b] = i;
  }
  __device__ void fix_top(int32_t i) {
    while (i > 0 && top_less(top[i], top[(i - 1) / 2])) {
      int32_t j = (i - 1) / 2;
      swap_layer(i, j);
      i = j;
    }
    while (2 * i + 1 < *top_size) {
      int32_t j = 2 * i + 1;
      if (j + 1 < *top_size && top_less(top[j + 1], top[j])) ++j;
      if (!top_less(top[j], top[i])) break;
      swap_layer(i, j);
      i = j;
    }
  }
  __device__ void refresh_top(int32_t layer, bool was_present) {
    bool present = size_for(layer) > 0;
    if (top_heat && present) top_heat[layer] = heat_for(layer)[heap_for(layer)[0]];
    if (was_present && !present) {
      int32_t i = top_position[layer], last = top[--*top_size];
      top_position[layer] = -1;
      if (i < *top_size) {
        top[i] = last;
        top_position[last] = i;
        fix_top(i);
      }
    } else if (!was_present && present) {
      int32_t i = (*top_size)++;
      top[i] = layer;
      top_position[layer] = i;
      fix_top(i);
    } else if (present) {
      fix_top(top_position[layer]);
    }
  }
  __device__ void erase(int32_t layer, int32_t expert) {
    int32_t* h = heap_for(layer);
    int32_t* p = positions_for(layer);
    int32_t i = p[expert];
    if (i < 0) return;  // Protected experts are temporarily outside the index.
    bool was_present = size_for(layer) > 0;
    int32_t last = h[--size_for(layer)];
    p[expert] = -1;
    if (i < size_for(layer)) {
      h[i] = last;
      p[last] = i;
      fix_expert(layer, i);
    }
    refresh_top(layer, was_present);
  }
  __device__ void detach(int32_t layer, int32_t expert, bool refresh = true) {
    int32_t* h = heap_for(layer);
    int32_t* p = positions_for(layer);
    int32_t i = p[expert];
    if (i < 0) return;
    bool was_present = size_for(layer) > 0;
    int32_t last = h[--size_for(layer)];
    p[expert] = -2;
    if (i < size_for(layer)) {
      h[i] = last;
      p[last] = i;
      fix_expert(layer, i);
    }
    if (refresh) refresh_top(layer, was_present);
  }
  __device__ void insert(int32_t layer, int32_t expert, bool refresh = true) {
    bool was_present = size_for(layer) > 0;
    int32_t i = size_for(layer)++;
    heap_for(layer)[i] = expert;
    positions_for(layer)[expert] = i;
    fix_expert(layer, i);
    if (refresh) refresh_top(layer, was_present);
  }
  __device__ int32_t pop_free() {
    if (!*free_size) return -1;
    int32_t slot = free[*head];
    *head = (*head + 1) % slots;
    --*free_size;
    return slot;
  }
  __device__ void push_free(int32_t slot) {
    free[*tail] = slot;
    *tail = (*tail + 1) % slots;
    ++*free_size;
  }
  __device__ HybridVictim minimum() const {
    int32_t layer = top[0], expert = heap_for(layer)[0];
    return {top_heat ? top_heat[layer] : heat_for(layer)[expert], layer, expert, -1};
  }
  __device__ bool protect_minimum(int32_t layer, const int32_t* histogram) {
    while (*top_size) {
      int32_t first = top[0], expert = heap_for(first)[0];
      if (first != layer || !histogram[expert]) return true;
      // Active readers are ineligible victims, but normally never reach the
      // cold root. Detach only those that actually obstruct a victim query.
      detach(layer, expert);
    }
    return false;
  }
};
