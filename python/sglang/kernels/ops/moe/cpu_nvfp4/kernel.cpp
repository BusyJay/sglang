// SPDX-License-Identifier: Apache-2.0
// CPU experiment: CUTLASS-packed NVFP4, BF16 activation, FP32 accumulation.
#include "rows.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
struct CsfPlane {
  const uint8_t *storage;
  int rows, columns;
  uint8_t *expanded;
  const uint32_t *external = nullptr;
};
static void expand_plane(const CsfPlane &p, uint8_t *out, int thread,
                         int threads) {

  const size_t tiles = size_t(p.rows) * p.columns / 512;
  const auto *metadata = reinterpret_cast<const uint32_t *>(
      p.storage + 1024 + p.rows * (1 + p.columns / 2));
  const auto *records = p.external ? p.external : metadata + tiles * 8;
  // Decode regular values in native 128-word tiles, then patch only the
  // bitmap exceptions. This keeps division and metadata out of the SIMD loop.
  const __m128i nibble = _mm_set1_epi8(15);
  const __m128i spread =
      _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
  for (size_t tile = tiles * thread / threads;
       tile < tiles * (thread + 1) / threads; ++tile) {
    const size_t slab = tile / (p.columns / 4);
    const auto *fixed = p.storage + 1024 + slab * 128 * (1 + p.columns / 2);
    const auto *codes = fixed + 128 + (tile % (p.columns / 4)) * 256;
    for (size_t lane = 0; lane < 128; lane += 4) {
      uint32_t bases;
      memcpy(&bases, fixed + lane, 4);
      __m128i code = _mm_loadl_epi64((const __m128i *)(codes + lane * 2));
      __m128i lo = _mm_and_si128(code, nibble);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(code, 4), nibble);
      __m128i value =
          _mm_add_epi8(_mm_unpacklo_epi8(lo, hi),
                       _mm_shuffle_epi8(_mm_cvtsi32_si128(bases), spread));
      _mm_storeu_si128((__m128i *)(out + tile * 512 + lane * 4), value);
    }
    const auto *meta = metadata + tile * 8;
    size_t rec = meta[0];
    for (int group = 0; group < 4; ++group) {
      uint32_t mask = meta[2 + group];
      while (mask) {
        unsigned lane = group * 32 + __builtin_ctz(mask);
        memcpy(out + tile * 512 + lane * 4, records + rec++, 4);
        mask &= mask - 1;
      }
    }
  }
}

thread_local char error[256];
class Workers {
  std::mutex mutex;
  std::condition_variable work, finished;
  std::vector<std::thread> workers;
  std::function<void(int)> task;
  size_t generation = 0;
  int pending = 0, ready = 0;
  bool stopping = false;
  std::string failure;

public:
  const std::vector<int> cores;
  std::vector<uint16_t> hidden;
  std::vector<uint8_t> expanded;
  std::vector<float> sum, value;
  explicit Workers(const int *ids, int count) : cores(ids, ids + count) {
    for (int index = 0; index < count; index++)
      workers.emplace_back([this, index] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cores[index], &mask);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        std::unique_lock<std::mutex> lock(mutex);
        if (rc)
          failure = "cannot pin CPU expert worker: " + std::to_string(rc);
        ready++;
        finished.notify_one();
        size_t seen = 0;
        while (true) {
          work.wait(lock, [&] { return stopping || generation != seen; });
          if (stopping)
            return;
          seen = generation;
          auto call = task;
          lock.unlock();
          call(index);
          lock.lock();
          if (--pending == 0)
            finished.notify_one();
        }
      });
    std::unique_lock<std::mutex> lock(mutex);
    finished.wait(lock, [&] { return ready == int(workers.size()); });
    if (!failure.empty()) {
      stopping = true;
      work.notify_all();
      lock.unlock();
      for (auto &worker : workers)
        worker.join();
      throw std::runtime_error(failure);
    }
  }
  ~Workers() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      work.notify_all();
    }
    for (auto &worker : workers)
      worker.join();
  }
  void dispatch(std::function<void(int)> call) {
    std::unique_lock<std::mutex> lock(mutex);
    task = std::move(call);
    pending = workers.size();
    generation++;
    work.notify_all();
    finished.wait(lock, [&] { return pending == 0; });
    task = {};
  }
};
thread_local std::unique_ptr<Workers> cached;

Workers &pool(const int *cores, int count) {
  if (!cached || cached->cores != std::vector<int>(cores, cores + count))
    cached = std::make_unique<Workers>(cores, count);
  return *cached;
}

// Workers synchronize only inside an active expert task. The pool still sleeps
// between calls; occasionally yield if another worker has been descheduled.
class PhaseBarrier {
  alignas(64) std::atomic<unsigned> remaining;
  alignas(64) std::atomic<unsigned> phase{0};
  const unsigned participants;

public:
  explicit PhaseBarrier(unsigned n) : remaining(n), participants(n) {}
  void wait() {
    const unsigned seen = phase.load(std::memory_order_relaxed);
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // The acq_rel arrival chain publishes every worker's writes before
      // releasing the next phase (decoded scales, then hidden activations).
      remaining.store(participants, std::memory_order_relaxed);
      phase.fetch_add(1, std::memory_order_release);
    } else {
      unsigned spins = 0;
      while (phase.load(std::memory_order_acquire) == seen) {
        _mm_pause();
        if (++spins == 4096) {
          std::this_thread::yield();
          spins = 0;
        }
      }
    }
  }
};

template <int NT>
void expert_worker(const uint8_t *w13, const uint8_t *w2,
                   const uint8_t *s13, const uint8_t *s2, const uint16_t *x,
                   int h, int i, const int *tokens, float g1, float g2,
                   bool post_scale, Workers &workers,
                   const CsfPlane (&planes)[2], const a16global::Tables &table1,
                   const a16global::Tables &table2, PhaseBarrier &barrier,
                   int thread, const float *route_weights = nullptr) {
  const uint16_t *ap[NT], *hp[NT];
  for (int t = 0; t < NT; t++) {
    ap[t] = x + size_t(tokens[t]) * h;
    hp[t] = workers.hidden.data() + t * i;
  }
  const int threads = workers.cores.size();
    if (planes[0].storage || planes[1].storage) {
      for (int p = 0; p < 2; ++p)
        if (planes[p].storage)
          expand_plane(planes[p], planes[p].expanded, thread, threads);
      barrier.wait();
    }
    for (int row = thread * 4; row < i; row += threads * 4) {
      int rows[8];
      for (int j = 0; j < 4; j++) {
        rows[2 * j] = i + row + j;
        rows[2 * j + 1] = row + j;
      }
      float v[8][NT];
      a16global::project<NT, 8>(w13, s13, rows, ap, h, table1, v);
      for (int j = 0; j < 4; j++)
        for (int t = 0; t < NT; t++) {
          float g = v[2 * j][t] * (post_scale ? g1 : 1.0f);
          float up = v[2 * j + 1][t] * (post_scale ? g1 : 1.0f);
          workers.hidden[t * i + row + j] =
              a16::to_bf16(g / (1 + std::exp(-g)) * up);
        }
    }
    barrier.wait();
    for (int row = thread * 8; row < h; row += threads * 8) {
      int rows[8];
      for (int j = 0; j < 8; j++)
        rows[j] = row + j;
      float v[8][NT];
      a16global::project<NT, 8>(w2, s2, rows, hp, i, table2, v);
      for (int j = 0; j < 8; j++)
        for (int t = 0; t < NT; t++)
        {
          float value = v[j][t] * (post_scale ? g2 : 1.0f);
          workers.value[t * h + row + j] = value;
          if (route_weights)
            workers.sum[tokens[t] * h + row + j] += route_weights[t] * value;
        }
    }
    // The next expert may reuse hidden, value and CSF expansion storage only
    // after every worker finishes this expert's FC2 and aggregation.
  if (route_weights) barrier.wait();
}

template <int NT>
void expert(const uint8_t *w13, const uint8_t *w2, const uint8_t *s13,
            const uint8_t *s2, const uint16_t *x, int h, int i,
            const int *tokens, float g1, float g2, bool post_scale,
            Workers &workers, const CsfPlane (&planes)[2]) {
  workers.hidden.resize(NT * i);
  static const a16global::Tables unit(1.0f);
  std::optional<a16global::Tables> scaled1, scaled2;
  if (!post_scale) {
    scaled1.emplace(g1);
    scaled2.emplace(g2);
  }
  const auto &table1 = post_scale ? unit : *scaled1;
  const auto &table2 = post_scale ? unit : *scaled2;
  PhaseBarrier barrier(workers.cores.size());
  workers.dispatch([&](int thread) {
    expert_worker<NT>(w13, w2, s13, s2, x, h, i, tokens, g1, g2,
                      post_scale, workers, planes, table1, table2, barrier,
                      thread);
  });
}

struct BatchExpert {
  const uint8_t *w13, *w2, *s13, *s2;
  CsfPlane planes[2];
  int tokens[3], count;
  float g1, g2, route_weights[3];
  std::unique_ptr<a16global::Tables> scaled1, scaled2;
};
} // namespace
extern "C" int qwen38_cpu_abi() { return 2; }
extern "C" int qwen38_cpu_csf_abi() { return 2; }
extern "C" int qwen38_cpu_csf_tiered_abi() { return 1; }
extern "C" void qwen38_cpu_release() { cached.reset(); }
extern "C" const char *qwen38_cpu_error() { return error; }
static int run(const uint8_t *payload, uint64_t bytes, const uint64_t *offsets,
               int experts, int h, int i, const uint16_t *x, const int32_t *ids,
               const float *weights, int nt, int topk, int threads,
               const int *cores, int nc, float *out, bool post_scale,
               const uint64_t *csf_strides = nullptr,
               int32_t *completed = nullptr, int sequence = 0,
               bool batch_dispatch = false) {
  try {
    if ((completed && sequence <= 0) || (!completed && sequence != 0))
      throw std::runtime_error("invalid CPU completion sequence");
    if (!payload || !offsets || !x || !ids || !weights || !out ||
        experts <= 0 || h <= 0 || i <= 0 || h > 65536 || i > 65536 || h % 32 ||
        i % 32 || nt < 1 || nt > 3 || topk < 1 || topk > 32 || threads < 1 ||
        threads > 4 || nc != threads || !cores)
      throw std::runtime_error("unsupported CPU shape/thread configuration");
    for (int c = 0; c < nc; c++) {
      if (cores[c] < 0 || cores[c] >= CPU_SETSIZE)
        throw std::runtime_error("invalid CPU core");
      for (int d = 0; d < c; d++)
        if (cores[c] == cores[d])
          throw std::runtime_error("duplicate CPU core");
    }
    for (int k = 0; k < nt * topk; k++)
      if (ids[k] < -1 || ids[k] >= experts || !std::isfinite(weights[k]))
        throw std::runtime_error("invalid CPU route");
    for (int k = 0; k < nt * h; k++)
      if (!std::isfinite(a16::from_bf16(x[k])))
        throw std::runtime_error("nonfinite CPU input");
    size_t stride[] = {size_t(2 * i) * h / 2,
                       size_t(h) * i / 2,
                       size_t((2 * i + 127) / 128 * 128) *
                           ((h / 16 + 3) / 4 * 4),
                       size_t((h + 127) / 128 * 128) * ((i / 16 + 3) / 4 * 4),
                       4,
                       4};
    size_t source_stride[6];
    std::copy(stride, stride + 6, source_stride);
    if (csf_strides) {
      for (int p = 0; p < 2; p++)
        if (csf_strides[p])
          source_stride[2 + p] = csf_strides[p];
    }
    for (int k = 0; k < 6; k++)
      if (offsets[k] > bytes ||
          source_stride[k] > (bytes - offsets[k]) / experts)
        throw std::runtime_error("CPU source range exceeds payload");
    auto &workers = pool(cores, threads);
    workers.sum.assign(nt * h, 0);
    workers.value.resize(3 * h);
    if (csf_strides)
      workers.expanded.resize(stride[2] + stride[3]);
    auto &sum = workers.sum;
    std::vector<int> active;
    for (int k = 0; k < nt * topk; k++)
      if (ids[k] >= 0)
        active.push_back(ids[k]);
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    std::vector<BatchExpert> batch;
    if (batch_dispatch) batch.reserve(active.size());
    for (int e : active) {
      int tokens[3], n = 0;
      for (int t = 0; t < nt; t++) {
        bool found = false;
        for (int k = 0; k < topk; k++)
          found |= ids[t * topk + k] == e;
        if (found)
          tokens[n++] = t;
      }
      if (!n)
        continue;
      float g1, g2;
      memcpy(&g1, payload + offsets[4] + e * 4, 4);
      memcpy(&g2, payload + offsets[5] + e * 4, 4);
      // offsets[6:8] are per-layer reciprocal activation scales used by A16.
      float q1, q2;
      if (offsets[6] > bytes || bytes - offsets[6] < 4 || offsets[7] > bytes ||
          bytes - offsets[7] < 4)
        throw std::runtime_error("CPU layer scale out of bounds");
      memcpy(&q1, payload + offsets[6], 4);
      memcpy(&q2, payload + offsets[7], 4);
      g1 *= q1;
      g2 *= q2;
      if (!std::isfinite(g1) || !std::isfinite(g2))
        throw std::runtime_error("nonfinite CPU global scale");
      const uint8_t *w13 = payload + offsets[0] + e * stride[0],
                    *w2 = payload + offsets[1] + e * stride[1],
                    *s13 = payload + offsets[2] + e * source_stride[2],
                    *s2 = payload + offsets[3] + e * source_stride[3];
      CsfPlane planes[2]{};
      if (csf_strides) {
        const uint8_t *scales[] = {s13, s2};
        for (int p = 0; p < 2; p++) {
          if (!csf_strides[p])
            continue;
          int rows = ((p == 0 ? 2 * i : h) + 127) / 128 * 128;
          int cols = (((p == 0 ? h : i) / 16 + 3) / 4) * 4;
          size_t records_offset =
              1024 + size_t(rows) * (1 + cols / 2) + size_t(rows) * cols / 16;
          if (source_stride[2 + p] < records_offset)
            throw std::runtime_error("invalid CPU CSF slab size");
          uint32_t count, magic;
          memcpy(&count, scales[p] + 1008, 4);
          memcpy(&magic, scales[p] + 1012, 4);
          uint32_t tier_magic, base_stride, extent;
          uint64_t external_offset;
          memcpy(&tier_magic, scales[p] + 1016, 4);
          memcpy(&base_stride, scales[p] + 1020, 4);
          memcpy(&extent, scales[p] + 1000, 4);
          memcpy(&external_offset, scales[p] + 976, 8);
          const bool tiered = tier_magic == 0x31524954;
          const uint32_t *external = nullptr;
          if (tiered && (base_stride != source_stride[2 + p] ||
              (external_offset && (external_offset % 16 || extent % 16384 ||
               external_offset > bytes || extent > bytes - external_offset ||
               uint64_t(count) * 4 > extent)) || (!external_offset && extent)))
            throw std::runtime_error("invalid CPU tiered CSF bank");
          if (tiered && external_offset)
            external = reinterpret_cast<const uint32_t *>(payload + external_offset);
          if (magic != 0x32465343 ||
              (!external && count > (source_stride[2 + p] - records_offset) / 4))
            throw std::runtime_error("invalid CPU CSF header");
          const auto *meta = reinterpret_cast<const uint32_t *>(
              scales[p] + 1024 + size_t(rows) * (1 + cols / 2));
          uint32_t expected = 0;
          for (size_t tile = 0; tile < size_t(rows) * cols / 512;
               tile++, meta += 8) {
            if (meta[0] != expected || meta[7])
              throw std::runtime_error("invalid CPU CSF bitmap offset");
            uint32_t prefix = 0;
            for (int group = 0; group < 4; group++) {
              if (((meta[1] >> (group * 8)) & 255) != prefix)
                throw std::runtime_error("invalid CPU CSF bitmap rank");
              prefix += __builtin_popcount(meta[2 + group]);
            }
            expected += prefix;
            if (meta[6] != expected || expected > count)
              throw std::runtime_error("invalid CPU CSF bitmap extent");
          }
          if (expected != count)
            throw std::runtime_error("invalid CPU CSF replacement count");
          planes[p] = {scales[p], rows, cols,
                       workers.expanded.data() + (p ? stride[2] : 0), external};
        }
        auto *expanded = workers.expanded.data();
        if (csf_strides[0])
          s13 = expanded;
        if (csf_strides[1])
          s2 = expanded + stride[2];
      }
      if (batch_dispatch) {
        BatchExpert task{};
        task.w13 = w13; task.w2 = w2; task.s13 = s13; task.s2 = s2;
        task.planes[0] = planes[0]; task.planes[1] = planes[1];
        task.count = n; task.g1 = g1; task.g2 = g2;
        for (int j = 0; j < n; ++j) {
          task.tokens[j] = tokens[j];
          for (int k = 0; k < topk; ++k)
            if (ids[tokens[j] * topk + k] == e)
              task.route_weights[j] += weights[tokens[j] * topk + k];
        }
        if (!post_scale) {
          task.scaled1 = std::make_unique<a16global::Tables>(g1);
          task.scaled2 = std::make_unique<a16global::Tables>(g2);
        }
        batch.push_back(std::move(task));
        continue;
      }
      if (n == 1)
        expert<1>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers, planes);
      else if (n == 2)
        expert<2>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers, planes);
      else
        expert<3>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers, planes);
      for (int j = 0; j < n; j++) {
        int t = tokens[j];
        float rw = 0;
        for (int k = 0; k < topk; k++)
          if (ids[t * topk + k] == e)
            rw += weights[t * topk + k];
        for (int k = 0; k < h; k++)
          sum[t * h + k] += rw * workers.value[j * h + k];
      }
    }
    if (batch_dispatch && !batch.empty()) {
      workers.hidden.resize(3 * i);
      static const a16global::Tables unit(1.0f);
      PhaseBarrier barrier(threads);
      workers.dispatch([&](int thread) {
        for (const auto &task : batch) {
          const auto &table1 = post_scale ? unit : *task.scaled1;
          const auto &table2 = post_scale ? unit : *task.scaled2;
#define RUN_BATCH_EXPERT(N) \
          expert_worker<N>(task.w13, task.w2, task.s13, task.s2, x, h, i, \
                           task.tokens, task.g1, task.g2, post_scale, workers, \
                           task.planes, table1, table2, barrier, thread, \
                           task.route_weights)
          if (task.count == 1) RUN_BATCH_EXPERT(1);
          else if (task.count == 2) RUN_BATCH_EXPERT(2);
          else RUN_BATCH_EXPERT(3);
#undef RUN_BATCH_EXPERT
        }
      });
    }
    for (size_t k = 0; k < sum.size(); k++) {
      if (!std::isfinite(sum[k]))
        throw std::runtime_error("nonfinite CPU output");
      out[k] = sum[k];
    }
    // Publish while ctypes still has the GIL released. GPU result consumption
    // must not wait for Python statistics or the return-path GIL acquisition.
    if (completed)
      __atomic_store_n(completed, sequence, __ATOMIC_RELEASE);
    return 0;
  } catch (const std::exception &e) {
    snprintf(error, sizeof(error), "%s", e.what());
    return -1;
  }
}

// Triton rounds globally scaled weights to BF16; B12X scales each projection.
// Keep separate entry points so the serving adapter chooses the matching math.
#define QWEN38_CPU_ENTRY(NAME, POST_SCALE)                                     \
  extern "C" int NAME(const uint8_t *payload, uint64_t bytes,                  \
                      const uint64_t *offsets, int experts, int h, int i,      \
                      const uint16_t *x, const int32_t *ids,                   \
                      const float *weights, int nt, int topk, int threads,     \
                      const int *cores, int nc, float *out) {                  \
    return run(payload, bytes, offsets, experts, h, i, x, ids, weights, nt,    \
               topk, threads, cores, nc, out, POST_SCALE);                     \
  }
QWEN38_CPU_ENTRY(qwen38_cpu_run, false)
QWEN38_CPU_ENTRY(qwen38_cpu_run_b12x, true)
#undef QWEN38_CPU_ENTRY

extern "C" int qwen38_cpu_run_csf(const uint8_t *payload, uint64_t bytes,
                                  const uint64_t *offsets, int experts, int h,
                                  int i, const uint16_t *x, const int32_t *ids,
                                  const float *weights, int nt, int topk,
                                  int threads, const int *cores, int nc,
                                  float *out, const uint64_t *csf_strides,
                                  int post_scale) {
  if (!csf_strides) {
    snprintf(error, sizeof(error), "missing CPU CSF strides");
    return -1;
  }
  return run(payload, bytes, offsets, experts, h, i, x, ids, weights, nt, topk,
             threads, cores, nc, out, post_scale != 0, csf_strides);
}

// Additive ABI: old synchronous entry points retain their signatures. A failed
// computation leaves completion untouched; the runtime poisons before releasing
// its GPU consumer through the existing exception path.
extern "C" int qwen38_cpu_run_b12x_publish(
    const uint8_t *payload, uint64_t bytes, const uint64_t *offsets,
    int experts, int h, int i, const uint16_t *x, const int32_t *ids,
    const float *weights, int nt, int topk, int threads, const int *cores,
    int nc, float *out, int32_t *completed, int sequence) {
  return run(payload, bytes, offsets, experts, h, i, x, ids, weights, nt, topk,
             threads, cores, nc, out, true, nullptr, completed, sequence);
}
extern "C" int qwen38_cpu_run_csf_publish(
    const uint8_t *payload, uint64_t bytes, const uint64_t *offsets,
    int experts, int h, int i, const uint16_t *x, const int32_t *ids,
    const float *weights, int nt, int topk, int threads, const int *cores,
    int nc, float *out, const uint64_t *csf_strides, int post_scale,
    int32_t *completed, int sequence) {
  if (!csf_strides) {
    snprintf(error, sizeof(error), "missing CPU CSF strides");
    return -1;
  }
  return run(payload, bytes, offsets, experts, h, i, x, ids, weights, nt, topk,
             threads, cores, nc, out, post_scale != 0, csf_strides, completed,
             sequence);
}

// One native service owns an ordered mapped page. The Python launcher only
// starts/stops the service; route acquisition, dispatch and publication stay here.
extern "C" int qwen38_cpu_layered_service(
    const uint8_t *payload, uint64_t bytes, const uint64_t *all_offsets,
    const uint64_t *all_csf_strides, int layers, int experts, int h, int i,
    int topk, int threads, const int *cores, int32_t *page,
    int32_t *failed, int32_t *stop, int post_scale) {
  if (!page || !failed || !stop || !all_offsets || layers <= 0 || topk <= 0 ||
      topk > 10 || h <= 0 || !payload) return -1;
  auto *x = reinterpret_cast<const uint16_t*>(page + 3);
  auto *ids = reinterpret_cast<const int32_t*>(x + 3 * h);
  auto *weights = reinterpret_cast<const float*>(ids + 3 * topk);
  auto *result = const_cast<float*>(weights + 3 * topk);
  auto *completed = reinterpret_cast<int32_t*>(result + 3 * h);
  int32_t seen = 0;
  while (!__atomic_load_n(stop, __ATOMIC_ACQUIRE)) {
    int32_t seq = __atomic_load_n(page, __ATOMIC_ACQUIRE);
    if (seq == seen) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
      std::this_thread::yield();
      continue;
    }
    if (seq <= 0 || seq <= seen) {
      __atomic_store_n(failed, 1, __ATOMIC_RELEASE);
      return -1;
    }
    seen = seq;
    int layer = page[1], tokens = page[2];
    if (layer < 0 || layer >= layers || tokens < 1 || tokens > 3) {
      __atomic_store_n(failed, 1, __ATOMIC_RELEASE);
      return -1;
    }
    const uint64_t *strides = all_csf_strides ? all_csf_strides + 2 * layer : nullptr;
    int rc = run(payload, bytes, all_offsets + 8 * layer, experts, h, i,
                 x, ids, weights, tokens, topk, threads, cores, threads,
                 result, post_scale != 0, strides, completed, seq, true);
    if (rc) {
      __atomic_store_n(failed, 1, __ATOMIC_RELEASE);
      return rc;
    }
  }
  cached.reset();
  return 0;
}

extern "C" void qwen38_cpu_layered_stop(int32_t *stop) {
  if (stop) __atomic_store_n(stop, 1, __ATOMIC_RELEASE);
}
