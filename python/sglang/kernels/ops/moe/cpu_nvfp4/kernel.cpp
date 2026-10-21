// SPDX-License-Identifier: Apache-2.0
// CPU experiment: CUTLASS-packed NVFP4, BF16 activation, FP32 accumulation.
#include "rows.hpp"
#include <algorithm>
#include <array>
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

template <int NT>
void expert(const uint8_t *w13, const uint8_t *w2, const uint8_t *s13,
            const uint8_t *s2, const uint16_t *x, int h, int i,
            const int *tokens, float g1, float g2, bool post_scale,
            Workers &workers) {
  workers.hidden.resize(NT * i);
  // B12X applies global scales after projection, so both decode tables are
  // constant. Keep one immutable table instead of rebuilding 8192 entries
  // for every miss expert. The legacy pre-scale math retains its own tables.
  static const a16global::Tables unit(1.0f);
  std::optional<a16global::Tables> scaled1, scaled2;
  if (!post_scale) {
    scaled1.emplace(g1);
    scaled2.emplace(g2);
  }
  const auto &table1 = post_scale ? unit : *scaled1;
  const auto &table2 = post_scale ? unit : *scaled2;
  const uint16_t *ap[NT], *hp[NT];
  for (int t = 0; t < NT; t++) {
    ap[t] = x + size_t(tokens[t]) * h;
    hp[t] = workers.hidden.data() + t * i;
  }
  const int threads = workers.cores.size();
  workers.dispatch([&](int thread) {
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
  });
  workers.dispatch([&](int thread) {
    for (int row = thread * 8; row < h; row += threads * 8) {
      int rows[8];
      for (int j = 0; j < 8; j++)
        rows[j] = row + j;
      float v[8][NT];
      a16global::project<NT, 8>(w2, s2, rows, hp, i, table2, v);
      for (int j = 0; j < 8; j++)
        for (int t = 0; t < NT; t++)
          workers.value[t * h + row + j] = v[j][t] * (post_scale ? g2 : 1.0f);
    }
  });
}
} // namespace
extern "C" int qwen38_cpu_abi() { return 2; }
extern "C" void qwen38_cpu_release() { cached.reset(); }
extern "C" const char *qwen38_cpu_error() { return error; }
static int run(const uint8_t *payload, uint64_t bytes, const uint64_t *offsets,
               int experts, int h, int i, const uint16_t *x, const int32_t *ids,
               const float *weights, int nt, int topk, int threads,
               const int *cores, int nc, float *out, bool post_scale) {
  try {
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
    for (int k = 0; k < 6; k++)
      if (offsets[k] > bytes || stride[k] * experts > bytes - offsets[k])
        throw std::runtime_error("CPU source range exceeds payload");
    auto &workers = pool(cores, threads);
    workers.sum.assign(nt * h, 0);
    workers.value.resize(3 * h);
    auto &sum = workers.sum;
    std::vector<int> active;
    for (int k = 0; k < nt * topk; k++)
      if (ids[k] >= 0)
        active.push_back(ids[k]);
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
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
                    *s13 = payload + offsets[2] + e * stride[2],
                    *s2 = payload + offsets[3] + e * stride[3];
      if (n == 1)
        expert<1>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers);
      else if (n == 2)
        expert<2>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers);
      else
        expert<3>(w13, w2, s13, s2, x, h, i, tokens, g1, g2, post_scale,
                  workers);
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
    for (size_t k = 0; k < sum.size(); k++) {
      if (!std::isfinite(sum[k]))
        throw std::runtime_error("nonfinite CPU output");
      out[k] = sum[k];
    }
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
