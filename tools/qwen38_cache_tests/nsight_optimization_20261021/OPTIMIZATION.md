# Nsight optimization and acceptance, 2026-10-21

Default CPU/GPU hybrid remains enabled for the supported threaded adaptive LFU
W4A16 configuration. **Prefill expert computation remains entirely on GPU**, even
for a small prefill batch. The crossover applies only to decode/verify.

## Full-model performance

Same RTX 5090 D, Ryzen 9 9950X, four CPU cores (0/1/2/3), local FP4 checkpoint,
image, MTP configuration, memory budgets and workload as the
[original report](../global_cpu_resize_v2_20261020/PERFORMANCE.md).
No CPU enable, library or core overrides in the accepted default run.
No existing service was restarted. TP=1, concurrency=1.

| Measurement | Original default | Optimized default |
| --- | ---: | ---: |
| Short decode median, 3 × 1088 input / 512 output (tok/s) | 36.34 | 61.67 |
| Short decode range (tok/s) | 36.24–37.67 | 54.80–64.20 |
| Short TTFT median (s) | 2.059 | 1.040 |
| Short E2E median (s) | 16.122 | 9.325 |
| Natural chat decode, 91 input / 512 output (tok/s) | 33.69 | 55.08 |
| Natural chat TTFT (s) | 1.591 | 0.737 |
| Natural chat E2E (s) | 16.760 | 10.014 |
| 170064 input / 8 output TTFT (s) | 102.741 | 36.957 |
| Long-request E2E (s) | 103.123 | 37.085 |
| Post-resize short TTFT (s) | 3.894 | 0.927 |
| Post-resize short decode (tok/s) | 32.06 | 63.87 |
| Post-resize short E2E (s) | 19.834 | 8.928 |

Short decode improved **69.7%**; long TTFT fell **64.0%** (2.78× faster).
These are single sequential runs with three short samples, temperature zero,
salted distinct prefixes, and different generated continuations/MTP acceptance.
They establish this workload's improvement, not a general confidence interval.
The long request's eight output tokens are not a sustained decode benchmark.

**The global hybrid path still trails the legacy GPU path.** A fresh short-only
legacy control measured **106.51 tok/s**, range **103.16–107.86**; the original
full legacy pass measured 102.79 tok/s and 28.88 s long TTFT. The optimized
short median is still 42.1% below the fresh legacy median. The legacy control
bypasses these hybrid changes; it was measured before the final table/fill
optimizations. No claim that the regression is fully resolved.

Machine-readable evidence: [accepted default](performance-summary.json),
[fresh legacy control](legacy-control-summary.json),
[original paired runs](../global_cpu_resize_v2_20261020/performance-summary.json).

## Bottlenecks and changes

The original Nsight capture covers warmed 16K prefill / 8 output followed by
1K prefill / 256 output. GPU kernel-or-copy activity occupied only **46.4%**
of its 26.30 s GPU activity span, versus **91.8%** over 7.20 s for legacy.
This is a union of recorded GPU intervals, **not SM utilization**. The original
global path issued 810141 CUDA memcpy API calls and 40558 event synchronizations;
recorded H2D copies totaled 412.67 GB in that capture. Host gaps and redundant
transport were concrete bottlenecks; CPU expert arithmetic alone was not.

The final accepted capture reduced its GPU activity span to **10.30 s**, with
**62.2%** kernel-or-copy activity. The original and final traces use identical
request shapes; the final trace includes extra function annotations, so treat
these as diagnostic evidence rather than a controlled profiler-overhead study.

| Nsight measurement | Original global | Optimized global | Legacy GPU |
| --- | ---: | ---: | ---: |
| Profiled 16K request E2E (s) | 13.182 | 4.872 | 3.540 |
| Profiled 1K / 256-output request E2E (s) | 13.121 | 5.432 | 3.650 |
| GPU activity span (s) | 26.300 | 10.300 | 7.202 |
| Kernel-or-copy active fraction | 46.4% | 62.2% | 91.8% |
| CUDA memcpy API calls | 810141 | 141011 | 140946 |
| CUDA event synchronize calls | 40558 | 32358 | 1985 |
| Recorded H2D memcpy payload (GB, decimal) | 412.67 | 118.50 | 115.31 |

The smaller H2D memcpy count includes a transport change to mapped-host GPU
loads; it cannot be interpreted as a proportional reduction in PCIe traffic.
The full-run source-byte counters below account for batched fills separately.
Final CPU-wrapper ranges totaled 0.413 s across 1765 calls, overlapping recorded
GPU kernels for 0.069 s and prefill host ranges for **zero** seconds. These are
inclusive host wrapper intervals, not native CPU samples. The CPU-call reduction
also reflects the threshold change. Global planning still took 0.463 s,
route remapping 0.429 s, and arena waits 2.238 s inclusive. Ranges nest/overlap
and must not be added as disjoint wall-clock costs. Together with the remaining
37.8% GPU activity gaps, these show substantial host/fence work remains.

See [Nsight interval/API/function summaries](nsight-summary.json) and
[accepted trace request metadata](nsight-requests.json).

Changes follow the recorded costs:

1. **Prefill arena:** retire overwritten tail keys, exclude the leased tail from
   allocation, and admit selected experts directly from the GPU arena. Hold its
   lease until D2D completion; publish mappings only after that fence. This removes
   the old per-layer H2D restoration of unrelated tail experts. Exceptions leave
   unpublished destinations reserved. Resize still preserves surviving experts.
2. **Host planning/routing:** index per-layer ownership and free slots rather
   than scan the full 7K pool per candidate; populate pinned lookup tables through
   one NumPy assignment; reuse the existing CUDA route partition operator.
   A deterministic 9000-step differential replay, including 300 resize boundaries,
   matched the previous policy's admissions, evictions, keys and mappings.
3. **GPU fill dispatch:** reuse the registered-pack mailbox kernel, batching up
   to 32 experts per plan instead of host dispatch per expert/tensor. Keep the
   direct-DMA and pinned staging fallbacks. Physical traffic remains: GPU reads
   from mapped host memory do not appear as CUPTI H2D memcpy events. Do not infer
   transfer elimination from smaller memcpy counters. Full-run background
   `fill_bytes` fell from 1.798 TB to 580.42 GB; optimized arena admission added
   20.34 GB D2D. Foreground arena transfers are counted separately.
4. **Tiny GPU operations:** fuse routed FP32 alpha refresh into one CUDA kernel.
   Only routed slots are read/written, duplicates have one writer, unrelated rows
   remain untouched during concurrent fills, and padded alpha strides work.
   Rotating-buffer microbenchmarks measured Torch 8.1 µs versus fused 1.6–2.4 µs.
5. **CPU setup:** native ABI validation already checks values. Avoid duplicate
   Torch checks on the successful path, retain Python error types on failure,
   and cache immutable B12X unit decode tables rather than rebuilding two tables
   per miss expert. Hot synthetic CPU tests matched FP32 exactly; median 1-token
   calls fell 141.84→123.00 µs and 3-token calls 164.36→148.00 µs. This isolates
   table setup and is not a serving-throughput prediction.
6. **Miss crossover:** default threshold is now **3 distinct miss experts**,
   subject to capacity; 1–2 use CPU while resident GPU experts run. For larger
   miss sets, fill and execute one full GPU MoE instead of separate hit and miss
   MoEs. The worker publishes its branch before CPU math to preserve overlap.
   Earlier same-stage calibrations favored threshold 3 over 33 modestly
   (60.19 versus 58.77 tok/s); threshold 3 versus all-GPU preferred was also
   modest (60.86 versus 58.47). These are workload-specific tuning evidence.
   `SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD=33` restores the CPU-preferred policy.

The remaining architecture still reads routes back and joins host decisions per
MoE layer. A device-side all-hit fast path and further host/fill overlap remain
candidates; this patch does not pretend those waits have disappeared.

## Correctness and physical resize

Acceptance uses two distinct salted 170K contexts, alternating short requests,
then checks `17 + 25 = 42`. Each context now generates 128 tokens, ignoring EOS,
so periodic decode logging actually emits graph replay evidence. The earlier
8-token attempt completed both physical cycles and semantic checks but failed
the strict graph-evidence gate because no decode log was emitted; it was rerun
with longer outputs, rather than weakening that gate.

[The accepted serving summary](lifecycle-summary.json) observed stages
`0 → 1 → 2 → 0 → 1 → 2 → 1 → 0`, expert slots
`7006 / 6643 / 6388`, and physical KV capacities
`80000 / 160000 / 220032`. All 120 sampled budgets agreed. It counted
2279 successful CPU calls, 3998 CPU miss experts, 42885 retained experts
summed across eight resize events, and zero required relocations.

Retaining experts and relocating experts are different checks: an already-empty
retiring tail legitimately needs zero relocations. Serving verifies positive
retention plus correct physical budgets; the independent VMM test forces moves
and compares all payload bytes. CPU miss counters count successful native calls,
not routing attempts.

Focused checks passed: 77 unit tests; 45 real-GPU fill/arena/dispatch cases;
6 real GPU VMM stage transitions (two return cycles); 2 fused-alpha test methods (8 dtype/shape cases);
60 native scalar/partition/error checks across B12X and Triton semantics;
12 H=2560, M=1/2/3 CPU/GPU partition comparisons with exactly equal final BF16
outputs. Native invalid IDs and nonfinite inputs still fail as before.
These synthetic and semantic probes do not replace a broad model accuracy suite.
Vision, concurrency>1 and TP/EP>1 have not been accepted here. All applicable
pre-commit hooks and `git diff --check` passed for the changed files.

## Replay and artifacts

Use the local image/checkpoint and mounts encoded by the scripts, from repo root:

```bash
root="$PWD/tools/qwen38_cache_tests/nsight_optimization_20261021"
QWEN38_VERIFY_RESULTS="$root/performance-replay" bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_performance.sh --modes default
QWEN38_VERIFY_RESULTS="$root/lifecycle-replay" bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_default_lifecycle.sh
QWEN38_VERIFY_RESULTS="$root/nsight-replay" bash "$root/run_nsight.sh" --modes default --annotations
nsys export --type sqlite --output /tmp/qwen38-replay.sqlite "$root/nsight-replay/default.nsys-rep"
python "$root/summarize_nsight.py" replay=/tmp/qwen38-replay.sqlite --output "$root/nsight-replay/summary.json"
```

Profiling additionally requires host Nsight Systems (measured 2026.3.2) at
`/usr/lib/nsight-systems` and the `nvtx` Python wheel in
`/tmp/qwen38-profile-deps`, matching container CPython 3.12 (`nvtx==0.2.16` here).
Docker tracing used SYS_PTRACE and an unconfined seccomp profile. Native CPU IP
sampling is unavailable in this environment; conclusions use CUDA/NVTX/OSRT and
Python function ranges, not native instruction samples. Profiling overhead makes
trace request latency unsuitable as the unprofiled benchmark result.

Raw reports/logs remain in local `performance-accepted/`,
`lifecycle-graph-accepted/`, `nsight-accepted/`, and the original
`../global_cpu_resize_v2_20261020/nsight-baseline-20261021/` directories. Large
Nsight/SQLite files are kept locally rather than committed. Source scripts and
compact summaries are committed for replay and review.
