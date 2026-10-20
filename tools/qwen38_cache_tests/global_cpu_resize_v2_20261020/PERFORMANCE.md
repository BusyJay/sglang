# Full-model performance report

Measured 2026-10-21 (Asia/Shanghai), runtime source `6399d324a1`.
This supplements [DEFAULT_ON_VALIDATION.md](DEFAULT_ON_VALIDATION.md): that
report established correctness and bidirectional resizing, but its 8-token
responses did not establish sustained decode throughput.

## Measured results

**Performance regression found.** Default hybrid short-prompt decode is
36.34 tok/s versus 102.79 tok/s for the legacy GPU path (64.6% less
throughput). The 170K TTFT is 102.74 s versus 28.88 s (3.56 times as long).
Correctness and resize acceptance passing does not constitute a performance
acceptance. The default remains enabled as requested; these results do not
support deploying it as a performance improvement.

| Measurement | Default hybrid | Global GPU preferred | Legacy GPU |
| --- | ---: | ---: | ---: |
| Short decode median (tok/s) | 36.34 | 35.11 | 102.79 |
| Short decode after first 64 tokens, median (tok/s) | 38.05 | 35.66 | 107.52 |
| Short TTFT median (s) | 2.06 | 1.06 | 0.86 |
| Short E2E median, 512 output tokens (s) | 16.12 | 15.93 | 5.89 |
| Natural chat decode (tok/s) | 33.69 | 31.28 | 107.91 |
| Natural chat TTFT (s) | 1.59 | 0.73 | 0.52 |
| Natural chat E2E, 512 output tokens (s) | 16.76 | 17.06 | 5.25 |
| 170064-token TTFT (s) | 102.74 | 100.57 | 28.88 |
| 170064 input + 8 output E2E (s) | 103.12 | 100.84 | 29.05 |
| Post-resize short TTFT (s) | 3.89 | 3.88 | 0.78 |
| Post-resize short decode (tok/s) | 32.06 | 34.07 | 70.57 |
| Post-resize short E2E, 512 output tokens (s) | 19.83 | 18.88 | 8.02 |

Short decode ranges across three requests: **36.24–37.67**, **34.37–35.38**,
and **99.83–106.47 tok/s**, respectively. Effective 170K input/TTFT rates:
**1655.27**, **1691.05**, and **5888.85 input tok/s**. The 8-token long-request
decode sample is deliberately excluded from sustained-throughput comparisons.

Default hybrid improves short decode by only **3.5%** over GPU-preferred
global cache; its short E2E median is **1.2% longer**. Natural chat decode is
**7.7% higher**, with E2E **1.8% shorter**. These small differences come from
one ordered pass, with different generated continuations and MTP acceptance;
they do not establish a universal CPU crossover threshold. The much larger
legacy/global difference remains under GPU-preferred miss execution, so CPU
arithmetic alone does not explain this regression.

## Prefill is GPU compute; cache management still costs time

Prefill does **not** execute CPU experts. The target derives execution phase
from ForwardMode; only `decode` and `verify` enter `hybrid.run`. All prefill,
including small prefill batches, enters `hybrid.prefill`, fills missing weights
into the GPU arena, and runs the GPU MoE implementation. The benchmark explicitly
sets `SGLANG_QWEN38_CACHE_PREFILL_W4A16=0`, selecting GPU CUTLASS W4A4.
CPU-side routing/cache bookkeeping and host-to-device DMA remain involved.

The global prefill implementation leases the borrowed tail arena, waits for
outstanding cache work, and restores overwritten cached experts afterward.
This is a concrete overhead candidate, not a measured attribution of the full
slowdown. Decode likewise performs per-layer host routing/readback, ownership
management and result joins; profiling is needed to apportion those costs.
No optimization to those paths was made during this measurement.

Both global modes finished at KV stage 0, with 80000 backed tokens and 7006
expert slots after reaching 220032 tokens/6388 slots. Each reports five resize
events; default hybrid retained and relocated 614 experts, GPU-preferred 618.
The legacy mode also returned from stage 2 to stage 0. Recovery is functional,
but the post-resize request remains significantly slower on the new path.

Final global counters cover the entire ordered run, including prefill and
background arena restores; they are not decode-only transfer counts:

| Counter | Default hybrid | Global GPU preferred |
| --- | ---: | ---: |
| Decode/verify unique miss observations | 214179 | 252932 |
| All-GPU miss windows | 48 | 50607 |
| Fill/restore bytes (decimal TB) | 1.798 | 2.131 |
| Result joins | 57920 | 57536 |

At M<=3 and top-k=10, the default mode necessarily handled at least
214179 - 48×30 = **212739 miss observations on CPU**. Prefill has separate
GPU-miss counters and is excluded from this bound.

Machine-readable results, individual-case timing, MTP acceptance lengths,
CPU initialization and observed resize transitions are in
[performance-summary.json](performance-summary.json). Every request returned
the required output-token count and zero prefix-cache hits. This is a local
single-request performance comparison; vision, concurrency scaling, quality
benchmarks, per-token latency percentiles and isolated resize latency remain
outside its scope.

## Existing-log diagnosis: asynchronous fill versus global cache

The legacy baseline already uses a background transfer worker and asynchronous
CUDA copies. `async_pack_prefetch=true` is identical in all three commands and
controls startup pack prefetch, not the steady-state global admission worker.
Consequently these measurements do not support blaming asynchronous transfer
as a general mechanism. They locate the regression in the new global execution
path, whose background filling, arena restore, host policy and synchronization
are coupled.

The saved logs provide two concrete findings:

| Evidence | Default global | GPU-preferred global | Legacy |
| --- | ---: | ---: | ---: |
| Fused decode planning calls | 0 | 0 | 57151 |
| Device-gated decode calls | 0 | 0 | 57151 |
| Device-gated all-hit calls | 0 | 0 | 16107 |
| Shared-expert/fill overlap calls | 0 | 0 | 57151 |
| Approx. long-prefill foreground transfer (TB) | 0.603 | 0.586 | 0.666 |
| Same interval, global fill/restore transfer (TB) | 1.355 | 1.351 | N/A |
| Median interval between long-prefill log entries (s) | 5 | 5 | 1 |

The new branch bypasses the old device-side planning/gating and shared-fill
overlap paths. Each decode/verify layer instead reads back routes, coordinates
with the host worker and joins a Future; the GPU-preferred variant retains that
structure. Zero legacy counters mean these specific paths were bypassed, not
that no GPU work overlaps anywhere in the new implementation.

For prefill, the foreground copy count is actually lower on the global path,
but it also performs about **1.35 TB of global fill/restore copies** in the same
approximate interval. The sum is about **1.96 TB versus 0.67 TB** on the legacy
path. Code inspection confirms separate counters: `transferred_bytes` records
arena transfer, while `global_cpu_hybrid.fill_bytes` records `_copy` operations
for admission and arena restore. They must be added for this comparison. The
new prefill restores overwritten tail-cache weights and subsequently admits
requested experts through its background manager; these additional copies are
real even though all prefill expert computation runs on the GPU.

The sampled interval spans 1024 layer calls per mode and brackets most of the
170K prefill. Endpoints have 128-call snapshot granularity, so they include some
adjacent work and do not provide exact request-only byte counts. Log timestamps
have one-second precision and scheduler logging overlaps GPU execution; log gaps
are evidence of persistent chunk slowdown, not precise kernel durations.

The global manager also scans the full slot array while selecting slots for
each admission (`GlobalLfu.plan`), then waits for copy completion before cache
publication. Prefill waits for borrowed-arena restoration before using the
next layer. These are specific implementation candidates for the observed
regression. Existing counters do **not** measure their individual elapsed
costs. In particular, `result_join_waits` increments on every join and must not
be interpreted as a count or duration of actual blocking. Admission and restore
share one byte counter, so the logs cannot apportion their individual traffic.

Thus the actionable diagnosis is **new global-path management and extra
fill/restore work**, with loss of the old optimized decode paths. These logs
cannot establish an exact split between asynchronous admission, arena restore,
host LFU scanning, GPU kernels and synchronization, nor prove that a global
cache design is inherently slower.

Reproduce this analysis without starting a server:

```bash
python3 tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/analyze_performance_logs.py \
  tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/performance-20261021 \
  --output /tmp/qwen38-log-analysis.json
```

[log-analysis.json](log-analysis.json) retains the counters and exact raw-log
line ranges used for the sampled comparison. Raw server timestamps are UTC;
16:xx on October 20 corresponds to 00:xx on October 21 in Asia/Shanghai.

## Environment and method

- One RTX 5090 D (32 GB); Ryzen 9 9950X (16 physical cores), default hybrid
  workers on four physical cores (0/1/2/3). Existing services on GPU 1 stayed up.
- Local NVIDIA Qwen3.8-Flash-Next FP4-Local checkpoint, revision
  `b9ce18a6887e18c089fdea8509a33e63429a3300`; B12X W4A16 decode/verify,
  CUTLASS W4A4 prefill, pinned PLE, FP8 KV, BF16 GDN state.
- Image `swallow-sglang:nightly-dev-20261018-b234b5c3`, Torch 2.14.1+cu130,
  FlashInfer 0.7.2. MTP steps=2, three draft tokens, breakable CUDA graphs,
  concurrency=1, chunk=8192, context limit=220000, initial expert budget=6388.
- Each mode starts a fresh server, performs one 128-token warmup, three paired
  independent 1088-token synthetic prompts with 512 generated tokens, one
  natural-language chat with 512 generated tokens, one 170064-token prompt
  with 8 generated tokens, then another 1088-token/512-token request.
  Synthetic prompts repeat a cache-related phrase after distinct 64-token
  prefixes; they are a microbenchmark, not a representative quality dataset.
- Paired input token IDs and random seed=42 are identical across modes.
  Temperature=0 and ignore_eos=true ensure fixed output lengths. In particular,
  forced continuation beyond EOS is possible. Each recorded cached_tokens must
  equal zero. Caches evolve during a mode; this is an ordered sequence, not
  an isolated cache-reset measurement for each request.
- TTFT is client wall time to the first SSE event containing output tokens.
  Decode rate is the output-token delta between the first and last token-bearing
  SSE events divided by their elapsed time. MTP can emit several tokens per
  event; the calculation counts tokens, not events. The separate after-64 rate
  excludes the first 64 generated tokens. Mean ms/token is the reciprocal of
  decode rate, not a per-token p50/p99 latency claim. Startup is excluded.
- Long-prompt input tokens / TTFT is an effective prefill rate including request
  handling, cache migration, and time to the first output; it is not pure kernel
  throughput. Post-resize TTFT includes the complete request setup and prefill,
  and does not isolate the resize operation's own latency.

Three modes distinguish the global-cache policy from CPU miss computation:

| Mode | Setting | Meaning |
| --- | --- | --- |
| Default hybrid | No CPU overrides; miss threshold 33 | Global LFU, CPU computes misses while GPU processes hits |
| Global GPU preferred | `SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD=1` | Same global implementation; prefer GPU for misses when owned capacity permits; CPU fallback remains possible |
| Legacy GPU | `SGLANG_ENABLE_QWEN38_CPU_HYBRID=0` | Existing per-layer adaptive GPU path; changes cache policy/layout as well as disabling CPU computation |

The default/legacy comparison measures the overall runtime-mode change. It
cannot attribute all differences to CPU arithmetic alone. Routing, MTP acceptance,
and cache contents can diverge between modes; acceptance lengths are retained
alongside timings. One sequential run per mode does not provide confidence
intervals, and the single natural-language case is not a broad workload claim.

## Reproduce and evidence

```bash
bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_performance.sh
```

The runner uses the existing local image/checkpoint/shared source cache, a
separate disposable container on GPU 0, and the checked-in
[performance-command.json](performance-command.json). It stops only the servers
it starts. `QWEN38_VERIFY_RESULTS` selects the output directory. Optional mode
selection, for example `--modes default global_gpu`, is passed to the benchmark.

[benchmark_performance.py](benchmark_performance.py) records all SSE timestamps,
responses, metadata, commands, and case timings in `performance.json` alongside
per-mode server logs. [summarize_performance.py](summarize_performance.py) rejects
incomplete runs or prefix hits and emits the compact summary. Raw evidence is
kept locally in `performance-20261021/`; the compact summary is checked in.
