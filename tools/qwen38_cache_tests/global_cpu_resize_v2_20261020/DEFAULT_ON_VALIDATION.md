# Default CPU hybrid and bidirectional serving resize

Follow-up completed 2026-10-21 (Asia/Shanghai), based on `9e4c66a657`.
This supersedes the initial review's opt-in default and its serving-validation
limitations for the text/MTP workflow tested here.

## Result

CPU hybrid is on by default for the supported **threaded, adaptive LFU,
W4A16, prefill-arena** configuration. Other runtime/precision configurations
keep their existing path. Set `SGLANG_ENABLE_QWEN38_CPU_HYBRID=0` to opt out.
No running deployment configuration was changed or service restarted.

An empty `SGLANG_QWEN38_CPU_LIBRARY` builds a source/compiler-keyed library
under the user's cache directory, on one available CPU, with atomic publication
and a process lock. A supplied library path bypasses compilation. The cache
includes every native source/header and the compiler identity/flags. An empty
`SGLANG_QWEN38_CPU_CORES` selects up to four distinct physical cores from the
process affinity. The native workers block on condition variables and do not
use OpenMP, so unrelated OpenMP environment settings are no longer required.
AVX512 BF16/BW/VL/F and a compiler or prebuilt library remain prerequisites.

## Integration defects fixed

1. Global prefill slots could exceed the per-layer expert count. The CUDA
   route partition wrapper incorrectly rejected that valid limit, causing the
   server's first warmup prefill to fail. Expert-ID bounds and slot-prefix
   bounds are now independent; the GPU regression covers slots above 512.
2. The hybrid path bypassed the runtime's last-layer result-boundary work.
   Both prefill and decode/verify now queue staged trims and reborrow snapshots
   through the same `finish_layer` hook as legacy execution. They also record
   invocations, making the global mode visible in periodic cache statistics.
3. A completed 170K prefix counted as live demand, so the original serving
   run passed every generation probe but remained at the largest KV stage.
   Probe eligibility now considers protected tokens and the complete active
   request, including a chunked prefill's unprocessed suffix.
4. Probes attempted during a long prefill could consume the later short
   request's bounded eviction budget. New long-to-short transitions receive a
   fresh budget; repeated short decode steps do not reset it.
5. Sorting existing free pages alone was insufficient when a long cached
   prefix occupied all low addresses. Before allocating a new short request,
   settled CPU-backed nodes are demoted first, then a bounded cold-prefix
   eviction can make room below the desired physical boundary. Incoming long
   requests retain their prefix-cache opportunity. Active or transfer-locked
   pages remain protected. Physical unmap still requires the completed
   free-page snapshot and the existing result boundary.
6. Background global fills/restores now reuse the runtime's registered source
   and existing DMA operator. The fallback retains safe pinned staging. This
   removes the extra host copy and per-expert staging-reuse wait without
   publishing a mapping before transfer completion.

## Full-model evidence

Disposable container, RTX 5090 D, Ryzen 9 9950X, existing local FP4-Local
checkpoint/pack, FlashInfer 0.7.2, MTP steps=2, breakable target verify/draft
CUDA graphs, 8192-token prefill chunks, 220000 context limit. The Rust tree
extension was rebuilt and cached for this source checkout. No model download.
CPU enable/library/core environment overrides were deliberately absent.
Startup selected B12X, CPU cores 0/1/2/3, and the cached native library.

Final salted sequence: **1024 -> 170000 -> 1024 -> 170000 -> 1024**, followed
by the semantic probe `17 + 25`, which returned `42`. Salts prevent the second
long request from becoming a pure prefix hit. All requests passed. Serving
CUDA graph replay was observed.

| KV backing (tokens) | Global expert slots |
| --- | --- |
| 80000 | 7006 |
| 160000 | 6643 |
| 220032 (page-rounded) | 6388 |

Observed sequence:

```
KV:    80000 -> 160000 -> 220032 -> 80000 -> 160000 -> 220032 -> 160000 -> 80000
slots:  7006 ->   6643 ->   6388 ->  7006 ->   6643 ->   6388 ->   6643 ->  7006
```

The verifier checked 33 snapshots against physical stage/budget pairs, eight
resize events and 1085 retained-expert relocations. The final counters report
19086 miss-expert observations and only 48 all-GPU windows. Since M<=3 and
K=10 allow at most 30 unique misses per all-GPU window, at least 17646 miss
observations necessarily used CPU computation. Both return cycles recovered
the maximum expert budget; the cache was not simply cleared at transitions.

Machine-readable evidence: [validation-summary.json](validation-summary.json).
Raw final canary/report/server logs remain locally under
`lifecycle-reborrow-final/`. The earlier `lifecycle-default-long-rust/` run
passed generation but failed the new resize acceptance gate, demonstrating
why successful responses alone were insufficient.

The follow-up [performance report](PERFORMANCE.md) provides paired 512-token
streaming measurements and long-context timings.

## Other checks

- 58 focused CPU unit tests passed, including 23 subtests; 351 existing CPU
  regression tests passed.
- Both new cold-prefix/low-page regressions failed on the original
  `9e4c66a657` stage implementation and passed after the fix.
- CUDA partition tests: 3 passed, including noncontiguous global slots.
- Legacy GPU pipeline, borrowed-arena restore, hot-row resize, adaptive
  repartition and graph replay after migration: 6 passed on the 4080 SUPER.
- Direct/staged hybrid fill byte checks: 6 cases passed on the 4080 SUPER.
- CPU native build cache invalidation/failure publication and loader opt-out
  tests passed. Applicable pre-commit checks passed for all changed files.
  No Rust source changed in this follow-up.

## Replay

From the repository root, using the same locally available image/checkpoint
and mounts encoded in the runner:

```bash
bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_default_lifecycle.sh
```

`QWEN38_VERIFY_RESULTS` may point to a different output directory. The runner
then invokes `verify_lifecycle.py`; it rejects runs without two return cycles,
maximum expert-budget recovery, physical budget consistency, CPU work,
retained experts across resizing, graph replay, or the final semantic probe.
Forced relocations are checked separately by the GPU VMM test; retiring an
empty tail legitimately needs zero moves. The replay now generates 128 tokens
per context so serving decode reaches its periodic graph-replay log.

See [the optimization follow-up](../nsight_optimization_20261021/OPTIMIZATION.md)
for current behavior and measurements; the results above describe the initial integration.

The remaining scope limits are vision requests, concurrent requests, TP/EP>1,
full-model benchmark/accuracy suites and the miss crossover threshold. Long
prefill still restores borrowed arena bytes and can be expensive; this run is
a correctness/lifecycle gate and makes no 80/110 tok/s promise. A genuinely
active or transfer-locked high page can legitimately delay reborrowing.
