# Asynchronous LFU admission, 2026-10-21

[Detailed timing analysis and Nsight handoff (中文)](HANDOFF.md). Three complete raw reports are committed under `captures/`; hashes and provenance are in `capture-manifest.json`.

Decode/verify misses execute on CPU while resident experts execute on GPU.
The cache manager admits a first touch into an empty slot. When slots are full,
admission requires decayed frequency >= 1.9 and a colder local victim. The old
miss-count crossover and zero-ownership synchronous GPU recovery are removed.
`SGLANG_QWEN38_CPU_GPU_MISS_THRESHOLD` is deprecated without replacement.

An admission becomes visible only after its copy completes. Its triggering
window can finish on CPU; even an empty cache does not force that window to wait
for a GPU fill.

## Scheduling and safety

- Layer entry queries the previous maintenance Future. If it is pending, use an
  immutable survivor mapping, with all unpublished destinations routed to CPU.
  Before planning has retired victims, the survivor mapping is empty. No serving
  thread joins a maintenance Future except at an explicit drain boundary.
- Management polls reader events before freeing any slot globally, then submits
  copies without waiting for them. It polls copy completion before publishing a
  mapping. An unfinished layer does not block another layer's maintenance.
- Windows overlapping maintenance still contribute CPU-miss heat. Route IDs are
  decoded once by the CPU worker and passed as immutable host tuples; management
  no longer rereads the tensor. Resident GPU work is submitted without a CPU
  branch-decision handshake. Inputs use one readback fence instead of two.
- The borrowed prefill arena is permanently excluded from persistent admissions.
  This removes per-layer arena reclamation/drain and arena-source copy waits.
  Background fills read the immutable host pack, including prefill admissions.
  Resize relocates surviving keys out of the new reserved tail before unmapping.
- The unregistered-pack fallback shares one pinned expert buffer. Tasks poll its
  DMA event before reusing it, so this fallback also avoids manager-side waits.
- Failed copies do not publish mappings. A shutdown sentinel cannot retire the
  worker while DMA remains outstanding; a timed-out close retains its source.
  Initial legacy handoff, CPU data dependencies, and physical resize/close keep
  their necessary synchronization.

The arena reservation is a deliberate capacity tradeoff: 512 experts plus B12X
slot zero, leaving 6493 / 6130 / 5875 usable slots at physical budgets
7006 / 6643 / 6388. `usable_cache_slots` reports this explicitly.

## Measurements

Same local model, RTX 5090 D, Ryzen 9 9950X, four physical CPU cores, TP=1,
concurrency=1, and workload as the previous performance harness. The fresh
baseline mounts the three runtime files from commit `1c44f94100` over the same
repository and runs `--modes default`.

| Measurement | Fresh baseline | Final run |
| --- | ---: | ---: |
| Background expert source bytes | 536.90 GB | 309.19 GB |
| Natural chat TTFT | 0.737 s | 0.657 s |
| Natural chat decode | 51.88 tok/s | 59.75 tok/s |
| 170K request TTFT | 37.456 s | 33.408 s |
| Short decode median, three samples | 63.64 tok/s | 61.50 tok/s |
| Short decode range | 58.92–64.00 tok/s | 61.39–66.07 tok/s |
| Post-resize short decode | 64.50 tok/s | 60.56 tok/s |

The first optimized pass measured a 65.75 tok/s short median, 32.650 s long TTFT,
and 298.11 GB of background fills. **A stable short-decode throughput improvement
is not established.** The final pass also briefly overlapped an unrelated
diagnostic SQLite query. These sequential measurements have differing generated
continuations/MTP acceptance and should not be treated as a confidence interval.
Post-resize decode remains slower. More CPU miss arithmetic and result joining
remain practical limits even though cache-maintenance waits have been removed.

Background source bytes fell 42.4% in the final run. These counters include
mapped-host GPU loads; they exclude foreground prefill-arena transfers. They are
not CUPTI memcpy totals, and prefill admission now contributes H2D rather than
the previous D2D source traffic.

The same-shaped Nsight workload compared with the saved accepted baseline shows
`cudaEventSynchronize` calls falling **32358 -> 5443 (83.2%)**. The manager thread
issued **zero synchronizing CUDA API calls**. `cudaMemcpyAsync` calls increased
141011 -> 155064: eager activation readback and more CPU result uploads trade
small copies for fewer decision/fill waits. GPU activity spans were 10.30 s and
10.15 s; this is not evidence of a large decode speedup. Function annotations and
generated routes differ between captures.

Compact evidence: [comparison](comparison.json),
[baseline](baseline-performance.json), [first pass](first-performance.json),
[final pass](performance.json), [Nsight](nsight-summary.json), and
[lifecycle](lifecycle.json). Selected raw Nsight reports and their logs are committed in `captures/`; intermediate runs remain local.

## Validation and replay

73 focused registered unit tests passed. The first-touch empty-slot regression
fails against the baseline policy and passes against this policy. Tests cover
blocked readers/copies, survivor routing, failed publication, worker shutdown
with DMA outstanding, and arena reservations changing during resize.

Real GPU checks covered registered DMA, mapped-host fills, concurrent staging
leases, CPU/GPU result merging with repeated pinned-buffer reuse, reserved-slot
relocation, and six physical VMM release/growth transitions. Two salted 170K
contexts alternating with short requests completed physical stages
`0 -> 1 -> 2 -> 0 -> 1 -> 2 -> 0`, preserved experts across eight resize events,
and returned `42` after the transitions. CUDA graph replay was observed.

From the repository root, with the existing local Docker image and model mounts:

```bash
root="$PWD/tools/qwen38_cache_tests/async_admission_20261021"
QWEN38_VERIFY_RESULTS="$root/performance-replay" bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_performance.sh --modes default
QWEN38_VERIFY_RESULTS="$root/lifecycle-replay" bash tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/run_default_lifecycle.sh
QWEN38_VERIFY_RESULTS="$root/nsight-replay" bash tools/qwen38_cache_tests/nsight_optimization_20261021/run_nsight.sh --modes default --annotations
```

The benchmark now accepts only `default` and `legacy_gpu`. Removed crossover
calibration modes must not silently benchmark the new default under old names.
