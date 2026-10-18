# B12X launch, decode fill and prefill transport experiments

GPU and service experiments completed on the actual Swallow image:
see [REPORT.md](REPORT.md) and `results/` for the selection, 15 fresh-worker
service runs, numerical checks, memory warnings and the final profile.
Selected deployment: prepared launch **off**, fill CTA **8**, prefill **auto**.
The remaining sections preserve the package author's original experiment plan.

Base: `bed9c124a2e7b627bbdf84ae6e92265e79160f76`. Image:
`lmsysorg/sglang:nightly-dev-20261018-b234b5c3`, tested local image ID
`sha256:bfef46e5a45a815f8c2f88a722980baf47b71a8689b89c6cc738f596ceeaf47c`.
No GPU was available for this package's validation. Keep changes experimental.

## Runtime change and rollback

`SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=1` opts into prepared pointer bindings
for the existing B12X W4A16 1..3-token path. It caches static native weight
pointers and compiled specializations, rebinds dynamic inputs each call, and
removes repeated general-wrapper/descriptor construction. Cache replacement,
capacity, M, top-k and ID dtype cause rebinding; bindings are capped at eight
per adapter and do not retain expert pool tensors. Global scales are still
refreshed after fills. The upstream wrapped compiled program and external
scratch completion event remain in use for graph lifetime and stream ordering.
This adapter still requires FlashInfer 0.7.2 and SM12x.

Keep `SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x` as in the validated deployment.
The new flag defaults to 0. Change it before worker initialization, then restart.
To roll back this experiment, set it to 0 and restart. Fill/prefill algorithms
are unchanged by applying the patch; their experiments are standalone scripts.

## Run in the existing image

Mount this source checkout as `/workspace/sglang`, work in that directory, and
set `PYTHONPATH=/workspace/sglang/python:/workspace/sglang`. Use your existing
runtime overlays/dependency setup for the pinned B12X image. No image export
or rebuild is needed. The following commands run inside that environment.
Run standalone GPU microbenchmarks with the service stopped so memory and
workload contention do not distort results. They do not load model weights.

```bash
python test/manual/kernels/moe/qwen38_b12x_launch_ab.py \
  --rounds 3 --iterations 100 --output-dir /results/launch
python test/manual/kernels/moe/qwen38_fill_ab.py \
  --rounds 3 --iterations 50 --output /results/fill.json
python test/manual/kernels/moe/qwen38_fill_ab.py \
  --rounds 3 --iterations 50 --overlap-rows 16 128 1024 \
  --bulk-layer --prefill-sweep --output /results/fill-prefill-overlap.json
```

Launch A/B alternates execution order across rounds and runs existing dense
reference parity, graph replay with changed inputs/scales, alternating streams
and cache replacement tests. Event-based eager timings include host issue gaps;
the graph result is a separate comparison. This is not an end-to-end speedup.

Decode fill compares mapped CTA 1..8/adaptive against registered-host DMA,
DMA with coalesced source AND destination runs, and DMA with a GPU plan readback.
It checks exact bytes using distinguishable expert/segment/offset patterns,
including untouched slots. Synthetic plans cover zero/1..30 misses, both
contiguous and fragmented IDs/slots. Supply `--plans real-plans.json` to replay
a JSON list of plans, each a list of `[expert_id, cache_slot]` pairs (<=30 misses,
slots 0..31). This script does not capture production plans itself. The default
native payload is four weight/scale segments, 2,764,800 bytes per expert;
per-layer global alphas are already resident in the production mailbox path.
Source allocation/registration/JIT and warmup are outside timed regions.

Known-plan DMA omits route readback and CPU descriptor generation. The readback
variant includes those costs but omits production worker handoff. Fixed CTA
grids reserve 30 miss blocks, matching a three-token top-10 verify; one-token
production uses fewer blocks. Interpret one-token synthetic timings accordingly.
Randomized variant order and three rounds help expose drift. GPU event spans
include host submission gaps; wall time also includes per-iteration completion
synchronization. Both matter for a launch/transport experiment.

`--overlap-rows` compares isolated BF16 GEMM, serial fill+GEMM and parallel
copy-stream fill+GEMM with explicit wait/join events. This measures resource
competition and synchronization overhead; it does not reproduce the model's
routing dependencies or implement a pipeline. Compare serial/concurrent wall
and GPU spans, especially misses 2/3/10 and fragmented plans. A mapped fill
uses SMs whereas DMA can use copy engines, but either can contend for memory
and PCIe bandwidth. Determine benefit from results rather than engine labels.

`--prefill-sweep` compares row-wise sparse DMA, coalesced sparse DMA and dense
full-layer DMA at 64/128/256/384/448/512 missing experts, with contiguous and
fragmented runs. It allocates a full 1.32 GiB GPU target and about several GiB
of host validation buffers. It uses registered host data/precomputed plans;
it does not model the production staging ring, mmap faults or cache scatter.
Use it to identify candidate boundaries, then confirm against service data.
A final partial chunk can behave differently from a full prefill chunk.

## Full-model baseline and confirmation

On an isolated existing server, use real token IDs (JSON array at least as long
as the largest requested prompt). Prewarm the worker, then collect:

```bash
python test/manual/kernels/moe/qwen38_prefill_baseline.py \
  --url http://127.0.0.1:8329 --input-ids /data/real-input-ids.json \
  --lengths 4096 8192 16384 32768 46000 --rounds 3 --flush-cache \
  --output /results/prefill-baseline.json
```

Prefix-cache flushing is explicit. The script rejects reused prefix tokens or
incorrect token counts. It flushes prefix cache, not expert-cache residency;
record expert cache stats separately. Returned one-token wall time includes
queueing, prefill and first decode, so pair it with a stage-specific profile.
The first round can still include warming effects: inspect rounds separately.

For prefill transport A/B, existing `model_loader_extra_config` already supports
`prefill_copy_mode=auto/dense/sparse` in threaded mode. Restart a fresh worker
for each candidate, retaining all other flags, chunk size, cache/KV budget,
input IDs, and expert-cache warmup policy. Record `prefill_copy_stats` (dense,
sparse, host_missing, missing_runs), H2D bytes, host dispatch time and GPU trace.
Do not select a threshold from synthetic results alone. Existing auto logic
already chooses dense replacement for heavily fragmented missing experts and
supports contiguous runs and cache reuse; this patch does not duplicate it.

For launch and decode fill confirmation, use the existing service benchmark:

```bash
python tools/qwen38_cache_tests/bench_mtp_service.py \
  --url http://127.0.0.1:8329 --stream --case 0 --tokens 192 \
  --output /results/warmup.json
python tools/qwen38_cache_tests/bench_mtp_service.py \
  --url http://127.0.0.1:8329 --stream --tokens 1024 \
  --output /results/macro.json
```

Restart workers for new prepared-launch/CTA settings. Compare baseline/prepared
in alternating order, with at least three rounds. The existing fill setting is
`SGLANG_QWEN38_FILL_CTA=1..8` or `adaptive` (default 8). Select promising CTA
values from microbench rather than running every combination on the service.
Preserve prompt/context, MTP steps, accepted length, KV budget and cache
capacity. Compare sustained decode tok/s, per-verify fill/wait times, miss counts
and H2D bytes, not only peak tok/s. Measure a short prompt and the actual long
context workload; their acceptance and attention costs differ. Use the existing
profiler entry points separately from macro timing to avoid profiler overhead.
Do not enable this experimental launch flag by default before GPU parity and
service A/B pass. If DMA only wins without plan readback, a future implementation
needs asynchronous plan publication/worker scheduling before it is worthwhile.

## Strata's relevant implementation

Inspected upstream commit `257727d58ff85a9aef6f314f9a95f75e4fc330d2`:
[Strata prefill.cpp](https://github.com/sergqwer/strata-nvfp4/blob/257727d58ff85a9aef6f314f9a95f75e4fc330d2/src/prefill/prefill.cpp).

- Long chunks build a fixed sequence of nonresident experts across layers
  (around lines 2520..2590), issue asynchronous DMA into a ring on `m.copy`,
  wait for slot-use events before overwrite, and publish copy completion.
  A host issuer thread separates DMA submission from kernel submission.
- Short chunks use actual routed experts and keep STAGE actual transfers ahead,
  skipping resident experts (around lines 4170..4200). Fixed-order whole-layer
  traffic is justified by amortization across many routed token rows.
- Group gather waits once for the last ordered copy and releases a group with
  one event (around lines 4003..4050); NVFP4 scales are gathered in that launch.
  Supported MMQ products can instead read pointers into cache/ring slots in
  place, releasing them after the down projection. This requires a kernel/layout
  interface change; it is not available just by adding a stream to our arena.
- Shared scratch and borrowing expert-cache slots permit larger prefill chunks.
  Larger chunks amortize experts copied once per chunk. Strata's published
  Q2_0/IQ3_S results are from other formats/hardware and do not predict our gain:
  [prefill experiments](https://github.com/sergqwer/strata-nvfp4/blob/257727d58ff85a9aef6f314f9a95f75e4fc330d2/bench/results/2026-09-28-prefill-speed/README.md).

Useful next step is a bounded ring with copy/compute overlap and grouped events,
if long-prefill profiles show enough compute to cover copying. Routing-free
prefetch copies every nonresident expert, so evaluate extra traffic and reduced
cache/KV capacity as well as overlap. It need not allocate three complete layers.
No dependency-correct full-model ring pipeline is implemented in this package.

## Current profile and pipeline budget

Latest committed 16-verify profile: GPU envelope 22.873 ms/verify; fill 11.065 ms
(48.4%), of which 9.783 ms is not overlapped with the classified compute groups;
B12X kernels 2.928 ms. CPU B12X wrapper duration 6.039 ms/verify is inclusive and
cannot be counted as an additive saving. No PCIe utilization counters were
recorded, so bandwidth saturation is unproven.

Native H=2560, I=640, E=512 gives 2,764,800 bytes/expert, 1.318 GiB/layer and
63.28 GiB per 48-layer round if all experts are copied. Adjacent cache snapshots
estimate about 313 MiB of miss payload/round, so that would amplify decode
traffic roughly 207x. At the observed 26.263 ms round interval full-layer loading
would require about 2.59 TB/s from host. Two layers of classified non-fill work
provide only about 0.492 ms. These are geometry/profile estimates, not measured
host link throughput; the optional bulk benchmark reports actual isolated DMA.
Three full-layer buffers need about 3.96 GiB before activation workspace.

For long prefill, all-expert traffic is amortized across the chunk and can be
reasonable. Test bulk bandwidth, sparse/dense crossover and compute overlap
first, then decide ring depth using long-prefill stage times and memory budget.
For decode/verify, prioritize per-miss efficiency, cache placement and small
bounded prefetch rather than a full 48-layer stream of all experts.

## CPU-only validation

Existing container CPU suite: 334 tests; prepared-binding regression: 2 tests.
Offline SM120a compile plus real vendor marshaling: M=1/2/3 x int32/int64 IDs,
six specializations. Weight storage and hardware facts are mocked in that
compile probe; numerical, graph and service validation require the GPU scripts.
Manual scripts pass syntax/help checks. Logs are included in the package.
