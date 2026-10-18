# Qwen38 expert-cache maintenance

The target Qwen38 threaded runtime supports an optional loader configuration:

```json
{
  "adaptive_cache": true,
  "kv_cache_stages": [80000, 160000, 220000]
}
```

Merge these fields into the existing `--model-loader-extra-config` object. Keep
its pack path, runtime selection, and other deployment settings. The feature
requires target QSA, unquantized NHD KV storage, the standard paged allocator,
and chunked prefill. Post-capture KV, unified memory, and mixed chunks are rejected.
Draft KV and QSA index storage keep their normal backing.

Only the first 80K target KV tokens are physically backed at startup. Unused
backing funds additional directly executable expert slots; startup logs report
the actual slot counts after VMM alignment. At page size 64 the final capacity
rounds up to 220032. Stage planning uses physical occupied pages, including cached
prefixes and outstanding allocations, plus the upcoming prefill chunk or decode
candidate reserve. Thus a single 8192-token prefill chunk reaches the first
boundary at approximately 71808 occupied tokens. Concurrent requests and page
rounding can advance it earlier than an individual request's context length.

At the final target MoE layer, expert tail invalidation is queued on the compute
stream. Result processing uses its existing completion wait before unmapping
the corresponding expert extensions and backing the next KV prefix. Overlap
processing consumes this result before preparing the next batch; this can reduce
overlap at the two boundaries, but does not add a device synchronization. Only
the result belonging to that forward can publish the return. Stages can also move backwards to lend KV space to experts again. A reborrow
probe asynchronously reads a free-page bitmap before the current result is
processed. At that result's existing completion boundary, the controller checks
that every page above the smaller capacity is free. Cached prefixes and in-flight
allocations block shrinking; total context length alone is insufficient. Later
result frees are merged before removing the free tail from the allocator.
Reborrow requires demand plus two upcoming batch reserves to fit the smaller
stage, providing hysteresis relative to growth's single reserve.
The reserve retains the largest prefill/decode batch requirement seen so far.
A small decode must not shrink the backing below the space required by the next
full prefill chunk while a large cached prefix remains live.

The expert pool is contiguous, with stable reserved virtual addresses and
physical extension mappings for each stage. Adaptive slots may repartition all
layers within each stage's total budget, using the existing relocation algorithm.
On stage changes, layer offsets can change. Such layers are invalidated and
refilled on demand; unchanged prefix layers retain their policy and payload.
This avoids adding a synchronization for GPU relocation before unmapping a tail,
but stage changes can temporarily increase misses. The borrowed prefill arena
moves to the current expert pool tail; overwritten residents are invalidated
before that arena's old storage can be released or repurposed.
When the arena is enabled, its route-bound threshold is capped at the original
per-layer slot count. Lending KV to experts or adaptive growth must not switch a
short prefill from the batched arena transport to the slower per-expert path.
New resident slots still participate in the arena's cache reuse.
An oversized CPU HiCache prefix that cannot fit the current physical KV prefix
falls back to chunked recomputation.

Ordinary adaptive analysis, including with staged lending, remains in its background
process. A pending layout causes asynchronous policy D2H at the end of a target
forward; its existing result completion covers that snapshot. The scheduler
then queues resident relocation, policy restoration, and descriptor publication
on the same stream. Metadata and expert transfer streams wait on that stream.
The migration path no longer joins the job queue or synchronizes CUDA explicitly.
Borrowed prefill arena use postpones the snapshot until the arena is returned.
Analysis replies computed against a superseded layout are discarded.

`adaptive_last_rebalance_seconds` now measures CPU snapshot processing and CUDA
submission, **not** completed GPU migration time. The log labels it `enqueue`.
Relocation traffic, CPU sorting/submission, and first-use kernel compilation
still have costs. GPU measurements are required to quantify the latency gain.

CPU verification:

```bash
python3 -m unittest \
  tools.qwen38_cache_tests.test_cpu_kv_stages \
  tools.qwen38_cache_tests.test_cpu_adaptive_boundary \
  tools.qwen38_cache_tests.test_cpu_adaptive_cache \
  tools.qwen38_cache_tests.test_cpu_borrowed_arena \
  tools.qwen38_cache_tests.test_cpu_benchmark -q
```

The registered allocator test requires PyTorch. CUDA validation must additionally
check retained expert payloads and addresses across both stage returns, physical
memory accounting, repeated return/reborrow cycles, long prefills, cached prefixes, speculative verification,
and ordinary adaptive migrations under overlap. These GPU cases have not been
executed in the current CPU-only environment.
