# Online adaptive Qwen3.8 expert cache

The original implementation and measurements below describe the first version.
For the current lower-overhead implementation, SLRU and smaller-pool test options,
see [CACHE_POOL_V2.md](CACHE_POOL_V2.md).

`adaptive_cache: true` enables a live, fixed-budget allocator for the threaded
runtime. The initial `cache_slots_per_layer` or `layer_cache_slots` list sets the
total GPU slot budget. The runtime allocates that budget once as six contiguous
tensor pools and gives each layer a view into each pool. The shared full-expert
prefill arena remains separate. Each Qwen3.8 slot costs about 2.64 MiB; 48 layers
at 112 slots reserve about 13.84 GiB for decode experts.

At every one-token decode MoE call, the existing metadata stream copies a 513-int
route histogram into pinned host memory alongside its transfer plan. A separate
CPU thread feeds the active expert set into shadow copies of the production
decay-LFU policy at candidate capacities. It computes a miss curve for every
layer over a rolling observation window, then uses dynamic programming to find
the allocation with the fewest estimated misses under the original slot budget.
The default window is 512 decode calls per layer (`adaptive_cache_window`).

The controller requires both a 5% improvement and a projected eight-window
payback of the slots evicted by the proposed layout. It waits two windows after
a proposal. This limits oscillation when tool workloads change briefly. Capacity
choices range from roughly half the smallest initial layer through 16 slots above
the largest, in steps of sixteen, always including the initial capacities. The
minimum is never below the model's top-k.

When a proposal is accepted, the next eager MoE boundary drains pending transfer
jobs and synchronizes CUDA work. The runtime keeps each layer's hottest resident
experts up to its new capacity, moving their six tensors within the shared pool
with one scratch slot. It then restores lookup, frequency and age metadata for
those experts; only the residents displaced by a shrinking layer become cold
misses. Pool addresses and total bytes do not change. CUTLASS is outside the
segmented decode graph, so new views are provided at replay time. The swap can
cause a visible pause. It never changes the KV-cache reservation.

Current Qwen3.8 logging reports `adaptive_rebalances`, `adaptive_sample_drops`,
`adaptive_migrated_experts`, `adaptive_evicted_experts`, the last rebalance pause,
and cumulative per-layer hit/miss counts when this mode is enabled. A queue-full
sample is dropped, not allowed to delay the transport worker.

Example loader JSON fragment:

```json
{"cache_slots_per_layer":112,"runtime_mode":"threaded","adaptive_cache":true,"adaptive_cache_window":512}
```

The mode is opt-in. Synthetic GPU tests cover moving slots between layers with
an unchanged pool address, all six tensor values after refill, and a naturally
triggered hotspot migration.

## Real-model finding (5090, 2026-10-11)

With the same 5,376-slot budget, an 8,192-token context and a 600-token decode,
the allocator moved slots from relatively cold layers to hot layers. The
periodic cache report improved from 84.2% hits with uniform 112 slots to 85.6%
with the proposed layout. The measured decode rate, however, fell from roughly
21–27 tokens/s before migration to roughly 8 tokens/s afterward. A separate
server started directly with the proposed static layout ran at about 9 tokens/s
and reached 85.6% hits; the expensive path is associated with that capacity
layout, not the brief in-place migration. A static layout moving just one pair
of layers from 112/112 to 96/128 ran at about 22–23 tokens/s in a shorter
240-token decode.

Production profiling identified the slow path. FlashInfer shares one native
`FusedMoeRunner` across calls with different expert counts. Its
`MoeGemmRunner::getMaxWorkspaceSize()` caches only the *last* expert count, so
the heterogeneous layers repeatedly recompute the maximum workspace size.
That computation tries unsupported CUTLASS tactics by throwing/catching
exceptions and building stack traces. In a 20-second `py-spy --native` sample
of a live 480-token request, about 92% of the scheduler main thread's samples
included `calcMaxWorkspaceSize`; the cache analysis thread was nearly idle.
GPU SM utilization averaged 12.5%. The migration pause (about 0.12 seconds)
cannot explain the sustained slowdown. This diagnosis comes from the installed
FlashInfer source at `flashinfer/fused_moe/core.py` (shared runner dictionary)
and `nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h`
(single expert-count workspace cache).

The controlled live request completed 480 tokens in 93.95 seconds including
prefill; the periodic cache delta covering the first 398 decode tokens was
83.56% hits. The production workload's interval hit rate varies substantially,
so its cumulative 82% figure does not establish an improvement over the
uniform layout. This allocator can improve hit rate for a matching workload,
but the shared runner must cache workspace sizes by expert count before
heterogeneous layouts are practical.

The local image `swallow-sglang:v0.5.21-qwen38-capacity` applies that fix in
FlashInfer's Python runner cache: `num_experts` is included in the key, so each
capacity gets its own native runner and its one-entry workspace cache stays
warm. Build it with
`docker build -f tools/qwen38_cache_tests/Dockerfile.flashinfer_capacity -t swallow-sglang:v0.5.21-qwen38-capacity .`.
With the same static heterogeneous layout and 8,192/600 canary, hit rate stayed
at 85.6% while final decode throughput rose from 9.17 to 39.36 tokens/s.
The live allocator canary completed one in-place rebalance: hit rate was 85.4%
and final decode throughput was 26.45 tokens/s, compared with 85.6% and 7.98
tokens/s for the same allocator in the unpatched image. The difference between
the patched static and adaptive runs includes the per-token route sampling and
CPU shadow policies; it is a separate opportunity to reduce overhead. The
production Compose `llm` service is set to the patched local image, while
`vit` continues using the base image. Starting the service applies the fix.
The trace files and reports are under
`results/adaptive-cache-canary-v4-20261011/`,
`results/static-hetero-canary-20261011/`, and
`results/static-hetero-patched-20261011/` and
`results/adaptive-patched-20261011/`; the live profile is in
`/home/user/workspace/prod/cache/qwen38-adaptive-live-speedscope.json`.
