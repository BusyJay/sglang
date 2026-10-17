# A16 + MTP OOM diagnosis, 2026-10-17

## Established facts

The `CUDACachingAllocator.cpp:508 expandable_segments: memory mapping failed
with OOM` messages are device allocator mapping failures, not evidence of CPU
mmap pages migrating into VRAM. Messages request 20 MiB when only 4–19 MiB is
free. Some requests finish despite these messages; allocator retries/reclaims
can succeed. Final failure is a separate 96 MiB tensor allocation.

Expert source uses CPU mmap + cudaHostRegisterMapped + cudaHostGetDevicePointer.
The mailbox fill kernel reads host memory and writes existing GPU slots.
Adaptive expert pool and prefill arena are allocated in reserve(), before KV
budget measurement. This path does not use cudaMallocManaged. Host registration
can have driver overhead; this evidence does not attribute every driver byte.

Same serving capacities throughout: A16 decode/verify + MTP2, .90 static fraction,
8192 prefill chunk, 220000 context, 5340 expert slots. Only allocation history and
an OOM snapshot callback were enabled. No smaller chunk/context/cache workaround.

## Reproduction

1. `bench_mtp_service.py --stream --tokens 16 --context-repeat 60000 --case 0`:
   120015 input tokens, no cached prefix; completed in 23.85 s, despite repeated
   allocator mapping warnings. Result: synthetic-120k.json.
2. `--context-repeat 100000 --case 1`: 200017 input tokens (120000 cached prefix;
   inferred exactly from first new chunk 8192 + pending 71825 + cached 120000).
   OOM during continued prefill, at GDN chunk_fwd_o -> torch.zeros_like(v).
   Last completed chunk had 47249 pending tokens. This was not a fresh 200K
   prefill and is not a direct reproduction of the user's exact prompt.

OOM snapshot contains 302954 allocator events, including original resident
allocations (history started before model load). Requested output: 100663296
bytes = 96 MiB; driver free: 90701824 bytes = 86.5 MiB. Total tracked segments:
32656850944 bytes; active blocks: 32466449408 bytes (~30.24 GiB). Inactive reserved
memory is small relative to active tensors; this is not evidence of a large
reservoir of free cache that empty_cache could fix.

## Active allocations at failure

See snapshot-summary.json for full grouped stacks. Selected allocations:

| Allocation | MiB | Location |
| --- | ---: | --- |
| Adaptive expert GPU pool, 6 tensors | 14080.12 | qwen38_cache_threaded.reserve |
| Shared prefill expert arena, 6 tensors | 1350 | qwen38_cache_runtime._allocate |
| KV K, 13 buffers (12 target + 1 draft) | 1495 | memory_pool._create_buffers_normal |
| KV V, 13 buffers | 1495 | memory_pool._create_buffers_normal |
| GDN chunk state h | 192 | chunk_delta_h.py:396 |
| GDN transformed w and u | 96 each | wy_fast.py:131-132 |
| GDN v_new | 96 | chunk_delta_h.py:398 |
| GDN A | 48 | chunk_fwd.py:392 |
| GDN contiguous q/k/v copies | 32 + 32 + 96 | fla/utils.py input_guard |
| Normalized q/k | 32 each | fla/l2norm.py |
| Current projected qkvz output | 256 | flashinfer mm_mxfp8 |
| Causal conv output | 160 | causal_conv1d_triton.py:454 |
| Hyperconnection outputs | 160 each, 2 live | hc_combine.py:86 |
| PLE addition + grouped norm outputs | 160 each | qwen4_exp.py:1661 / qwen38_fp32_norm.py:38 |

These allocations overlap in lifetime. In particular u is no longer read after
chunk_gated_delta_rule_fwd_h returns, but remains live while allocating o; w is
also retained even when SUPPRESS_LEVEL<3 returns None for w. Their combined
192 MiB is a concrete avoidable overlap to investigate, not a validated fix.
No numerical/kernel change or lifetime fix was deployed in this diagnostic.

## Why .90 does not establish safety versus .912

mem_fraction_static controls KV sizing, not a hard cap on all GPU allocations.
_profile_available_bytes subtracts pre-load free memory * (1-fraction), while
later graphs/workspaces and forward activations still consume physical memory.
With 30.66 GiB pre-load free, changing .912 to .90 adds only ~377 MiB to the
nominal reserve if all other settings are identical.

The located historical .912 startup at 00:47:11 was NVFP4, no MTP, FP32 Mamba,
lazy PLE. Current .90 is FP4-Local, MTP2, BF16 Mamba, pinned PLE. They are not a
controlled comparison. Both logs report ~2.09 GiB free after graph setup despite
the fraction change. The old log also contains a 00:50:04 GDN 96 MiB OOM; this
does not establish the user's previously working configuration or dismiss a
regression. Need the exact old matched configuration to quantify regression.

## Raw evidence / caveats

- service-initial.log: original user's OOM cycle (no prompt content captured).
- service-reproduction.log: diagnostic startup, 120K success, cached-prefix failure.
- *.pickle.gz: complete PyTorch snapshot, load via pickle.load(gzip.open(...)).
- *.json next to snapshot: original callback metadata. Its legacy field
  observer_allocated_bytes is actually DEVICE TOTAL bytes; corrected in source
  callback to device_total_bytes. Preserve original raw JSON unchanged.
- diagnostic-launch.json: frozen diagnostic source, parent source, backup path.
- startup-config-comparison.json: parsed historical/current server_args subset.

Allocator history has overhead; throughput under this diagnostic is not a new
performance baseline. The snapshot attributes PyTorch blocks, not driver page
tables or every non-PyTorch allocation. This report establishes the failing
allocation and major overlapping tensors; it does not claim all workloads are
stable or that a memory optimization has been validated.

The one-shot impossible-allocation observer smoke test passed in the serving
image before live capture (observer-smoke-test.log). Source py_compile and
git diff --check passed. Diagnostic history was removed after capture, and
production source returned to the parent snapshot; capacities were unchanged.
