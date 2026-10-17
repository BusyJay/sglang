# NVFP4 cache optimization settings

The expert source defaults to an anonymous mapping with transparent huge pages
(`SGLANG_QWEN38_SOURCE_MODE=anonymous_thp`), prepared asynchronously during model
loading. Registered-host prefill DMA and parallel LFU miss planning are enabled
by default. Lazy PLE uses two pinned staging buffers; the full PLE table remains
unloaded with the lazy backend. The optional BF16 decode GEMV path remains off
by default (`SGLANG_QWEN38_DECODE_GEMV=0`).

Set `SGLANG_QWEN38_FILL_CTA=adaptive` before starting the model process to select
fill parallelism from the GPU miss count. Values `1` through `8` select a fixed
CTA count; the default remains `8`. Restart the model process after changing this
environment variable. In Docker, pass the variable to the model process inside
the container.

| Miss count | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | >10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adaptive CTAs | 8 | 8 | 5 | 4 | 4 | 3 | 3 | 3 | 2 | 2 | 1 | 8 |

The mapping is compiled into `kMailboxCtasByMiss` in
`python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu`; there is no environment
variable for changing individual entries. Selection runs on the GPU during graph
replay, without copying the miss count to the CPU.

The CTA 1-8 scan covered miss counts 0-10, pinned and registered THP sources,
byte correctness, graph replay with changing plans, and full-model throughput at
512, 25K and 65K input lengths. Independent paired full-model validation did not
establish a consistent improvement over fixed 8, so the default remains fixed 8.
The scanned adaptive mapping used 2/2/4 CTAs for miss counts 5/6/7. Those entries
were subsequently changed to 3/3/3 by user choice; the new combination has not
been benchmarked as an adaptive policy. Earlier measurements are evidence for
the original scanned mapping, not for the revised combination.

Local raw results are retained under `optimization_20261016`,
`fill_cta_20261016`, and `fill_cta_full_20261016` alongside this document.
