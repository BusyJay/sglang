# Qwen3.8 patch comparison, 2026-10-13

This is a single-user RTX 5090 D field observation, not a controlled kernel
benchmark. The source is the append-only deployment log at
`/home/jay/workspace/prod/homelab-llm/home-jay/remote-mapping/llm/sglang-flash-next.log`.
The user replayed one conversation across the individual commits, then tried
a reasoning-text workload with L2 on, L2 off, and the earlier best commit.
The code versions below follow the user's test order and are corroborated by
the cache-stat fields in each server session; the log does not print Git SHA.

## Same-conversation replay

For this comparison, a sample is a `Decode batch` throughput line with
`#full token > 45000` and no `Prefill batch` line since the previous decode
line in that server session. This removes the most obvious prefill-mixed
intervals, but requests and cache state still differ. The number of usable
samples is small.

| Commit/configuration | Session start (CST) | Samples | Median decode tok/s |
| --- | --- | ---: | ---: |
| 02 `ed32f15fb7` | 01:27:27 | 10 | 30.29 |
| 03 `26691c852a` | 01:48:58 | 4 | 29.80 |
| 04 `384c83da44` | 01:57:46 | 5 | 32.12 |
| 05 `a1dea2df06`, packed H2D off | 02:04:16 | 5 | 31.63 |
| 06 `3f74a625a5`, L2 capacity zero | 02:11:11 | 4 | 30.88 |

The values support keeping 04 as the best candidate. They do not establish
a precise performance difference between 04 and 05 or 06, because of the
small samples and changing conversation/cache state. In particular, the 05
run had `packed_h2d=false`; these numbers do not measure its enabled path.

## Reasoning-text replay

The cleaner cross-configuration comparison uses the reported decode samples
with 27,000–29,000 full tokens. Each group has five samples and no prefill
line within that segment. All sessions reported `fused_decode_plan=true`,
`resident_alphas=true`, and `packed_h2d=false` where that field existed.

| Configuration | Session start (CST) | Median decode tok/s | Range |
| --- | --- | ---: | ---: |
| 06 with L2 enabled, capacity about 1.1K experts | 02:16:40 | 30.17 | 29.29–30.71 |
| 06 with L2 disabled, capacity 0 | 02:30:35 | 31.49 | 29.47–32.54 |
| 04 `384c83da44` | 02:36:21 | **32.98** | 30.65–35.34 |

At this context length, L2 off is faster than L2 on, and 04 is faster than
both 06 configurations. The L2-on session recorded hundreds of thousands of
L2 hits, so a functioning L2 cache did not translate into higher end-to-end
decode throughput here. This is a workload-specific result, not proof that L2
is always slower. The runs were sequential rather than randomized; thermal
state, cache warming, background load, and minor prompt differences were not
controlled. The SGLang `gen throughput` line is a periodic server estimate,
not a per-request wall-time measurement.

## Historical context for the original author

- The 2026-10-11 8K prompt / 256 completion canary in
  [CACHE_TUNING_112.md](CACHE_TUNING_112.md) found 112 slots/layer to be the
  highest configuration that started with `--mem-fraction-static 0.90`;
  128 slots/layer left too little KV headroom. Its maximum reported decode
  throughput was 16.80 tok/s. That short canary, with different slot count
  and prompt length, is **not** comparable to the 27–29K replay above.
- A prior 45–60K conversation comparison isolated patch 01
  (`a93ef9ca35`, decode batch staging) as a clear regression: median
  no-prefill decode was 29.4 tok/s before it, 20.7 with it, and 30.3 with
  patch 02 alone. Patch 01 reduced staging-batch count but still issued six
  H2D copies per miss; its added dependency/dispatch scheduling is a plausible
  cause, not a proven complete explanation. Patch 01 is absent from this
  branch and from the 02–06 series used above.
- The current deployment used 96 nominal cache slots/layer and adaptive
  rebalancing. A higher cache hit rate or a nonzero L2 hit count does not by
  itself imply higher decode throughput: promotion, demotion, D2D copies,
  bookkeeping, and transfer-stream dependencies also consume time. Compare
  actual decode rates in prefill-free windows at matched context lengths.

Decision: keep `384c83da44` as the latest performance code on
`qwen3.8-flash-next-offload`. Patches 05 and 06 remain available as separate
commits on `codex/5090-without-01` for further profiling; this report does
not claim either is universally detrimental.
