# Single-switch screening and selected defaults (2026-10-24)

The repeated factorial matrix was stopped at the user's request. The decision
uses one baseline launch and one launch per individual switch; already completed
combination runs remain local evidence. No further combination runs are needed
after rejecting the switches with clear negative results.

## Results

All four launches use the same production-relevant settings: RTX 5090 D,
HiCache enabled, FP8 target KV, BF16 draft KV, packed native expert transport,
three-token MTP, concurrency one, five Mamba slots, memory fraction 0.934 and
expert budget 18998075296 bytes. The startup selection fix is `e96bc2d50a`.
Commands were checked identical; each switch is explicitly set for screening.

Decode figures are tokens/second. "Short" is the median of three distinct
1088-input/512-output requests. The last three columns are fresh short requests
after the specified long document, not decode over an active long context.

| Switch | Short | Natural chat | After 32K | After 170K | After 219K | Composite change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All off | 107.77 | 122.58 | 111.82 | 96.56 | 114.39 | — |
| HC only | 111.87 | 118.32 | 105.03 | 94.75 | 122.76 | -0.18% |
| CTA only | 109.06 | 115.87 | 105.59 | 97.20 | 99.88 | -4.51% |
| MXFP8 only | 108.27 | 115.42 | 107.79 | 91.44 | 107.28 | -4.13% |

The composite is the equal-weight geometric mean of each of the five workload
throughput ratios relative to baseline. It is a descriptive summary of these
workloads, not a statistical significance claim. Each switch has only one
independent launch. Generated text, routing and MTP acceptance vary; the raw
results retain complete verify counts and acceptance metadata.

## Default decision

- **HC defaults on.** Its isolated graph benchmark reduces BF16 latency by
  about 25%. Serving performance is mixed and approximately flat overall:
  short-request median is +3.8%, natural chat -3.5%, after-32K short -6.1%,
  after-170K short -1.9% and after-219K short +7.3%. This follows the user's
  preference to enable a demonstrated kernel improvement when serving is flat
  overall; it does not claim every workload improves.
- **CTA and MXFP8 remain off.** Both show negative results across multiple
  serving workloads, with a composite reduction of approximately 4–5%. They
  remain available for explicit experimentation; their code is not removed.
- The earlier, already completed all-on launch has a -6.74% composite change;
  it supplies no reason to promote the full combination. No new all-on launch
  was performed after the user simplified the plan.

Set `SGLANG_OPT_HC_MIX_SPLIT_K=0` before startup to restore the persistent HC
kernel. CTA and MXFP8 retain their existing opt-in switches.

## Validation and limits

- The four screening launches pass **84/84 core answer checks**, including
  arithmetic/copying/JSON and 32K/170K/219K head/middle/tail retrieval.
- Four additional continuation answers are correct, but **0/4 host-hit coverage
  probes** observe a host hit. Their stricter original assertions remain failed
  in raw records; they are not counted as successful end-to-end cache reloads.
- Mixed-precision host layout is separately verified by **74 unit tests** and
  **7 real QSA HiCache GPU tests**, including byte-exact target/draft/indexer
  GPU-to-CPU-to-GPU restoration. See the incident report.
- One short launch with the three environment overrides absent confirms the
  actual defaults are **HC=true, CTA=false, MXFP8=false**. Startup, three short
  requests, natural chat and three exact-answer checks pass. This is a
  functional smoke check, not another performance comparison; its natural-chat
  prompt uses the older benchmark's thinking setting.

`single-switch-summary.json` contains the numbers and evidence paths. Raw
records remain under
`../csf_defaults_20261023/all_optimizations_matrix_hicache_20261024/r0-{000,100,010,001}/`.
The short default smoke is under
`../csf_defaults_20261023/hicache_startup_20261024/default-hc-smoke/`.
The retired large-matrix harness is archived locally with its raw evidence;
`run_serving_ab.py` is the retained simple single-case entrypoint.

The startup failure, replay and fix are described in
[`../hicache_startup_20261024/REPORT.md`](../hicache_startup_20261024/REPORT.md).
No production service was deployed by these checks. The test GPU is released.
