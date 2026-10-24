# Lossless memory work and checked expert capacity (2026-10-24)

## Scope retained

All serving runs retain the FP4-Local QAD checkpoint, resident native NVFP4 MTP,
BF16 output head, target FP8 E4M3 KV, BF16 draft KV and BF16 Mamba state. Initial
KV stays 80000; the later physical stages are 160000 and 220032. Context remains
220000, global prefill chunks remain 8192, concurrency remains one, and full
NEXTN MTP uses three draft tokens with two steps. Verify calls are not split.
No production service or configuration was changed.

## Memory changes

- Single-request target/draft QSA TRTLLM scratch sharing removes two redundant
  128 MiB workspaces: 256 MiB. The multi-request / other speculative algorithm
  paths retain separate workspaces; shape counters remain separate. CPU
  isolation tests and GPU B3/B1/B3 graph replay comparisons passed.
- `max_mamba_cache_size=5` instead of 10 saves approximately 278 MiB of persistent
  SSM/conv storage. The 324 MiB verify intermediate and all state precisions
  stay unchanged. This is a serving command change, not a global default.
- GDN prefill reuses the otherwise idle MTP intermediate for h/w/output and
  reuses private Q/K/V storage where safe. The operator's additional allocation
  fell from 722,993,152 to 1,572,864 bytes, excluding already allocated inputs
  and the idle MTP buffer. This is not a whole-model peak reduction: strided
  QKV input guards still materialize about 160 MiB. Four tests cover full 8192
  geometry, rectangular K/V dimensions, alias guards, fallback and graph
  replay after scratch poisoning. Output, per-chunk and final SSM states match
  bitwise; memcheck and racecheck found no device errors/hazards.
- PLE releases dead projections before convolution, and masks its owned output
  in place, avoiding two full outputs (320 MiB at 8192 tokens).
- QSA prefix prefill reads selected historical KV directly through the page
  table, avoiding full-history BF16 K/V copies. At 219007 history tokens and
  8192 queries it saves 448,658,432 bytes, approximately 428 MiB. The 1ULP issue
  was traced to Triton's source-width-dependent MMA accumulation order and
  corrected with aligned 16-bit FP8 pair loads. See `QSA_FP8_REPORT.md` for the
  intermediate comparisons, compatibility, strict tests and timing cost.

These savings occur in different phases and must not be added as if they were
one simultaneous peak reduction. The lower transient peaks permit a larger
explicit expert budget and static fraction; raising static alone would leave
this loader's explicit expert budget unchanged.

Expandable segments also exposed a CSF header validation bug: a valid GPU
bank pointer minus its file offset can be negative. The C++ check now accepts
such deltas only when the resulting pointer is nonnegative. The GPU test uses
actual negative and positive deltas and covers fill/reader/relocation paths;
the targeted test and memcheck passed.

## Capacity and limits

| Format | Prior slots at 80K / 160K / 220032 | Checked candidate slots | Gain |
| --- | --- | --- | --- |
| native NVFP4 | 7006 / 6643 / 6388 | 7489 / 7125 / 6871 | +483 / +482 / +483 |
| CSF | 7296 / 6917 / 6652 | 7801 / 7422 / 7156 | +505 / +505 / +504 |

The matched candidate budget is 18,998,075,296 bytes versus 17,661,593,504
previously (+1,336,481,792 bytes, about 1.245 GiB); static is 0.934 versus 0.9.
Both candidates use `max_mamba_cache_size=5` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Commands, environment and
SHA256 hashes of relevant source are retained in each final run directory.
The logs confirm real transitions through both later KV stages.

7801 is the largest CSF candidate validated here, not an exhaustive maximum
or a production headroom recommendation. Higher CSF budgets did fail:

- `capacity-csf7900-fp8`: 7901 initial slots, OOM allocating 128 MiB QSA indexer
  logits during long prefill.
- `capacity-csf8000-fp8`: 8000 initial slots, OOM allocating a 32 MiB GDN input
  contiguous copy during long prefill.

Both allocations precede historical KV packing, so eliminating that packing
alone cannot make these failed configurations fit. No failed or incomplete run
is counted as a successful capacity result. The invalid native manifest trial
`capacity-native-matched` is likewise excluded; the native final run uses the
native manifest, not a CSF manifest relabeled as native.

Strata's documented ~8350 slots use different reserve/context and model format
choices, including Q2 draft experts and a Q8 head. Those two format differences
alone amount geometrically to about 472 native slots. They were not adopted;
see `STRATA_CAPACITY_REVIEW.md` for the source audit and remaining comparison
limitations.

## Validation interpretation

Each final serving replay runs eight complete workloads: warmup, three 512-token
short generations, natural chat, a 170064-token prompt, a 219064-token prompt,
and another 512-token generation after KV resizing. Three exact-answer checks
then exercise addition, counting and JSON formatting. Complete verify counts
come from returned `spec_verify_ct`, not from assuming that an interrupted
48-layer probe completed a round. The serving harness uses production execution,
without the earlier diagnostic reference hooks or illegal routing probes.

This combines bitwise operator tests with complete serving smoke regressions;
three exact-answer checks are not a general model quality evaluation. Performance
numbers are individual replay measurements with varying routes and speculative
acceptance, not a calibrated claim of a universal throughput improvement.


## Final-source full replays

| Run | Workloads | Complete verify calls | Exact QA | Short median tok/s | Natural tok/s | 170K TTFT | 219K TTFT | After resize tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `capacity-native-half-final` | 8 | 1211 | 3/3 | 109.63 | 101.16 | 31.253 s | 38.726 s | 97.21 |
| `capacity-native-half-repeat` | 8 | 1191 | 3/3 | 103.02 | 109.25 | 30.441 s | 39.758 s | 102.96 |
| `capacity-csf-half-final` | 8 | 1179 | 3/3 | 99.08 | 98.20 | 47.243 s | 60.197 s | 92.39 |
| `capacity-csf-half-headroom` | 8 | 1196 | 3/3 | 97.94 | 97.20 | 47.299 s | 60.171 s | 89.57 |

The previous correct software-pair implementation at exactly these budgets had
219K TTFT 41.397 s native (`capacity-native-matched-valid`) and 62.821 s CSF
(`capacity-csf7800-fp8`). The final compatible native conversion reduced those
single-run times by about 6.5% and 4.2%, respectively. This comparison does not
remove routing/acceptance variation. Against the original lower-capacity runs,
170K TTFT was 29.957 s native / 46.881 s CSF; final times are about 4.3% / 0.8%
higher. With the production CUDA 13.0 assembler, the synthetic QSA operator measures
12.206 ms versus the original packing + attention at 12.161 ms (about +0.37%).
The earlier bundled CUDA 13.3 timing was slower and is retained separately; see
the FP8 report for the actual-toolchain comparison. These optimizations primarily exchange less temporary memory
for more resident experts, rather than guaranteeing every operation is faster.

Both final replays were sampled from startup through completion every 100 ms.
Native peak used memory was 32164 MiB, minimum free 39 MiB. CSF peak used was
32200 MiB, minimum free **3 MiB**. GPU reported physical total is 32607 MiB with
about 405 MiB driver-reserved. These measured candidates have very little peak
headroom, especially CSF; they are not recommended production reserves. A
lower-budget CSF replay is recorded separately below.


## Identical-configuration repeat and quality boundary

`capacity-native-half-repeat` repeats the final native source and command
unchanged, verified by `source-manifest.json` and `command.json`. The only saved
environment difference is the diagnostic output pathname. Both complete replays
passed all eight workloads and all three exact QA checks. Their 170K and 219K
response texts match; six other generated responses differ. The short no-prefix
prefills bypass the modified paged QSA branch entirely. Comparison excerpts and
metadata are retained in `native-identical-config-comparison.json`; the earlier
software-pair versus final comparison is retained separately in
`native-greedy-cross-run-comparison.json`.

This establishes run-to-run variation even with identical final source and
commands. It does **not** establish the source of that variation, prove it
predates all memory changes, or prove equivalent general model quality. The
paged KV arithmetic claim rests on direct bitwise operator/intermediate tests,
not on asserting that entire serving replays are deterministic. General quality
and full-model token-by-token parity remain beyond these smoke checks.


## CSF candidate with measured headroom

`capacity-csf-half-headroom` passed the same eight workloads, **1196 complete
verify calls** and all three QA checks. Static is **0.926**, explicit expert
budget **18,732,653,696 bytes**; other precision, context and MTP settings are
unchanged. Actual capacities are **7700 / 7321 / 7056**, a gain of **404 slots
at every stage** over the prior CSF control. The minimum sampled free GPU
memory is **257 MiB**, with peak used 31946 MiB. Its 219K TTFT is 60.171 s.
This provides substantially more measured margin than the 7801-slot candidate;
7801 remains a capacity boundary experiment. Neither configuration was deployed.
The full command is `capacity-csf-half-headroom/command.json`; the reproducible
launcher is `run_capacity_case.py` with `capacity-csf7700-command.json`.

Across the two native runs and two CSF candidates, the final unchanged source
completed **32 workloads and 4777 timed full verify calls**, with all **12 QA
checks** passing. These counts exclude verify calls made by the separate QA
checks. Final source hashes still match all recorded serving manifests.
