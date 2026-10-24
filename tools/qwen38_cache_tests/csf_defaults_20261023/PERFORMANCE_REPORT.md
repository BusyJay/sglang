# Performance after expert-cache expansion (2026-10-24)

The final source retains initial KV 80000, stages 160000/220032, prefill 8192,
full NEXTN MTP with three draft tokens, and original precisions. Mamba cache is
five for the expanded runs. Concurrency is one. Decode figures below are tokens
per second; TTFT figures are seconds. Short decode is the median of three
1088-input / 512-output requests. Natural chat is 91 input / 512 output.

| Configuration | Initial expert slots | Short decode | Natural chat decode | 170064-token TTFT | 219064-token TTFT | Short decode after KV resize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original native | 7006 | 107.68 | 102.89 | 29.957 | unmeasured | 93.60 |
| Expanded native, run 1 | 7489 | 109.63 | 101.16 | 31.253 | 38.726 | 97.21 |
| Expanded native, repeat | 7489 | 103.02 | 109.25 | 30.441 | 39.758 | 102.96 |
| Original CSF | 7296 | 96.65 | 92.20 | 46.881 | unmeasured | 86.28 |
| Expanded CSF with headroom | 7700 | 97.94 | 97.20 | 47.299 | 60.171 | 89.57 |
| Expanded CSF boundary candidate | 7801 | 99.08 | 98.20 | 47.243 | 60.197 | 92.39 |

## Interpretation

- CSF 7700 adds 404 slots at each KV stage. Short decode is approximately
  1.3% higher, natural chat 5.4% higher, and 170K TTFT 0.9% higher than the
  original CSF record. Minimum sampled free GPU memory is 257 MiB.
- CSF 7801 adds 505 initial slots. Short decode is approximately 2.5% higher,
  natural chat 6.5% higher, and 170K TTFT 0.8% higher. Minimum sampled free
  memory is only 3 MiB; this is a boundary experiment.
- Native 7489 adds 483 initial slots. The two repeated short medians span
  -4.3% to +1.8% relative to the original record, so there is no demonstrated
  stable short-decode improvement. Natural-chat throughput also varies; 170K
  TTFT is 1.6% to 4.3% higher than the original record.
- At the same expanded byte budget, native is faster than CSF in these runs:
  219K TTFT 38.73--39.76 s versus 60.20 s, with higher short decode throughput.
  CSF provides more resident slots at that budget. The headroom candidate has
  a smaller byte budget and must not be described as an equal-budget comparison.

The baseline source/settings and final source/settings differ in memory
optimizations and Mamba sizing. These are end-to-end before/after observations,
not an experiment isolating only the slot count. Speculative acceptance and
generated text vary between runs, including identical-source native repeats.
There are too few independent runs for statistical significance claims.

The final column needs special care: original runs execute that request after
170K and have a 160K KV pool, while expanded runs execute it after 219K and have
a 220032-token pool. All requests in that column are fresh 1088-input /
512-output requests; they are not decode over a 219K active context. The figures
show successful performance after pool growth, but their differences are not
matched-stage speedup measurements. The long requests generate only 8 and 32
tokens respectively; their decode speeds are not stable long-context throughput
benchmarks, so TTFT is reported instead.

## FP8 paged-prefill operator

With the production-selected CUDA 13.0 assembler, original full-history packing
plus attention measures 12.161 ms; the final paged path measures 12.206 ms,
about +0.37%, while saving approximately 428 MiB. This synthetic timing uses
8192 queries, 219007 history tokens, top-k 2051 and seven rotating CUDA-event
samples. Every output is checked bitwise before timing. The earlier CUDA 13.3
assembler result has a larger overhead and is retained separately; it is not
the final production-toolchain figure.

## Evidence

Raw per-request timing, returned complete verify counts and speculative
acceptance metadata are in each directory's `performance.json`:

- `capacity-native-baseline`
- `root-final-bucket-csf-full`
- `capacity-native-half-final`
- `capacity-native-half-repeat`
- `capacity-csf-half-headroom`
- `capacity-csf-half-final`

`capacity-summary.json` provides compact machine-readable results. Full commands,
environment and final source hashes are retained with each final serving run.
See [CAPACITY_REPORT.md](CAPACITY_REPORT.md) for capacity and complete-serving
validation and [QSA_FP8_REPORT.md](QSA_FP8_REPORT.md) for arithmetic exactness and
operator timing details. These measurements do not prove general model quality
or full-model token-by-token invariance. No configuration was deployed.
