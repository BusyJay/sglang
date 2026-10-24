# Qwen38 optimization bundle verification (2026-10-24)

The bundle `qwen38-all-optimizations-20261024.tar.gz` was applied to the existing
Qwen38 working tree. Existing changes were first preserved as commit
`5e8b2e3d13`; the original archive base is
`674461f5b6988882a3aeaa818f8513d266186bca`.

The archive patch SHA256 was verified against its manifest before application.
All 21 patch targets applied without conflicts. Nineteen files initially matched
the archive manifest exactly. The expert-cache CUDA source and CSF GPU test
additionally retain the existing local changes. Every pre-existing patch-target
file was checked byte for byte against the saved baseline commit. Subsequent
changes only normalize formatting/imports and mark the shebang benchmark script
executable, as required by pre-commit.

The three optimization switches retain their default `False` values:

- `SGLANG_OPT_HC_MIX_SPLIT_K`
- `SGLANG_OPT_QWEN38_FILL_CTA_TABLE` (requires `SGLANG_QWEN38_FILL_CTA=adaptive`)
- `SGLANG_OPT_QWEN38_MXFP8_SMALL_M`

## Correctness and repository checks

Tests ran against the current source mounted into the existing
`qwen38-csf-defaults-20261023` container, using GPU 0 (RTX 5090 D / SM120).
The separate vision service on GPU 1 was left running.

- Seven bundle-related test files: **72 passed, 142 subtests passed**.
  This covers HC arithmetic and graph replay, MXFP8 projection accuracy and row
  invariance, quantized/unquantized runtime dispatch and fallback, CTA byte
  preservation, CSF GPU regressions, role marking, and benchmark coverage.
- Final formatted source under Compute Sanitizer memcheck:
  **72 passed, 142 subtests passed; ERROR SUMMARY: 0 errors** with CUDA API
  reporting disabled. The initial run with API reporting enabled also passed
  every test and reported seven `cuGetProcAddress_v2` invalid-value errors from
  the CUDA Python driver bindings, with no device memory access errors. Both
  logs are retained; the zero-error result is specifically a memory-access
  check and must not be represented as a clean CUDA API check.
- Kernel namespace and fused-op/dispatch CPU regressions: **121 passed**.
- Python AST, `git diff --check`, registered-test validation and package-test
  placement checks passed. All applicable pre-commit hooks passed after the
  formatting and executable-mode fixes.

## Isolated operator measurements

GPU measurements ran serially after the initial numerical tests. The first HC
measurement overlapped the tests and was discarded and replaced. Timings below
come from the subsequent isolated run. Hardware/toolchain: RTX 5090 D, driver
615.78.08, PyTorch 2.14.1+cu130, FlashInfer 0.7.2.

HC uses the bundle's graph benchmark, five alternating measurements per
implementation. For BF16:

| Rows | Persistent median (us) | Split-K median (us) | Latency change |
| --- | ---: | ---: | ---: |
| 1 | 8.533 | 6.382 | -25.2% |
| 3 | 8.583 | 6.417 | -25.2% |
| 16 | 8.858 | 6.683 | -24.6% |

FP16 latency decreases by 24.4–28.2%. This benchmark repeatedly uses one weight
pair; these timings do not establish full-model throughput or cold-weight gains.

MXFP8 uses the supplied rotating-input benchmark to exceed L2 capacity. Its
comparison is to **pinned CUTLASS**, not the best autotuned CUTLASS configuration.
The runtime candidate uses the default `32x128` tile:

| Projection | M | Pinned CUTLASS (us) | Default candidate (us) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| shared_gate_up | 1 | 12.595 | 10.916 | 1.15x |
| shared_down | 1 | 7.660 | 3.932 | 1.95x |
| gdn_qkvz | 1 | 133.317 | 28.357 | 4.70x |
| gdn_ba | 1 | 6.820 | 10.261 | 0.66x |
| gdn_out | 1 | 50.008 | 23.280 | 2.15x |
| attention_qkv | 1 | 108.050 | 23.661 | 4.57x |
| attention_out | 1 | 50.064 | 23.264 | 2.15x |

M=3 follows a similar pattern. The small GDN BA projection regresses with the
default tile. `16x128` is faster than `32x128` on every measured projection,
but this integration preserves the supplied runtime selection rather than
introducing additional tuning changes.

CTA verification completed **741 measurements**, with exact bytes checked for
each case. Measurements use 20 timed iterations, three randomized rounds and
both contiguous and fragmented synthetic plans. Medians pooled across plan
layouts and rounds:

| Misses | Existing adaptive (ms) | Candidate table (ms) | Latency change |
| --- | ---: | ---: | ---: |
| 1 | 0.08793 | 0.08990 | +2.2% |
| 3 | 0.20702 | 0.19114 | -7.7% |
| 4 | 0.26562 | 0.25149 | -5.3% |
| 20 | 1.37498 | 1.20655 | -12.2% |
| 30 | 2.09281 | 1.94086 | -7.3% |

Other measured counts (0, 2, 6, 8, 10) change little. These are isolated synthetic
transfer measurements and do not measure overlap with model compute.

## Complete serving comparison

The reproducible runner is `run_serving_ab.py`. It uses the established native
cache command, identical loader settings, random seed 42, concurrency 1, initial
KV 80000 with stages 160000/220000, prefill 8192, FP8 KV, five Mamba slots and
full NEXTN MTP with three draft tokens. The paired workload includes warmup,
three 1088-input/512-output requests, natural chat, a 170064-token prompt and a
fresh short request after KV growth. Addition, counting and exact JSON tasks
exercise the complete target/draft/acceptance path in every run.

All four runs completed, including **12/12 exact-answer checks**. Commands were
checked equal across all four runs; the three explicit switches were the only
variant settings. All runs allocated the same expert-slot budgets:
7201 / 6837 / 6583 for KV stages 80000 / 160000 / 220032.

| Switches | Short decode median (tok/s) | Natural chat (tok/s) | 170K TTFT (s) | Post-resize short (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| All off, baseline | 111.17 | 105.31 | 30.493 | 98.28 |
| HC + CTA | 109.26 | 109.10 | 30.405 | 93.72 |
| HC + CTA + MXFP8 | 106.65 | 105.93 | 30.487 | 91.43 |
| All off, repeat | 107.93 | 108.73 | 30.605 | 95.65 |

**No stable end-to-end speedup was demonstrated.** HC+CTA short throughput falls
inside the baseline-repeat range. All switches enabled are 1.2–4.1% slower on
short requests and 4.4–7.0% slower after KV growth than the two baseline runs.
Natural-chat throughput and long-prompt TTFT are close. Keep the candidates
opt-in; these measurements do not support enabling them together by default.

The post-resize case is a fresh 1088-input/512-output request after the 170K
request, not decode over an active 170K context. The long case generates only
eight tokens, so only TTFT is used for that comparison. The two baseline runs
already show variation; three short prompts per run and one candidate run do
not establish statistical significance. Generated text and MTP acceptance can
change with floating-point reductions. The numerical checks and three simple
answer tasks do not establish general model quality or tokenwise equivalence.
The isolated pinned-CUTLASS measurements also do not replace a comparison to
the serving baseline's full execution and autotuning behavior.

Serving results, including complete verify counts and speculative acceptance,
are recorded in `summary.json`; detailed commands, environments,
generated responses and server logs remain under
`../csf_defaults_20261023/all_optimizations_20261024/`.

## Evidence and reproduction

- `package-manifest.json`: original bundle manifest and patch hash.
- `source-manifest.json`: hashes of the final merged production and test files.
- `serving-command.json`, `serving-environment.json`: retained serving settings.
- `hc-benchmark.jsonl`, `mxfp8-benchmark.txt`: isolated operator results.
- `fill-benchmark.json`, `fill-benchmark.log`: all transfer measurements (local).
- `memcheck.log`, `memcheck-memory.log`, `namespace-tests.log`: local test logs.
- `pre-commit-host.log`: successful hook results (local).
- `summarize.py`: creates `summary.json` from all completed measurements.

The serving runner is invoked inside the test container with Python, e.g.:

```bash
docker exec -w /workspace/sglang qwen38-csf-defaults-20261023 \
  /opt/sglang/bin/python -u \
  /workspace/sglang/tools/qwen38_cache_tests/all_optimizations_20261024/run_serving_ab.py baseline
```

The other variants are `hc_cta`, `all`, and `baseline_repeat`. Each invocation
terminates its test server before returning. No production configuration is
deployed or changed by these measurements.
