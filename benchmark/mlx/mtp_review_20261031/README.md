# GLM MLX MTP patch evaluation — 2026-10-31

This report records the local Apple M5 Ultra evaluation and final implementation.
[M4_REVIEW.md](M4_REVIEW.md) and [MECHANISM_REVIEW.md](MECHANISM_REVIEW.md)
are the supplied patch's historical M4 review, not measurements of this machine.
The `source_*.json` files preserve the original provenance and synthetic evidence.

## Decision and defaults

Keep the patch, repair the measured regressions and enable both optimizations:

- `SGLANG_MLX_GLM_MTP_VERIFY_MODE=ffn` batches stateless verification FFNs,
  router, mHC nonlinear work and expert down/combine. `serial` is the previous
  reference; `head` isolates the head path for diagnosis.
- `SGLANG_MLX_GLM_CACHE_LATENT_PROJECTION=1` retains split MLA operands at the
  original matmul dtype-promotion boundary, for dense and quantized weights.
- The standalone `load_mtp_model` uses these same defaults. Native MTP serving
  continues to use `--mlx-mtp-num-steps`; fixed **1** is the best measured budget
  on the three 128-token request panels below. `0` disables drafts and `-1`
  enables the existing adaptive controller. This patch changes optimization
  defaults, not the CLI's explicit opt-in to loading the additional MTP head.

The persistent external-patch acceptance, investigation, default-enable and
commit policy lives in [AGENTS.md](../../../AGENTS.md).

## Reproducible environment

- Baseline: `a7a39c8b3d4a363f097b5971bb5a9a2ec7b377e5`, branch `metal`.
- Input patch SHA256:
  `76208fc5194685a8c6fc81165b4fc09ef5a67734c16c2f4e9950d154145528a2`.
  Base, manifest hashes and `git apply --check` were verified before application.
- Apple M5 Ultra, 256 GiB unified memory; MLX 0.32.4, mlx-lm 0.32.2.
  Existing `sglang-metal` Python 3.12 environment; no dependency installation.
- Unchanged `mlx-community/GLM-5.3-Flash-4bit` snapshot
  `fa04cbc3622b16aa7475404060b05e506d81bd4e`.
  Expert affine metadata is FP16 with mixed 4/5/6-bit weights; KDA large
  projections, dense KV-B and the vocabulary head remain BF16. This is not
  interchangeable with a recipe that quantizes those projections.
- One full model process at a time. Serving's process-local recommended working
  set (240 GiB) is used consistently; no OS wired-memory setting was changed.
  The unpinned preliminary run stalled on pageable GPU work and was interrupted;
  its partial output is excluded from performance conclusions.

## What regressed, and what was fixed

The original FFN candidate ran about 30.8–31.0 versus serial 33.1 tok/s in its
opposite-order model panel. It changed generated tokens. A forced two-row chain
had max logit error 2.5625 and cache error 1.0; at nine rows the errors grew to
3.1328 and 1.3025. This was an actual output/acceptance risk, not a Metal limit.
See [initial_ab_pinned.json](m5/initial_ab_pinned.json).

[Layer localization](m5/localize_initial.json) compared each FFN from the same
serial input, without propagating earlier errors. The first layer already had a
0.00390625 normalization difference. Batched FP32 HC mix and residual products
changed reduction/rounding at cancellation-sensitive boundaries. Native batched
quantized FFNs introduced another independent difference on identical inputs.

The repaired path retains the original HC mix/residual GEMV reductions and
batches the nonlinear HC work. A new `gemm.mlx_affine_rows` Metal kernel places
multiple token/expert rows in one dispatch while retaining native singleton QMV
unpack, accumulation, reduction and output-rounding order. Shared and routed
FFNs use it, then the multirow down/combine preserves both local affine dtype
rounding and residual dtype restoration. No model equation, checkpoint or
quantization recipe changed. `batch_fusion` is an explicit call parameter,
including shared experts; no temporary module flags are mutated.

The original cache only covered `QuantizedLinear`. All twelve KV-B projections
in this checkpoint are dense BF16, so it missed the actual repeated promotion
cost. The cache now covers dense projections too. It casts each split operand
separately, retaining the uncached matmul's layout; casting the interleaved
parent can itself change reduction behavior. Weight/metadata identity, affine
spec and activation dtype invalidate the cache. It is independent of request KV
state and is shared across normal decode, MTP, chunk and prefix reuse.

## Model A/B

[improved_ab.json](m5/improved_ab.json) loads one full checkpoint and alternates
candidate order across four warm repetitions per forced chain. Cache creation
is warmed outside timed verification. All tested logits, hidden states and all
per-position caches are **bitwise equal** to serial with the cache disabled,
for 2/3/5/9 rows. All eight 128-token generation panels produce identical tokens
and identical acceptance counts (61 correct drafts / 65 proposed).

| Forced rows | Serial, ms | Batched FFN, ms | Batched FFN + retained operands, ms |
| --- | ---: | ---: | ---: |
| 2 | 57.25 | 55.93 | 51.92 |
| 3 | 77.43 | 75.05 | 66.47 |
| 5 | 174.80 | 126.67 | 99.93 |
| 9 | 275.11 | 245.29 | 223.61 |

Opposite-order generation is 31.37–31.41 tok/s serial, 31.89–31.94 with the FFN
change alone and **34.15–34.22** with both (+8.9% over serial's mean).
These are model-level measurements, not HTTP throughput.

This cache can retain an additional **768 MiB** of FP32 operands for the twelve
local KV-B matrices, while retaining the original BF16 weights. That is an
allocation estimate from checkpoint geometry, not a measured RSS delta. It is
lazy and only retains the last activation dtype per layer. The complete mixed
candidate panel's observed peak MLX allocation is **195.03 GiB**; it includes
weights, activations and the allocator's measured peak, not just cache storage.

## HTTP and step selection

The server uses context length 4096, chunk size 256, maximum two running
requests, deterministic sampling seed 1234 and ordinary streaming. Every timing
request flushes request caches and emits 128 greedy tokens. English, Chinese
and code prompts are repeated twice in opposite prompt order. Throughput is
decode-only, excluding TTFT. Results describe these short serial requests;
long contexts, concurrent serving and quality benchmarks need separate panels.

[http_final.json](m5/http_final.json) records the optimized path:

| Draft step | Mean HTTP decode tok/s |
| --- | ---: |
| 0 | 27.23 |
| 1 | **33.14** |
| 2 | 31.83 |
| 3 | 27.70 |
| 4 | 23.52 |
| 6 (default-only repeat) | 16.53 |
| 8 (default-only repeat) | 11.02 |
| adaptive (-1) | 31.86 |

The subsequent [serial reference](m5/http_serial_after.json) measured 27.21
(step zero) and 32.28 (step one) tok/s. Its timing hashes and all 14 contract
hashes, including logprobs, match the optimized panel across processes. The
first optimized step-one HTTP panel improves 2.7% over this reference; HTTP
and model-level percentages are intentionally reported separately. The final
[default-only run](m5/http_defaults_after.json) launches without either new
environment override and also sweeps six/eight drafts. It measures **33.87**
tok/s at step one (+4.9% over serial), confirming the improvement in the
opposite process order. The two optimized HTTP panels bracket the serial
reference rather than relying on a single favorable run.

All six configurations pass all **14 HTTP contract comparisons** each: greedy
outputs, seeded sampling, grammar/logit edits, token/top logprobs, streaming,
stop/budget behavior, chunk/prefix reuse and disconnect cleanup. All 36 timing
panels have identical output hashes to ordinary decode. The full-model A/B
separately establishes equality to the original serial arithmetic, rather than
comparing only different modes of the optimized path.

Increasing draft depth still costs more than it saves. Aggregate conditional
correctness in the fixed-three HTTP panel is 326/375 (86.9%), 224/320 (70.0%) and
113/217 (52.1%) at depths one through three. The forced-chain panel removes
acceptance as a variable and shows the separate verification cost. A reduction
in accepted drafts and additional row/draft work both matter; there is no
observed loss of acceptance caused by the repaired optimization. Dynamic mode
learns useful small budgets but pays exploration costs, including occasional
6/8-step trials. Fixed one wins this tested workload; it is not a universal
optimum for every prompt or context length.

## Remaining mechanisms and diagnostic controls

Attention still executes per row to form exact per-position KDA/MLA snapshots.
Batched attention projections with row-state outputs and cross-row expert
weight staging remain future implementation work, not platform limitations.
The affine row kernel covers FP16/BF16 activations, 4/5/6/8-bit affine metadata,
32/64/128 groups and its aligned tiles. Unsupported shapes use general MLX
operations and need their own numerical validation; this report's exactness
claim is limited to tested geometries, profiles and weights.

A compatible affine vocabulary head can use the row kernel. This checkpoint's
BF16 dense head retains the original GEMV calls; the original native batched
head changed logits by up to 0.03125. A dense head kernel retaining native
reduction is still an optimization opportunity. The `head` mode intentionally
has no dense-head speed claim. The candidate as a whole is retained because
its repaired FFN/cache mechanisms provide measured gains without the observed
output regression. The reference flags remain for diagnosis and retesting.

## Re-run

Use the existing environment and set `GLM_MODEL_DIR` to the existing snapshot:

```bash
python benchmark/mlx/bench_glm5_next_mtp_verify_ab.py \
  --model-dir "$GLM_MODEL_DIR" --rows 2 3 5 9 --repeats 4 \
  --output-length 128 --output benchmark/mlx/results/mtp_verify_ab.json

python benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$GLM_MODEL_DIR" --steps 0 1 2 3 4 -1 --repeats 2 \
  --output-length 128 --check-contracts \
  --output benchmark/mlx/results/mtp_http_default.json

SGLANG_MLX_GLM_MTP_VERIFY_MODE=serial \
SGLANG_MLX_GLM_CACHE_LATENT_PROJECTION=0 \
python benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$GLM_MODEL_DIR" --steps 0 1 --repeats 2 \
  --output-length 128 --check-contracts \
  --output benchmark/mlx/results/mtp_http_reference.json
```

Compare timing and contract `sha256` fields across files, including logprobs;
compare throughput per prompt as well as panel means. Do not count adaptive
exploration as steady-state fixed-step performance.

## Validation

The production model/kernel suite, including the PyTorch oracle, passes
38 tests and 130 subtests, including affine 32/64/128 groups and
512/1536/4096-column reductions. The kernel namespace suite passes 105 tests with two AMX import probes
deselected. The unchanged `srt/utils/common.py` AMX probe fails on this Mac
because its CPU path imports unavailable Triton; this same baseline failure
was recorded by the prior fusion commit. No dependencies were installed to
hide it. [http_comparison.json](m5/http_comparison.json) verifies **66 timing
outputs and 154 contract outputs** across all eleven independent server
configurations, including token and logprob hashes. Device test logs and
[source/evidence fingerprints](m5/validation.json) are in `m5/`.

The supplied M4 bootstrap-bypass harness and old bundle builder remain in the
unchanged original tarball. They are replaced in the final checkout by normal
registered tests, full-model A/B and HTTP comparisons; acceptance does not use
extracted test bodies. The original tarball remains unchanged.

For a focused rerun of the ordinary registered device suite:

```bash
python -m pytest -q \
  test/registered/unit/hardware_backend/mlx/test_glm5_next_mlx.py \
  test/registered/kernels/ops/gemm/test_mlx_affine_rows.py \
  test/registered/kernels/ops/layernorm/test_mlx_hc.py \
  test/registered/kernels/ops/moe/test_mlx_glm.py
python -m pytest -q test/registered/unit/kernels/test_kernels_namespace.py \
  -k 'not amx_backend_probe_is_lazy'
```
