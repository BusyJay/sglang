# GLM MLX decode boundary acceptance — 2026-11-01

Adopt the validated production mechanisms by default: parallel Sinkhorn
coefficient normalization, fused RMS/mix projection, shared routed gate/up
inputs, parallel expert down projections, segmented latent cache reads,
shared snapshot storage and independent submission groups. Decode, prefill
and verification submission groups still default to four layers each.

The input base is `f71465f98d4c3a3d8dae33a55784ee1327927ea0`.
`decode_boundary_fix_portable.tar.gz` has SHA-256
`fe16e3681a7c04720c30fe07fc52841636cd79cbf0ab6b38a711911f9bcded91`.
All 148 manifest entries matched. Applying the cumulative patch to a clean
index reproduced its 68 supplied files exactly. The cumulative patch already
includes the incremental empty-prefix fix; do not apply that fix twice.

## Mechanisms and numerical behavior

- SIMD lanes normalize independent Sinkhorn coefficient lines while retaining
  the original summation order and rounding. The serial oracle remains in
  `sglang.test.kernels.mlx_hc`.
- RMS/mix fusion avoids the normalized intermediate and FP32 input copy,
  preserving the native singleton GEMV reduction. Additional device checks
  covered all advertised widths from 256 through 4096 in increments of 256,
  FP16/BF16 and rows 1/2/3/17: 128 bitwise-exact combinations.
- Joint routed gate/up projections reuse lane-local activation loads and
  affine sums. Compiled graphs receive checkpoint arrays explicitly, and
  changes to weights or activation limits remain observable.
- Expert down projections run independently before the final routing-order
  weighted reduction, preserving both local output casts.
- MLA caches retain a sealed prefix and bounded append block. Cache type and
  logical visibility survive verification, accepted consumption, rejected
  suffixes, chunked prefill, prefix reuse and request cleanup. Empty segment
  arguments alias an unreachable nonempty buffer while retaining the original
  split; zero-visible attention returns zero. No historical KV concatenation
  or serial replay is introduced for this boundary fix.
- Snapshot store/fork/restore uses distinct MLX array views sharing evaluated
  storage. Copy-on-write isolation and slot release remain covered. The same
  memory regression rejects the old approximately 202 MB copy allocation.
- Phase submission groups have separate configuration. Verification submits
  only its current cache group, including a partial trailing group.

Lengths, segment splits and index widths are runtime arguments. The changes
preserve checkpoint values, model equations and the mixed quantization recipe.
They do not add a diagnostic opt-out as a substitute for default adoption.

## Local validation and performance scope

Hardware: Apple M5 Ultra, 256 GiB. Python 3.12.16, MLX 0.32.4, mlx-lm 0.32.2,
Xcode 27.2. Checkpoint: `mlx-community/GLM-5.3-Flash-4bit`, revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`, 203,992,076,296 shard bytes.
Original mixed 4/5/6-bit affine expert weights and FP16 metadata are unchanged;
the large KDA projections, vocabulary head and MLA KV-B retain their original
BF16 values. One full model ran at a time with normal serving process defaults.
No libraries were installed, weights downloaded or OS memory limits changed.

The final six registered files pass 58 tests and 191 subtests using the normal
package imports, fixtures and Torch oracle. Launch these numerical tests with
`MLX_ENABLE_TF32=0`: setting it in the model fixture after another test already
initialized MLX is too late. The unchanged base reproduces that combined-test
initialization failure. Full-model and HTTP comparisons use the default TF32
policy on both sides. Fourteen additional local device cases pass.

Independent full-model comparisons at prefix lengths 1/255/511/2050 and
verification rows 3/5 preserve logits, hidden states, routes, logical cache
contents and offsets exactly. Two initial 64-token MTP generations preserve
tokens and proposal/acceptance/reuse counts. Repeated prefix511 and prefix2050
generations also preserve tokens and these counts. Observed full-model peak
allocation is approximately 204.9 GB on both sides.

Serial streaming HTTP uses English, Chinese and code prompts, 128 greedy
tokens each, two reversed prompt-order rounds, and a cache flush before each
timing request. Configuration is context4096, total tokens4352, prefill
chunk256, max-running2, seed1234 and deterministic inference. Process order
is baseline A, candidate B, candidate C, reverse baseline D.

| Decode configuration | Baseline A | Candidate B | Candidate C | Baseline D | Mean improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary decode, tok/s | 29.27 | 33.72 | 33.42 | 29.29 | 14.66% |
| Two-step MTP, tok/s | 50.05 | 52.75 | 53.24 | 49.05 | 6.96% |

These rates exclude TTFT and the initial output burst; they are neither
end-to-end nor saturated concurrent throughput. All 48 timing outputs and
112 service-contract outputs preserve token/logprob hashes. Exact/suffix
prefix reuse, chunked prefill, sampling, output constraints, budgets 1/2/3,
two requests and disconnect recovery pass. Two-step MTP retains 428 verifies,
850 proposed drafts and 609 reused tokens across all four process rounds.
The aggregate improvement belongs to the combined implementation; it must
not be attributed independently to every mechanism commit.

The prior-candidate synthetic segmented verification comparison still shows
approximately 1–4% extra short-path overhead. Long-prefix and full-checkpoint
verification improve, and HTTP does not regress. Further host/kernel trace
attribution of that local overhead remains open. KDA DRAM counters, general
quality datasets, saturation throughput and other hardware remain untested.
Namespace checks have 113 passes and one missing-Triton failure reproduced
on the unchanged base. Existing swap usage increased during standard-serving
startup cycles; raw resource observations are retained rather than claiming
that system swap remained unchanged.

## Reproduction and retained evidence

Use the existing environment, with `PY` pointing to its Python executable:

```sh
SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 PYTHONPATH=python "$PY" -m pytest -q \
  test/registered/kernels/ops/attention/test_mlx_ragged.py \
  test/registered/kernels/ops/kvcache/test_mlx_latent.py \
  test/registered/kernels/ops/layernorm/test_mlx_hc.py \
  test/registered/kernels/ops/moe/test_mlx_glm.py \
  test/registered/unit/hardware_backend/mlx/test_auxiliary_state_acquire.py \
  test/registered/unit/hardware_backend/mlx/test_glm5_next_mlx.py

SGLANG_USE_MLX=1 PYTHONPATH=python "$PY" \
  benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$MODEL" --output benchmark/mlx/http_recheck.json \
  --steps 0 2 --port 30131 --output-length 128 --repeats 2 --check-contracts
```

`MODEL` must identify the complete local checkpoint revision above. Run the
HTTP command on the base and candidate in A/B/B/A process order, using fresh
output paths and reversing the step order in the corresponding runs.

Raw acceptance files remain local under
`benchmark/mlx/decode_boundary_acceptance_20261101/`: `http_{baseline_a,
candidate_b,candidate_c,baseline_d}.json`, `http_comparison.json`,
`full_comparison.json`, `long_comparison.json`, `final_device/`,
`environment.json`, `final_source_manifest.json` and `ACCEPTANCE.json`.
Per-commit registered checks are recorded separately in
`benchmark/mlx/decode_boundary_commit_20261101/validation.json` and its stage
logs. These raw experiment files, detached validation checkouts, profiling
tools, profile data and large captures are intentionally not committed.
