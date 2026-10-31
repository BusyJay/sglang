# GLM MLX delivery review — 2026-10-31

Review of `glm_mlx_delivery_20261031.tar.gz` against
`202d800b066829dc10ca5df40f83d40303c822ca`. This report distinguishes the
supplier's M4 operator evidence from local M5 full-checkpoint and HTTP evidence.

Adopt the verified batched mechanisms by default. Final streaming HTTP decode
improves from 45.23–45.59 to 49.40 tok/s at two draft steps; ordinary decode
is 29.29 versus 29.06–29.10. Fix M5 reduction parity, short-context dispatch
and growing-context compilation before adoption. Remove the regressing
singleton routed replacement while retaining multi-row fusion.

## Provenance and unchanged workload

`input_validation.json` records the archive and cumulative patch SHA-256,
base commit, fourteen verified base files, fourteen matching applied files,
47 supplier evidence hashes and 34 independent acceptance evidence hashes.
The supplied cumulative patch is preserved as `source_candidate.patch.gz`;
`source_manifest.json`, `source_README.md`, `source_REVIEW.md` and
`source_acceptance_REVIEW.md` describe the input, not the final implementation.
The original archive remains unchanged and is not included in this report.

Local hardware: Apple M5 Ultra, 256 GiB; MLX 0.32.4, mlx-lm 0.32.2,
Python 3.12.16, Xcode 27.2. `environment.json` records package versions,
checkpoint shard sizes/Hugging Face blob identities and resource observations.
The complete local `mlx-community/GLM-5.3-Flash-4bit` snapshot is
`fa04cbc3622b16aa7475404060b05e506d81bd4e`. The recipe is unchanged: mixed
4/5/6-bit routed affine weights with FP16 metadata; KDA large projections,
dense vocabulary head and MLA KV-B remain BF16. No requantization, altered
model equations, checkpoint replacement or relaxed correctness thresholds.

One full model runs at a time. Controls use the actual input commit in a
clean detached checkout, with an explicit `PYTHONPATH`. Processes use the
normal serving recommended working-set limit; no OS wired-memory settings
were changed, dependencies installed, or weights downloaded.

## Adopted mechanisms and corrections

- Fused affine gate/up/clamped SwiGLU for multi-row verification, plus mixed
  metadata projection helpers. Singleton routed serving keeps the existing
  compiled QMV dispatch: the supplied replacement repeatedly lost about 3%
  in real HTTP decode. Shared-input and rows-per-SIMD variants did not give a
  stable improvement (`m5/tune_routed.json`). This is a measured dispatch
  decision, not a Metal geometry restriction. The batched candidate remains
  enabled.
- Batched MLA projection and headed latent transforms, explicit legal index
  sets and indexed QK/softmax/PV. The original M4 reduction implementation
  failed ten M5 device subtests. Stage diagnostics isolated QK/PV arithmetic,
  while softmax remained exact. M5 MPP paths now preserve the installed MLX
  thin/regular/split reductions; the FP32 SIMD path also preserves the Ultra
  tile choice with TF32 disabled. `MLX_ENABLE_TF32` remains MLX's startup
  integer policy, default 1, not a per-request module mutation.
- Fused KDA recurrence emits every FP32 recurrent and convolution boundary;
  chains longer than eight use tiles with independent cache snapshots.
  Small and output projections share row tiles. Cache publication consumes
  only the accepted boundary, without replaying attention.
- Dense draft top-one epilogue uses the rounded vocabulary logits and native
  first-index tie behavior. Its initial compilation moves to model startup.
- Request build, execute and publication are separate; a shared verifier
  execution boundary avoids a host wait for each request. The sampled input
  token is passed explicitly by the runner. Zero-step decode stays lazy.
  This does not implement cross-request per-layer weight reuse or establish
  a concurrent throughput improvement.

The most consequential integration correction was **unbounded compilation**.
The input encoded index width and KV length as Metal template parameters.
Both grow during generation, causing a new build almost every token. Fixed
shape warm microbenchmarks hid this cost. They are now tiny runtime arguments.
The same first step-one HTTP request improved from 5.78 to 45.49 tok/s, with
maximum token gap falling from 360 to 44 ms. Row count remains a bounded
specialization; adaptive exploration can still encounter a new row-count
build. Root `AGENTS.md` now requires first-request and growing-context checks.

## Local evidence

Raw HTTP results are under `m5/http_*.json`; compressed server and driver logs preserve
startup and request timings. Each process uses context 4096, total token
budget 4352, prefill chunks 256, at most two requests, seed 1234 and
`--enable-deterministic-inference`. Three public English/Chinese/code prompts
produce 128 greedy tokens, twice in reversed prompt order, with cache flushed
before each timing. Decode throughput excludes TTFT and the first output
burst; it is not end-to-end throughput.

The fourteen service contracts per configuration cover chunked prefill,
exact/suffix prefix reuse, seeded sampling and logprobs, logit bias, regex,
stop, budgets one/two/three, two simultaneous requests and disconnect recovery.
They check output/cache semantics, not a general quality benchmark.

Final-source HTTP sweep (`m5/http_release.json`, process order 2/0/3/1/4/-1):

| Draft steps | Adopted HTTP decode tok/s |
|---:|---:|
| 0 | 29.29 |
| 1 | 44.25 |
| 2 | **49.40** |
| 3 | 48.49 |
| 4 | 38.37 |
| adaptive (-1) | 45.19 |

The final sweep passes all 36 timing outputs and 84 service contracts.
`m5/http_comparison.json` additionally verifies all 120 timing outputs and
280 contract outputs across twenty historical/final configurations. Raw
server, driver, diagnostic and test logs are preserved under `logs/*.log.gz`.

Initial independent control: ordinary decode 29.06 tok/s, step two 45.59.
Reverse control after the candidate sweep: 29.10 and 45.23. The full sweep
before restoring singleton QMV measured:

| Draft steps | HTTP decode tok/s |
|---:|---:|
| 0 (candidate singleton dispatch, subsequently removed) | 28.27 |
| 1 | 44.67 |
| 2 | 50.02 |
| 3 | 48.99 |
| 4 | 38.17 |
| adaptive (-1) | 44.35 |

Step two is best among these tested settings. This is workload-specific,
not a proof that two is optimal for every prompt or context. The final-source
sweep confirms this ranking. Adaptive mode
remains available but did not win this sweep; it includes exploration and
occasional new row-count compilation. Patch mechanisms are enabled by the
existing optimization profile/FFN verification defaults. The CLI still
requires explicit `--mlx-mtp-num-steps 2` to enable speculative decoding;
zero retains ordinary decode.

`m5/mechanism_ab.json` uses one loaded checkpoint and scoped reference methods
from the input commit. It keeps current serving common to every candidate,
so independent HTTP processes are the serving-level control. All 48 forced
panels have exact logits, hidden states and caches. Three opposite-order
short-context generation repeats give median 47.71 tok/s for reference,
48.74 head, 49.89 KDA, 48.43 MLA, 47.84 MoE and 52.82 combined. Tokens and
natural acceptance are unchanged. Same-process ordinary reference/MoE/
combined medians are 30.00/29.96/29.74; these did not reveal the larger
independent serving regression, which is why the latter was separately fixed.

The indexed attention implementation has a known long-context/small-row
cost: at prefix 2050, two-row forced combined verification initially takes
45.19 versus 42.16 ms, whereas nine rows take 110.33 versus 138.58 ms.
`tune_ragged.json` and `tune_ragged_inputs.json` preserve attempted K32,
slot prefetch, shared upcast and gathered-input alternatives. All tested
outputs are exact; none consistently improves the current indexed path.
These are random-activation operator timings, not full-model throughput.
Long-context generation and final-source checks are recorded separately below.

## Reproduction

Use an existing MLX environment and local checkpoint; do not install packages
or download weights just to run these scripts. From the repository root:

```sh
# Set PY to the persistent environment and MODEL to the unchanged snapshot.
PYTHONPATH=python "$PY" benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$MODEL" --steps 0 1 2 3 4 -1 --check-contracts \
  --output benchmark/mlx/delivery_review_20261031/m5/http_replay.json
PYTHONPATH=python "$PY" benchmark/mlx/delivery_review_20261031/full_model_probe.py \
  --model-dir "$MODEL" --steps 2 \
  --output benchmark/mlx/delivery_review_20261031/m5/probe_replay.json
```

For a baseline, create a clean worktree at the input commit and set
`PYTHONPATH` to its absolute `python` directory while running the same probe.
`compare_model.py` compares saved full-model arrays, routes, typed caches,
logits/KL and natural acceptance. Arrays are local diagnostic artifacts and
are excluded from portable evidence. `mechanism_ab.py` and `tune_routed.py`
accept `--baseline` pointing to that checkout. The compressed probe reports
can also be checked without local arrays using `compare_model.py --hash-only`;
non-equal hashes require regenerated NPZ files to measure errors. Operator tuners require no
checkpoint except `tune_routed.py`. Run device scripts sequentially.

## Final model checks and remaining scope

The final independently loaded full-model comparison is
`m5/final_model_comparison.json`: sixteen forced panels cover prefix lengths
1/7/26/2050 and row counts 2/3/5/9. Logits, hidden states, routed selections
and weights, every typed/shaped cache boundary and cache offsets are exact;
max absolute errors and KL are zero, and top-one selections match. This
extends the supplier's random/single-layer checks to the full checkpoint.

The additional one-token prefix exposed a real defect in the input's general
masked batch SDPA path: flattening multiple query batches prevents MLX from
collapsing the head dimension, changing its matrix dispatch. Full-model
logits differed by up to 0.78125 in the three-row case, with changed routing
and top-one selections (`m5/final_model_comparison_initial.json`). The fix
extends the indexed reduction path to legal lengths 1–7; no tolerance is
changed and no per-row SDPA is introduced. `diagnose_short.py` reproduces the
native masked-batch dispatch difference independently. A shifted low-precision
MPP tail also owns only its original output interval, avoiding overlapping
threadgroup writes. Device tests include short/empty rows, poisoned index
tails, mixed BF16/FP32 and all relevant reduction boundaries through 4096,
with both MLX TF32 startup policies.

Final step-two generation, 128 tokens with two repeats:

| Prompt tokens | Independent base tok/s | Adopted tok/s |
|---:|---:|---:|
| 26 | 47.55 / 47.69 | 52.44 / 53.15 |
| 2050 | 39.82 / 39.86 | 43.64 / 43.85 |

All generated tokens and acceptance counters match. At prefix 26 each run
has 51 verifications, 102 proposed drafts and 76 reused tokens (45/31 by
depth). At prefix 2050 each run proposes 101 and reuses 76. Peak MLX allocation
is 191.80 versus 191.30 GiB for these probes; this is not a general memory
saving guarantee or a measure of process RSS. Initial cold/JIT failures are
preserved separately, not mixed into these steady comparisons.

The final two-row prefix-2050 forced panel still takes 44.83 versus 43.37 ms;
three/five/nine rows take 49.17/68.90/110.18 versus 52.95/88.92/138.61 ms.
The recommended two-draft full-generation case improves at this context. Retain
the indexed mechanism for its overall verified gains; further optimization
of very small, long-context verification remains open. Do not infer that
all individual operators improve at every shape, or that unmeasured longer
contexts/other devices match these numbers. Cross-token expert compaction,
cross-request layer-level projection sharing, prefill pipelining and more
complete draft-head bandwidth reduction remain separate work.

The deeper-step cost is not evidence that this patch lowers natural acceptance:
independent model comparisons retain identical acceptance at the same budget.
The full HTTP reports include proposed/eligible/correct counters per depth
and budget. Added depths are reached less often and consume an additional
prediction/head pass. Five-row verification (four drafts plus the root) also
crosses the existing four-vector dense projection tile, causing another matrix
stream and dispatch. The forced-row timings demonstrate the cost increase;
these observations do not assign a precise fraction of the slowdown to each
cause. Adaptive exploration and bounded row-count compilation are separate
costs from the fixed two-step steady path.


## Validation and commit boundaries

The final workspace suite passes **171 tests and 276 subtests**. Two unchanged
AMX lazy-import probes that require unavailable Triton are excluded, as in the
input baseline; no dependency is installed to satisfy unrelated probes. Tests
cover GLM model/MTP/cache behavior, dense and affine GEMV, KDA snapshots,
indexed MLA, mHC, MoE, profiling hooks and lazy operator namespaces. The
expanded indexed attention case also passes all 60 subtests in an independent
`MLX_ENABLE_TF32=0` process. Format and the repository pre-commit Ruff selection
pass (`format_checks.json`); this is not a claim that the repository satisfies
all optional Ruff rules.

Each of the five cumulative mechanism boundaries is independently copied to
a clean checkout and runs the registered suite: 157/166/167/169/171 tests,
208/266/276/276/276 subtests respectively. `stage_validation.json` records their
source hashes and summaries. The fifth stage is byte-identical to the fifteen
final production/test files, and the normal working checkout is tested again.

`environment.json` records host swap: it was already 578 MiB before candidate
HTTP measurements and is 632.62 MiB after final HTTP/stage tests. These whole-host
snapshots increased by 54.62 MiB; no process-level attribution was captured.
Do not describe this experiment as zero-swap. No system memory limits are
modified; the complete model's measured MLX peak remains about 191 GiB.

Raw probe JSON is compressed losslessly, without NPZ arrays or weights.
`validation.json` fingerprints final source and portable evidence, separately
from the supplier manifest. The adopted patch is against the input commit;
its source files are the final validated versions, not the uncorrected input.

Mechanism commits (one mechanism per commit):

- `f6ba28fbfc` [Perf] Fuse GLM MLX affine verification activations
- `4cfc6ccbc6` [Perf] Batch GLM MLX latent attention with indexed cache reads
- `60e0b2cce6` [Perf] Emit GLM MLX KDA verification cache snapshots
- `712f1fd13a` [Perf] Fuse GLM MLX draft head top-one selection
- `e60b55226a` [Perf] Share GLM MLX MTP execution boundaries

`adopted_changes.patch.gz` contains only the sixteen final source/test/policy
files against the input commit; `adopted_patch_validation.json` records its
hashes and a clean application check in the independent baseline checkout.
The separate report commit preserves all review evidence. Temporary baseline
and stage-validation checkouts are removed after validation; no server is
left running and nothing is pushed.
