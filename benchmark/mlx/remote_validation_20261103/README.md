# M5 macro and full-model validation — 2026-11-04

The 40 tok/s autoregressive and 90 tok/s DFlash2 targets have not been achieved.
Final HTTP checks passed; the final remote health snapshot recorded DFlash port
30007 healthy (PID 69989), with no experiment process left running. The stride
ablation completed without a gain. This is historical, not a current availability check.

The complete Chinese mechanism and exclusion ledger is [EXPERIMENTS.md](EXPERIMENTS.md).
This archive is committed in the target host's /Users/jay/workspace/sglang on 2026-11-04.
A local companion commit was made first; the user clarified that the target-host commit was required. No Git hosting push is performed.
The inherited production working-tree changes remain separate; final source is sealed as patches.
This evidence commit does not stage production code or pyproject changes.
Failure to reach 40/90 does not establish that the platform has no optimization space.

## Environment and comparison

Remote Apple M5 Ultra, 256 GiB, applegpu_g17d; MLX 0.32.4. Remote repository
HEAD 303af39b966c48e18d2268bdd610405e1316dc74 on metal, with archived task
baseline that includes the existing FP4 implementation. Local checkout differs;
do not replace remote model source with the local model file.

Target nvidia GLM-5.3-Flash-FP4-Local revision
76348ddb2be2e35e39ecfc00e75dc07703f4f3d5; DFlash2 draft revision
bf582e4eacc1810f76656d1811693ff6c6737d2a. No checkpoint codes, scales,
quantization or model equations changed. Target+draft resident ~171.34 GiB.
Full-model and full-bank benchmark scripts set process memory/wired limits to
78% of recommended working set (~187.2 GiB), MLX cache 1 GiB,
ops per buffer 200, buffer threshold 1024 MiB; no system memory adjustments.
Local projection scripts use smaller budgets. HTTP uses the ordinary serving
runner defaults and --mem-fraction-static 0.78; this flag must not be equated
with the benchmark script's wired budget. System swap was already nonzero;
raw before/after values do not attribute paging to an individual process.
Short natural prompts generate 300 tokens. Full-model comparisons use the same
resident arrays, reverse arm order, clear/rewarm graph owners, and record raw
per-token timings, generated IDs, acceptance counters and memory.

## Findings

1. Same synthetic four-layer M4 macro rerun on M5: native 1.611 ms → candidate
   1.518 ms (~5.8%), versus ~27% on M4. This is a synthetic dependent macro,
   not a prediction of full-model throughput.
2. Actual routes pair only 36.9% of picks on average, versus 68.75% in the old
   M4 macro. Acceptance counters and generated tokens remain exact across the
   full-model candidate comparisons; the measured regression is not acceptance.
3. Rotating all 42 routed FFN banks reproduces the current RPS4 regression:
   native 14.857 ms → 18.273 ms. RPS1 improves this isolated FFN macro to
   14.271 ms, but complete model DFlash remains 49.095 vs FP8 reference 51.046
   tok/s. Geometry fixes the isolated macro, not the real generation path.
4. `macro_envelope.json` passes actual weights as explicit graph inputs:
   14.773 → 14.286 ms. Explicit weight binding alone does not erase the gain.
   Complete MoE with router, shared FFN and combine: 20.483 → 20.895 ms.
5. `macro_composition.json` narrows the reversal: combine 15.361 → 14.620 ms;
   captured-route shared+routed 18.956 → 17.768 ms; router computed inside graph
   20.429 → 20.891 ms. Reordering branches 20.451 → 20.869 ms does not fix it.
   Actual captured FP32 input uses the existing general router; the fused router
   does not cover this dtype. This is an implementation coverage gap, not an
   established Metal hardware limitation.
6. Explicit contiguous route materialization does not improve the candidate
   (20.968 → 21.014 ms). Only gate/up reuse (23.316 ms) and only down reuse
   (24.045 ms) both regress versus native 20.516 ms. General routing and
   composition are a reproducible boundary for gain reversal; the low-level
   scheduling/cache cause is not yet completely established.
7. Observer bypass has negligible critical-path benefit; larger eval groups,
   buffer thresholds, compact banks, fixed-width specialization, FP4 decode
   arithmetic variants and lossless FP8 expansion did not deliver complete-model
   gains. FP8 expansion cost +1.3125 GiB. GPU captures show no spilling but were
   serial separate captures; register/static instruction counts do not prove
   utilization or causality.

Route recomputation reproduces 41/42 layer index tensors; layer 6 differs by one
expert (not just ordering). Both candidate arms share the same computed router
in the composition macros. Capture-vs-live comparison is attribution, not a
numerical equivalence proof of the whole model. Full-model A/B tokens and
acceptance counters are exact.

## Decision

Retain paired shared block-FP8 projection and startup graph warmup. Three-prompt
full-model baseline AR/DFlash means 30.612/50.961; FP8 30.441/52.874 tok/s.
DFlash gain is modest; AR shows a small negative delta that must not be called an
AR throughput improvement. Shared FP8 scales with serialized singleton axes
are accepted by element count, preserving row-major data and values.

Remove the NVFP4 model integration after repeated complete-model regressions
and mechanism improvements failed to remove them. Restore the archived original
NVFP4 branch; no default-off flag is added. Keep candidate kernels and diagnosis
as reviewable unfinished work. Narrow/fused/reordered/contiguous/hybrid/half
variants are diagnostic only and are not enabled in serving.

Further optimization needs live-router + full-MoE + real dependency measurements,
including general-router output production and custom-kernel submission/resource
scheduling; a fixed-route hot FFN macro is insufficient evidence for adoption.
AR acceleration and the remaining DFlash gap remain unresolved.

## Evidence and reproduction

`results/` contains raw JSON with timings, device/checkpoint/source identities,
IDs/counters and memory. `final_model_summary.json` aggregates fullmodel results;
it does not aggregate macro times into tok/s. Full-model script
`fullmodel_ab.py`, then envelope/composition/hybrid scripts represent progressively
wider attribution boundaries. `.sh` launchers fix the resource environment.
All remote access uses authenticated tmux session sglang-remote pane %0 through
`tmux_pane.py`; full logs remain under the remote evidence directory. Large
GPU captures, model weights and virtual environments must not enter portable
artifacts. See the ledger for final checks, historical decisions and reproduction prerequisites.
Scripts are preserved as executed, with remote absolute paths and historical
module dependencies. They were not reformatted for this archive.
actual_ffn_operands.npz remains remote; rootcause_replay.py can recapture it,
but new routes are not the identical historical operand fixture.
Scripts can overwrite output files: preserve old results before rerunning.
The final removed NVFP4 model branch cannot reproduce the historical four-arm
comparison directly. Historical patches and baseline hashes are recorded in
the ledger; final patches represent the selected deployment, not old candidates.

## Final HTTP validation

Selected source: three natural prompts, 300 output tokens, two repetitions.
Decode rates AR 29.5463 and DFlash 51.1319 tok/s mean. Per-prompt DFlash
48.83–55.57 tok/s; all six AR/DFlash output ID sequences exactly match.
Prefix reuse at 32/514 prompt tokens, chunked prefill, logprobs, seeded sampling,
stop tokens, concurrent requests and abort followed by reuse all passed.
These checks are functional, not broad model quality evaluation.

Startup-to-health 42.35/41.35 seconds AR/DFlash. First natural prompt TTFT
1.286/1.323 seconds vs later mean ~0.203 seconds: startup warmup does not yet
cover all request-related setup/shape costs. Do not claim all compilation was
moved ahead. HTTP request rate is distinct from in-process model measurements.

Final device numerical checks: 2 FP8, 3 NVFP4, 1 warmup methods; GLM runtime
65 and DFlash serving 9 methods, DFlash2 numerical checks, pool 9 methods pass.
Ruff checks and format checks on nine candidate files pass.

Stride-aware expert worklist and projections preserve all finite/exact outputs,
but the live-router MoE macro regresses: native 20.5874 ms, normal reuse1
21.0706 ms, direct stride reader 21.3523 ms. This does not establish copies as
the sufficient root cause; the no-copy rewrite is not adopted. Remaining low-level
GPU scheduling, locality and submission behavior in the complete live-route graph
need further attribution. The full-model stride script was not run after its
complete-MoE macro regressed.

Kernel metadata added for the five candidate operators. Registry probes resolve
all five; no new Ruff violations versus the three archived registry files.
Full namespace suite: 117 passed, one unrelated AMX lazy-probe subprocess failed
because this Metal environment has no Triton. The failing common.py is identical
to remote HEAD. No dependency was installed and no assertion was disabled.

Final model SHA256: 7bde9e42538edb61545889f691c842b959189c46b09275f4116e110293d2f2f1.
`results/final_source_manifest.json` seals all 15 relevant sources. The selected
candidate and registry patches apply to the archived task baseline (remote HEAD
plus existing FP4 integration), not a clean checkout of that HEAD. Isolated
application recreates all 14 changed/new files with exact final hashes.
The previous cumulative candidate patches are historical experimental inputs,
not the final deployment deliverable. M4 reports retain their original local
adoption decisions; the M5 removal decision above supersedes them.

The archive contains source copies, raw evidence, final patches and a file hash
manifest. verify_archive.py checks archive consistency; it does not rerun GPU
or HTTP validation. The three remote registry init files are represented by a
patch and the remote apply-check result, not by complete local source copies.
