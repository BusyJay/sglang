# M4 AR NVFP4 gate/up + SwiGLU fusion

2026-11-03. This round adds an independent AR optimization after the preceding
small-batch expert-reuse work. The remote remains unavailable. Final local
source and a reviewed remote model patch are prepared; nothing was deployed,
committed or pushed. The measured AR benefit is small, and the requested remote
40 tok/s AR / 90 tok/s DFlash2 targets remain unverified.

## Mechanism and default behavior

`python/sglang/kernels/ops/gemm/mlx_nvfp4_swiglu.py` projects two independent
native NVFP4 gate/up banks and applies the clamped SwiGLU epilogue in one Metal
kernel. It shares activation loads, removes intermediate projection tensors
and preserves the native half-based FP4 decode, block reduction order, separate
gate/up dtype rounding, BF16 sigmoid intermediate overloads and both product
rounding boundaries. Checkpoint values, quantization and equations are unchanged.

The cumulative remote model patch defaults to this path for count1 AR, retains
the previous shared-expert worklist/projections for count2–8, and retains the
original native projections for count>8. No diagnostic environment flag or
shared model state is added. The local kernel has lazy GEMM KernelSpec
registration. The remote model files are absent in this checkout, so their
integration exists as a patch plus the actual archived-class integration test.
The patches do not invent changes against an unarchived remote __init__.py.

K%512==0, N%8==0, FP32/BF16 inputs and valid uint32 expert IDs are this
implementation's coverage, not established Metal hardware limitations.
K, N and pick dimensions are passed at runtime. Input layout and dtype/clamp
variants still participate in compilation; context-growth behavior elsewhere
in the model was not tested here.

## Final measurements and boundaries

M4 Pro 48 GiB, applegpu_g16s, MLX0.32.4, Python3.14.9, existing
`.venv/mlx-dtype-audit/bin/python`, Xcode/MetalToolchain and gpudebug. No tools,
weights or libraries were installed, and no system memory setting was changed.

Each sample runs 300 dependent steps. Four samples per arm use alternating
A/B/B/A order; actual-class tests use two ABBA cycles after ten warmups per arm.
Each synthetic layer uses gate/up N2048,K4096, down N4096,K2048, top8 routing,
the same packed operands and native gather_qmm baseline. AR uses eight compact
experts remapped from the archived route, rather than the full remote checkpoint
and its changing expert working set. Final chain outputs are bitwise equal and
finite. The reported statistic is the median of sample mean per-step times.

| Final-source experiment | Native ms/step | Candidate ms/step | Latency decrease |
| --- | ---: | ---: | ---: |
| AR FP32, eight-layer compact MoE chain | 4.156680 | 4.006215 | 3.62% |
| AR FP32, four-layer compact MoE chain | 2.185346 | 2.118723 | 3.05% |
| AR BF16, four-layer compact MoE chain | 2.216781 | 2.131678 | 3.84% |
| Actual archived ClampedSwitchGLU class, AR FP32, one layer | 0.626127 | 0.616171 | 1.59% |
| Actual class count4, previous expert-reuse mechanism | 2.003659 | 1.439582 | 28.15% |

Count4 is a recheck of the previous mechanism, not a gain from this new AR
fusion. Do not multiply these percentages by previous FP8 results or convert
compact M4 chain latency to remote M5 full-model tok/s.

The actual-class AR sample means are baseline
[0.624821,0.627434,0.620375,0.680358] and candidate
[0.707682,0.606002,0.615754,0.616589] ms. Including all samples, arithmetic
means are 0.638247 vs0.636507 ms, only 0.27% lower. The candidate has a slow
sample; the benefit is not established as stable at this scope. Eight-layer
FP32 raw sample means are baseline [4.161636,4.151724,4.055084,4.192296] and
candidate [3.975694,4.007714,4.004716,4.044883] ms. Four-layer samples are
noisier. The earlier four-sample actual-class result in
`iterations/integration_first.json` has a slow native sample; its larger
apparent gain is not used as the final result.

First candidate evaluations were 83.877 ms for four-layer FP32 and83.142 ms
for BF16, versus the already evaluated native path's2.184 and2.224 ms. The
eight-layer FP32 run later measured4.649 ms for candidate vs4.034 ms for native.
These are process-local first evaluation measurements with unequal cache
history, not a symmetric cold-cache experiment or whole-server TTFT. They
expose setup/JIT cost but do not isolate its source or prove short-request
end-to-end improvement. Remote startup warmup, first request and steady state
must all be compared. No GPU cache was cleared to manufacture cold conditions.

Peak device allocation was603,980,320 bytes (~0.56 GiB) for four layers and
1,056,965,536 bytes (~0.98 GiB) for eight layers. Macro before/after swap used
was zero. These compact synthetic banks are much smaller than the remote
171 GiB model/draft. Full checkpoint outputs, routing, draft acceptance,
generation quality and HTTP/cache lifecycle were not measured. AST isolation
executes the actual model-class methods and NVFP4SwitchLinear on Metal but
bypasses unavailable serving dependencies; it is not full server validation.

## Root causes pursued and candidate decisions

1. Bare gate/up dispatch pairing gave only roughly1–2% in the microbenchmark
   and no four-layer macro improvement. The RPS2 ablation measured native
   2.159591 vs pair2.185612 ms. Dispatch fusion alone did not remove most of
   the projection cost; the final mechanism also fuses the activation epilogue.

2. Direct FP32 bit decoding was exact but roughly60% slower in the early
   microbenchmark. GPU profiling localized the extra work: the compared pair
   shaders went from565 to1097 instructions and INT32 instructions from63 to383,
   with no spill. Branchless and scalar half variants reduced some of that
   cost but remained slower; vector-half decoding compiled to the same565
   instructions as the native-order pair. These alternatives remain diagnostic
   experiments and are not adopted. This is an implementation/code-generation
   finding, not a claim that Metal cannot decode FP4 efficiently.

3. The initial fused BF16 path promoted sigmoid inputs to FP32 and changed
   rounding. Keeping native BF16 overloads restores exact results, including
   clamping to a weak scalar rounded to the input dtype. The final regression
   proof changes0 outputs; the FP32-promoted-sigmoid mutant changes5585 outputs
   with max abs1.52587890625e-5. No tolerance was relaxed.

4. RPS1/2/4/8 and different SIMD-group counts were compared. RPS1 with a
   32x2 threadgroup is retained for the final AR candidate, supported by the
   dependent-chain measurements rather than a claim of universal best geometry.
   Final GPU capture has11 native dispatches vs3 fused dispatches, including
   two encoded-global-factor multiplies in both ranges. The native range also
   has metadata and activation operations. The final shader reports zero
   spilled bytes,39 temporary registers and366 instructions. The native QMV
   reports739 instructions/62 registers and is invoked twice, but geometries
   differ: raw instruction counts cannot be multiplied into an end-to-end
   speedup. Capture-replay cost shares are50.317% for the native projections
   and48.814% for the fused shader; replay time is not ABBA wall latency.

The final rounded epilogue candidate is retained and default-enabled in its AR
model patch. The previous count2–8 mechanism is preserved. Timing noise and
first-evaluation overhead remain open, and this change is not a large AR
acceleration. If remote ABBA finds a regression, localize setup, submission,
working-set and shader costs before revising the mechanism; a default-off flag
does not replace that work.

## Validation, provenance and reproduction

Three registered numerical methods pass on real Metal via the existing
dependency-isolated device harness. Coverage includes FP4/E4M3 codes, cross-block
rounding, repeated experts, per-pick inputs, FP32/BF16, K512/1024/4096 and fused
limits None/7/0.337. Actual class count1/4/8/9 x FP32/BF16 is exact; count1/4
300-step class chains are exact. Registered-test lint, no-package-registration
lint, syntax compilation and git diff --check pass. Normal registered/serving
imports lack torch/msgspec. ruff/black/isort/clang-format are unavailable;
none were installed. No cache code was changed; normal decode, MTP consumption
and rejection, chunked prefill, prefix reuse and request lifecycle require
remote full-model validation.

Final-source data: `macro_c1_fp32_r1.json` is the eight-layer final run;
`iterations/macro_c1_fp32_r1.json` is the four-layer final run;
`macro_c1_bf16_r1.json` is the four-layer BF16 final run.
`final_summary.json` marks earlier experiments by production SHA;
`integration_check.json` contains final raw class results. Final kernels/test
are frozen in `final_source/`. GPU captures remain local; portable evidence
includes shader and command statistics and excludes all .gputrace directories.

Remote base303af39b966c48e18d2268bdd610405e1316dc74 and model/file/patch hashes
are sealed in `patch_manifest.json`. `remote_nvfp4_cumulative.patch` includes
both the previous reuse mechanism and this AR mechanism, from the original
archived model. `remote_ar_swiglu_incremental.patch` applies only after the
previous round's NVFP4 integration patch. Both apply-check, apply and final hash
comparison pass; both are compatible with the preceding FP8 patch. Use the
cumulative final patch for a fresh application: the previous round's frozen
test has two methods while the current final test has three.

From the repository root, using the existing environment:

```sh
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/run_device_tests.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_pair_20261103/regression_proof.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_pair_20261103/macro.py --count 1 --layers 8 --dtype fp32 --rps 1 --modes production
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_pair_20261103/macro.py --count 1 --layers 4 --dtype bf16 --rps 1 --modes production
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_pair_20261103/integration_check.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_pair_20261103/check_patch.py
```

The evidence bundle includes archived helper dependencies for these scripts
and final source; overlay it on a repository checkout rather than treating it
as a standalone serving distribution. Rerunning scripts overwrites their named
results, so preserve the sealed bundle first. No background benchmark remains.
