# M4 NVFP4 expert reuse: root cause, final implementation and evidence

2026-11-03. This continues the failed remote same-expert candidate instead of
ending at its throughput regression. The remote is unavailable. All new device
measurements are local M4 Pro; no M5 full-model/HTTP throughput is claimed.

## Outcome and final source

A bounded two-vector gathered QMV shares the packed NVFP4 weight load and
E4M3 scale decode across two picks. One GPU expert-pair worklist is reused by
gate/up/down. Odd occurrences execute a native-order singleton in the same
kernel. No concatenated weight bank, checkpoint conversion, activation
quantization, CPU readback or model-global mutable state is introduced.

The reviewed model patch enables the mechanism by default for 2–8 input tokens,
including four-row DFlash verification. Single-token AR and >8-token prefill
retain their original native projection path: AR has no inter-token reuse, and
large-batch scheduling was not validated. This is a measured, scoped delivery,
not a claim of universal optimal NVFP4 execution. No production feature flag
was added. The callable kernel also supports inline pairing for independent
projection callers; the model patch uses explicit shared worklists.

Production source is in:

- `python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py`
- `python/sglang/kernels/ops/moe/mlx_expert_pairs.py`
- `test/registered/kernels/ops/gemm/test_mlx_nvfp4_gather.py`

These paths have lazy KernelSpec registration in the local GEMM/MoE groups.
The remote model files are absent in this checkout, so the model change is a
reviewable `remote_nvfp4_integration.patch`, not a remotely installed change.
No commit or push was made. Previous local FP8 work and unrelated files remain.

## Final measurements

M4 Pro, 48 GiB, `applegpu_g16s`, MLX 0.32.4, Python 3.14.9. Existing interpreter
`.venv/mlx-dtype-audit/bin/python`; existing Xcode/MetalToolchain and gpudebug.
No tools installed, weights downloaded or system wired-memory setting changed.

All final macro arms run 300 dependent steps with alternating A/B/B/A order,
four samples per arm. Below is the median per-step wall time. Each step executes
four distinct MoE FFNs with actual widths: gate/up [2048,4096], down [4096,2048],
top-k8, and the archived routing pattern remapped to 19 compact experts. The
baseline uses unchanged native `mx.gather_qmm(mode='nvfp4')` on the same packed
synthetic operands. Every sample's final output is bitwise equal and finite.

| Final-source experiment | Native ms | Candidate ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Four-row FP32 four-layer MoE chain | 7.448205 | 5.411710 | 27.34% |
| Four-row BF16 four-layer MoE chain | 7.512842 | 5.390906 | 28.24% |
| Exact remote ClampedSwitchGLU class, FP32 count4, single layer | 2.055971 | 1.486981 | 27.67% |
| Exact class AR count1, single layer | 0.634622 | 0.632094 | ~0.4%; no meaningful AR claim |

The actual-class rows use the mean of two 300-step samples per arm in ABBA.
`integration_check.py` AST-extracts the exact original/patched model classes
and original NVFP4SwitchLinear; only unavailable serving dependencies are
isolated. It uses the actual activation method, weights and ordered combine.
Count1/4/8/9 × FP32/BF16 outputs match. This is class integration with synthetic
weights, not a server/request-cache or full-checkpoint test.

Final-source macro files are:

- `moe_macro_real_c4_fp32_4layers_production,production_inline.json`
- `moe_macro_real_c4_bf16_4layers_production.json`
- `integration_check.json`

`final_summary.json` explicitly marks which earlier macro JSONs match final
production SHA256. Early iterations are retained with their hashes. Some raw
output filenames were reused; the associated v2/v3 log files and `iterations/`
freeze the earlier mechanism. Old unrelated results are not rewritten.

Peak device allocation for final four-layer chains is 1,283,458,096 bytes
(~1.20 GiB); synthetic packed weights/scales are 962,593,584 bytes (~0.90 GiB). Before/after swap
used is zero. No full GLM-5.3 checkpoint fits the same experiment: the remote
171 GiB model/draft and its much larger changing expert working set are not
represented by this compact bank. No draft acceptance or generation-quality
metric was measured locally. Do not multiply these speedups by earlier FP8
speedups, or convert them into M5 tok/s.

## Root causes and experiments

The frozen remote candidate is `frozen/nv_shared.py`, from the actual tested
remote snapshot, not the later top-level draft. The archived native source is
`frozen/native_fp_quantized.h`; original native qdot/fast-QMV were used for the
arithmetic localization. The compact top8 pattern is in `common.REAL_IDS`.

1. Reproduced old candidate and independently removed per-tile scanning using
   a GPU worklist. Worklist-only was not sufficient: final initial four-layer
   chain 7.597→9.013 ms, +18.63% latency, despite exact arithmetic after its fix.
   Strongly unrolling all four token accumulator arrays was worse than leaving
   dynamic indexing. The GPU profile reports 336 spilled bytes for the old
   dynamic kernel, prebuilt dynamic list, and four-static version. Four-static
   instructions rise from 356 to650 in that capture, with extra loads/stores.
   Those are compiler-reported per-shader statistics, not allocated tensor bytes.

2. Located the numerical change separately. At K512 the safe single-row custom
   kernel matches native; at K1024/4096 it diverges, while unchanged native
   fast-QMV remains exact. Cross-block `result += scale*accum` was contracted
   into a different rounding path. Explicitly separating only that product
   restores bitwise equality. Disabling contraction globally also changes
   within-block arithmetic and is incorrect; `parity_probe.json` records both.
   No tolerance was widened. `regression_proof.json` removes only the scoped
   rounding guard: final kernel changes 0 values, mutant changes1321 values.

3. Bounded sharing to two picks with two static accumulator arrays. Native-order
   block rounding was first enforced with volatile products: exact, faster,
   but16 spilled bytes. A scoped `#pragma clang fp contract(off)` around just
   the cross-block update preserves native qdot arithmetic and removes those
   spill accesses. Final production shaders report **zero spilled bytes** and
   64 temporary registers for both inline and explicit-worklist paths.
   `profile_final_shaders.log` corresponds to final source.

4. Compared RPS1/2/4/8 for pair output geometry. RPS4 is selected for the actual
   gate/down widths. Wider eight-output singleton tiles were then investigated
   to reduce no-repeat overhead: exact but four-layer no-repeat chain slowed
   7.584→8.002ms (+5.5%). This independent attempted improvement is rejected;
   the prior four-output singleton is retained, rather than silently disabling
   two-token reuse. Source is `iterations/production_before_single8.py`, and
   the mutated trial is reproducible with `single8_trial.py`.

5. No-repeat routing is still a boundary: same-source four-layer count4 result
   7.630→7.719ms (+1.16%) for explicit worklists, +0.78% inline. This small cost
   is the worklist/control path without weight-reuse benefit; no speedup is
   claimed there. Wider singleton tiles worsened it. The final default remains
   enabled in verified small batches because real overlap showed consistent
   material gains. Uniform dispatch specialization/compacting active groups
   remain implementation opportunities, not Metal platform limitations.

Native single-row AR macro has no stable improvement. This mechanism cannot
solve the user's AR 40tok/s target; preceding shared FP8 optimization remains
a separate candidate for that path. Remote DFlash baseline ~50tok/s and target
90tok/s remain unverified against the final combined patches.

## Validation and limits

Two registered numerical methods passed on real Metal, through
`run_device_tests.py` with dependency glue isolated. They cover FP4 nibble
signs, subnormal/max-finite E4M3 values, different global factors, repeated
experts within/across tokens, odd/even counts, changing route lists, per-pick
activations, noncontiguous activations, FP32/BF16 and K512/1024/4096. The normal
registered runner cannot import in this environment because torch/msgspec are
not installed; it was not declared passed. `device_tests.log` is explicit.

Repository registered-test and no-package-tests lint pass, AST/import syntax
and `git diff --check` pass. ruff/black/isort/clang-format are unavailable;
no formatter was installed. Kernel code is manually reviewed. This specialized
operator contract requires K%512==0,N%8==0 and valid uint32 IDs. Those are this
native-order QMV implementation's coverage, not hardware limitations.

No cache arrays, offsets, checkpoint values or acceptance decisions are changed.
Full HTTP normal decode, prefix reuse, chunked prefill and request lifecycle,
plus full-model outputs/routing/draft acceptance, still need remote testing.
Small batches without expert overlap and remote working-set behavior remain
performance risks; they are documented rather than hidden by a feature flag.

First-call times are included in macro JSONs; they include per-process compiled
chain setup, but do not prove cold OS/Metal-cache or full server startup cost.
K,N,pick count and picks are runtime dims; there is no ROWS=count accumulator
specialization or context-length constant in this kernel. Model cache growth
and context-dependent attention compilations are outside this experiment.

## Patch and reproduction

Remote baseline HEAD `303af39b966c48e18d2268bdd610405e1316dc74`, glm5_next SHA256
`2c2f0c3afb0fd7e53988aaaed82922e2d54bdadc9427680aeb4f148a1ee35691`.
`patch_manifest.json` seals model and new-file hashes; `patch_apply_check.json`
records apply-check and final hash equality against the archive. Previous FP8
integration patch applies cleanly after this patch. It includes the standalone
kernels, numerical test and model branch. Local KernelSpec additions are in
local __init__ files; no unverified remote __init__ baseline is fabricated.

Commands, from repository root:

```sh
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/run_device_tests.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/moe_macro.py --modes production,production_inline
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/moe_macro.py --dtype bf16 --modes production
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/build_patch.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/check_patch.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_rootcause_20261103/integration_check.py
```

Portable evidence excludes weights, environment and all `.gputrace` directories.
Native captures stay local; shader stats and logs are in the portable evidence.
