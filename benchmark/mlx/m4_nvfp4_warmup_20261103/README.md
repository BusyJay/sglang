# Move NVFP4 setup and compilation to startup

2026-11-03. Startup warmup now evaluates the eager routed FFN and the model's
real decode/verification graph owners before the runner completes construction.
Remote unavailable; only local M4 evidence and apply-checked patches exist.
This moves one-time setup from the first request; it does not increase steady
decode throughput or establish the remote40/90 tok/s targets.

## What changed

`sglang.srt.hardware_backend.mlx.warmup` owns the runtime warmup. Model
`prepare_for_inference()` materializes each layer, executes its NVFP4 switch
FFN for FP32/BF16 count1/count2, and evaluates actual prefill/decode after all
layers/head are ready. This covers the AR fused SwiGLU and small-batch expert
sharing paths without copying checkpoint storage or compiling fake kernels.

After constructing the speculative adapter, the runner additionally invokes
model warmup using actual DFlash proposal count/capture layer IDs and greedy
vs sampled output mode. It warms count2 and the configured depth up to8;
decode runs last. Warming every possible depth could evict the bounded graph
cache and waste startup time, so this is a deliberately bounded shape set.
Adaptive native MTP selects the maximum covered depth. Count>8 verification,
multiple requests and unseen cache layouts are not covered by this addition.

Every warmup evaluates outputs and private cache state using mx.eval. Temporary
startup caches never enter request ownership/prefix storage. Parameter arrays
are reused. No RNG state, cache state of existing requests, checkpoint values,
quantization or model equations are changed. No install, download, system
memory setting, remote deployment, commit or push occurred.

The existing runner already calls model.prepare_for_inference during loading;
the new runner hook happens after speculative adapter creation and before pool
and scheduler initialization. Direct model users also get decode warmup from
prepare_for_inference, and can call warmup_for_inference explicitly after final
serving configuration for their verification depth/capture set. Changing eval
groups or model configuration after warmup can invalidate prepared graphs.

## Why both levels matter

Archived MLX source confirms CustomKernel.eval_gpu obtains/creates the Metal
library and compute pipeline at GPU evaluation; Device caches library/kernel
objects. Creating a Python callable or wrapping it with mx.compile is not
sufficient. The same outer compiled function must be called and evaluated.

An initial same-process probe in initial_probe.json measured eager-first6.594
ms, eager-second0.708 ms, then compiled-first210.022 ms and compiled-second
1.664 ms. Its first compiled evaluation was slow even after eager execution.
This is illustrative setup evidence, not a symmetric benchmark or fixed cold
compile duration. Prior83–84 ms results were process-local first evaluations
with unequal cache history; neither83 ms nor210 ms is a constant overhead.
OS/compiler disk cache was not cleared or relocated.

Relevant archived primary source:

- `../dtype_audit_20261030/sources/mlx/mlx/backend/metal/custom_kernel.cpp`
- `../dtype_audit_20261030/sources/mlx/mlx/backend/metal/device.cpp`
- `../dtype_audit_20261030/sources/mlx/docs/src/usage/compile.rst`

Runtime K/N/picks avoid injecting changing dimensions into this custom Metal
source. Outer model graph keys still include shapes, dtype, layout, flags and
cache arrays, so unseen shapes/layouts can compile later. Existing fixed graph
owners pad small MLA prefixes/capacity buckets, but this round does not prove
all growth boundaries are free of compilation. Per-step graph submission and
small runtime metadata allocation remain; they cannot all be moved to startup.

## M4 measurements

M4 Pro48 GiB, applegpu_g16s, MLX0.32.4, Python3.14.9, existing virtualenv and
Xcode/MetalToolchain. No third-party installation. Same actual-width synthetic
NVFP4 operands, top8 compact expert routing, four MoE layers, K4096/N2048
gate/up and K2048/N4096 down, four dependency samples in ABBA order. Each sample
is a separate process and has300 steady dependent steps after the first call.
The same compiled chain objects used in the timed call are evaluated during
startup in the warm arm. New process does not imply a cold OS/compiler cache.

| Experiment | Unwarmed first step ms (two samples) | Warmed first step ms | Startup warmup ms |
| --- | --- | --- | --- |
| FP32 four-layer chain | 10.015 /5.175 | 2.176 /2.210 | 4.940 /5.108 |
| BF16 four-layer chain | 5.958 /5.543 | 2.128 /2.127 | 4.844 /6.391 |

Raw samples are chain_{dtype}_{lazy,warm}_{0,1}.json; chain_summary.json groups
them. Steady sample means were FP32 lazy2.217/2.148 vs warm2.227/2.138 ms,
BF16 lazy2.297/2.098 vs warm2.263/2.314 ms. No steady throughput gain is claimed.
Peak device allocation0.56 GiB, swap0. Original candidate parity evidence
remains in the preceding final-source bundle; no kernel arithmetic was edited.

model_probe.json additionally executes the final remote model and runner helper
with synthetic two-layer GLM KDA+MLA, hidden512,8 NVFP4 experts. The model module
is real; archived FP4 classes and runner method are AST-isolated to bypass
unavailable serving/import dependencies. Final measured preparation12.909 ms,
configured verification/greedy warmup20.181 ms, subsequent decode first1.150 ms
and subsequent samples1.116–1.283 ms. Those small-model times cannot estimate
the45-layer full-checkpoint startup cost. Existing cache contents/offsets remain
unchanged and fresh request output is bitwise equal. Peak device~26 MiB.

## Validation and delivery

One registered Metal test passes with four subcases. A deliberate invalid lazy
shader is placed in each FP32/BF16 AR/pair variant: startup must observe the
failure. The graph-construction-only mutant fails all four subcases, proving
the synchronization boundary is tested. The isolated harness changes only test
import/CI glue; the shader is real device compilation. The ordinary full runner
lacks torch/msgspec locally and is not claimed passed.

Registered-test lint, no-package-tests lint, py_compile and git diff --check
pass. ruff/black/isort/clang-format remain unavailable and were not installed.
No full HTTP readiness, full checkpoint generation, MTP/draft acceptance or
remote cache lifecycle was tested. Startup on full remote weights can be
materially longer; log and measure it with the unchanged process resource limits.

Patch choices:

- `remote_warmup_incremental.patch`: atop the prior final AR NVFP4 patch.
- `remote_nvfp4_with_warmup_cumulative.patch`: previous NVFP4 reuse+AR fusion
  plus current startup warmup, from remote303af39b966c48e18d2268bdd610405e1316dc74.
- `local_runner_warmup.patch`: local HEAD runner only, separate from remote.

The remote runner baseline was recovered from
../fp4_dflash_20261103/remote_source and verified against its source_manifest.
Both remote forms apply-check/apply/final hash compare pass; patch_manifest.json
records model, runner, helper, test and patch hashes. Existing remote runner
memory-budget and packed-head fixes are retained. Remote __init__ inventory
was not fabricated; prior kernel modules are imported directly by the model.

From repository root:

```sh
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_warmup_20261103/run_device_test.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_warmup_20261103/model_probe.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_warmup_20261103/run_chain.py
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_nvfp4_warmup_20261103/check_patch.py
```

Final delivery seals source and raw results. Portable bundle excludes weights,
environments, __pycache__ and .gputrace. Reuse the preceding evidence bundle for
archived fixture dependencies. Full remote verification remains: inspect live
HEAD/dirt, apply one patch form, keep same checkpoint/config, compare first
request and300-token steady requests in ABBA, record startup time, outputs,
routing, acceptance, memory and cache/layout growth. No background benchmark
remains running.
