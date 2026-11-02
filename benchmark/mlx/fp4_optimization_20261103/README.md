# NVIDIA GLM MLX short-context optimization — 2026-11-03

Remote repository: /Users/jay/workspace/sglang, branch metal.
Baseline: 53963f6369d71c9d3ef24c817ece002ed769125b.
Hardware: Apple M5 Ultra, 256 GiB; MLX 0.32.4. NVIDIA is the checkpoint publisher, not the serving GPU.
Checkpoint: nvidia/GLM-5.3-Flash-FP4-Local, revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5.
Native MTP2, unchanged checkpoint and activation quantization, MLX_ENABLE_TF32=0.

## Final same-machine full-model decode A/B

| Prompt | Baseline tok/s | Optimized tok/s | Change |
| --- | ---: | ---: | ---: |
| English explanation | 36.5647 | 44.5223 | +21.76% |
| Python code | 40.1724 | 48.7146 | +21.26% |
| Chinese explanation | 39.5704 | 47.8228 | +20.85% |

Source: final_ab.json, final_ab.py, final_ab.log, summary.json.
Single resident model, three short chat-template prompts, two repetitions, alternating A/B order; 128 output tokens, measured 127 decode steps (prefill excluded). Tokens are forced to the requested length, so this is not a complete-answer HTTP qualification. Raw per-step times, prompt IDs, output IDs, draft counts, first/steady costs and memory are retained. All 12 sample output sequences and draft/reuse counts matched within each prompt. Baseline uses original MXFP8 tile8, block output RPS4, separate head history/root, 64 MiB buffer cache and group4. Optimized uses tile4, block RPS2, merged history/root, 1 GiB buffer cache and decode/verify group2; prefill remains group4. New graph-observation code was active in both final arms; its independent A/B is observation_ab.json and is mixed/near-neutral. This is not an all-source pristine checkout comparison.

Peak measured allocation: 174.070679 GiB. Existing swap stayed 1219.38 MiB before/after. No system wired-memory settings were changed. The diagnostic runner used matching process memory/wired limits of 188 GiB in both arms. Goal 90 tok/s has not been reached.

## Adopted mechanisms and configuration

Four-row native MXFP8 padding reduces small verification work while preserving stable batch reductions. Block-FP8 output SIMD geometry reduces MLA output projection cost. Pending MTP history and root share a ragged head pass with independent request caches. Class snapshots avoid scalar tuple construction while retaining graph mutation invalidation; no independent speed claim is made for this last mechanism.

launch_optimized.sh records the serving configuration: existing cache-limit knob 1 GiB and decode/verify eval groups2, native MTP2, context8192, total tokens17024, chunk256, max requests2, seed1234, deterministic sampling and the original parser/memory settings. Code mechanisms are enabled by their normal paths. Runtime tuning is selected in this launcher; no new global environment default is introduced.

## Completed validation and scope

Real MXFP8 checkpoint invariance: 24 BF16/F32 batch cases exact across q/o/head shapes and row counts1/3/6/9 (mxfp8_invariance.json). GLM device suite: 65 tests passed in 7.374s (glm_tests.log), including ragged head history/caches and graph mutation coverage. FP8 kernel suite: three tests passed. Five targeted MTP serving lifecycle tests passed. Ruff F/I and formatting checks passed for the changed FP8/MXFP8 kernel files; git diff --check passed. Changed methods were subsequently formatted with AST equivalence checked. source_manifest.json records final source and evidence hashes.

Final optimized HTTP complete-answer/service lifecycle panel was not run: user requested stopping further testing and committing. http_baseline.json covers the earlier service only. Long-context optimization/qualification was deferred at user request. Health verification on restart establishes readiness only. No claim of final HTTP numerical/quality qualification is made.

## Investigated candidates

projection_ab.json and runtime_ab.json identify the principal gains. group_ab.json and mtp_ab.json cover execution grouping/history merging. tile2_ab.json regressed code/Chinese despite faster single-row microbenchmarks; tile4 retained. A 4 GiB buffer cache offered no further stable benefit. Deeper MTP variants slowed this workload; MTP2 retained.

Block K-split reduction was faster in a microbenchmark but full-model behavior changed token/routing acceptance and had unstable speed (block_split.json, split_ab.json); not integrated. Gate/up fusion was investigated with shared quantization, bank/native dispatch and separate-pointer fused kernels; exact-token final A/B regressed code/Chinese by roughly2–5% (pair_fused_ab.json). Rejected candidate source is retained for diagnosis; no production switch substitutes for an effective implementation. Bitwise quant conversion and unroll candidates had no stable end-to-end benefit. CPU profiling records verify/native wait as elapsed device work, not proven removable CPU overhead.

## Reproduction and provenance

Use the existing /Users/jay/workspace/prod/common/sglang-metal/bin/python with PYTHONPATH=python, SGLANG_USE_MLX=1 and MLX_ENABLE_TF32=0. final_ab.py loads one full model and switches explicit plans against source_baseline copies. Run only with the serving model stopped to avoid concurrent model allocations. Launch normal serving with /bin/sh benchmark/mlx/fp4_optimization_20261103/launch_optimized.sh. No dependencies were installed.

Source baselines, raw results and bounded text logs are versioned here; weights, virtual environments and binary profiler captures are excluded. source_manifest.json lists code commits and SHA256 hashes. Inherited pyproject edits and unrelated experiment files were preserved. Commits are local to the remote repository; no push was requested.
