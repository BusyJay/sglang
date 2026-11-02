# NVIDIA GLM FP4 with DFlash2 on M5 Ultra

Target revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5,
draft incoai/GLM-5.3-Flash-DFlash2 revision
bf582e4eacc1810f76656d1811693ff6c6737d2a. Apple M5 Ultra 256 GiB,
MLX 0.32.4, mlx-lm 0.32.2, TF32 disabled, native MTP disabled.
Source baseline e417d5c78fe8c623264216e950723d8fea96427e.
Target and original BF16 draft share the resident embedding/head. No checkpoint
requantization, downloads, installs, or system wired-memory adjustment.

The existing transactional DFlash2 implementation works with this FP4 target.
One startup bug is fixed: greedy-head warmup previously used the packed
uint8 weight dtype. Warmup now derives floating activation dtype from the
embedding and final norm. A regression test exercises a real packed MXFP8 head.
This does not change inference equations or sampled distributions.

## Short-context model experiment

widths.json: three natural prompts (English explanation, Python code, Chinese
explanation), 128 output tokens, two alternating rounds, identical resident
checkpoint, 1 GiB allocator cache, 200 GiB process memory/wired limits,
eval groups decode/prefill/verify 2/4/2, command-buffer limits 200 ops/1024 MB.
The process limit accommodates target plus draft; measured residency is
171.3391 GiB (target alone 169.1578 GiB), not 200 GiB allocation.
All 30 DFlash sequences match their six AR counterparts exactly.

Mean decode throughput (127 generated steps / elapsed decode time):

| Arm | tok/s |
|---|---:|
| AR | 30.5324 |
| DFlash1 | 41.8362 |
| DFlash2 | 49.7967 |
| DFlash3 | 51.1928 |
| DFlash4 | 50.7700 |
| DFlash5 | 51.2572 |

DFlash3 repeated prompt results: 46.327 / 56.291 / 51.278 tok/s.
DFlash5 repeated: 42.831 / 59.567 / 52.604. Width five helps code, hurts the
explanation. Keep width three as a stable deployment choice; explicit CLI
block size 4 means anchor plus three proposals. Width seven in the initial
96-token pilot was much slower on the English explanation. These are small
workload samples, not a universal width optimum or evidence of 90 tok/s.

## Bottleneck and candidates

phases_topk3.json synchronizes isolated stages, destroying overlap. Warm
cycles show draft about 4.3 ms, target verification about 58 ms, context consume
about 0.6 ms with occasional boundary costs. Target verification dominates.
Replay head/sort/selector timings are nested diagnostics and must not be added
again to whole-draft cost.

- Verification groups 1/2/4/8: groups.json; 8 regresses, 1/2/4 have small drift,
  no reliable improvement. Keep 2.
- Shared gate/up dynamic FP8 activation quantization: ffn_ab.json ABBA;
  50.4509 baseline vs 50.6475 candidate tok/s; steady 50.8251 vs 50.9391.
  All tokens/counters identical. No stable whole-model gain; not adopted.
- Sort NVFP4 expert picks and invert output order: sort_ab.json ABBA;
  50.4366 vs 50.5210 tok/s; steady 50.8407 vs 50.8539. Tokens/counters exact.
  Sorting cost offsets locality benefit at this small verification width;
  not adopted. Diagnostic scripts remain outside production.

Actual shapes and growing contexts remain runtime concerns. This phase does
not claim full-model ICB/MPSGraph replay or measured neural-accelerator
utilization. Existing FP4 quantization and reductions are preserved.

## HTTP validation

http_dflash_fixed.json passes startup, cold/warm prefix reuse at 32 and 2050
input tokens, raw logprobs and top-3 logprobs, stop token, repeated seeded
sampling, two concurrent requests, abort and post-abort prefix/slot reuse.
Cold 2050-token prefill took 15.36 s; prefix hit retained 2048 tokens and whole
32-token request took 0.719 s. This is lifecycle qualification, not a long
steady decode benchmark.

Six short natural 128-token streaming requests: DFlash decode 47.73–53.25
 tok/s, repeated end-to-end 45.53 / 49.33 / 46.46 tok/s. First end-to-end
32.31 / 45.42 / 42.34 includes first-shape costs. Concurrent aggregate
50.97 tok/s. Server startup 39.34 s.

Same HTTP script/environment AR control: stream decode 29.59–29.68 tok/s;
repeated end-to-end 28.68 / 28.52 / 28.72. Startup 38.27 s. These HTTP arms
are sequential separate processes, not alternating model-resident ABBA;
report drift and first request costs separately. Final-source DFlash HTTP
requalification and source hashes are recorded alongside the results.

System swap was already nonzero: DFlash HTTP 1394.31→1390.44 MiB. The model
comparison widths run and other raw JSON retain their before/after values;
system-wide values cannot attribute paging to one process.

Narrow native-like NVFP4 QMV candidate: narrow_ab.json. Repeated candidate
46.23/55.66/50.76 and 45.58/55.58/50.73 tok/s; baseline final
44.23/56.48/51.51. Generated sequences diverge at token 23/95/78; accept
counters change and code/Chinese throughput regresses. This candidate is not
adopted. narrow_numeric.json localizes a real FP32 K4096 projection discrepancy
(max absolute 3.05e-5, RMS 6.59e-6 in small synthetic banks). Native batch and
singleton outputs are exact in all six cases; most BF16 tests are exact.
Further compiler/reduction and original calibration boundary work is required
before a narrow kernel can replace the native operator. This experiment is
not evidence of a Metal platform restriction, nor permission to relax quality
checks. Original native kernels remain the qualified production mechanism.

Validation: packed-head regression fails with the original uint8 warmup and
passes with the fix; runner suite 9 tests, DFlash2 suite 5, DFlash serving 9,
GLM suite 65, FP8 rows suite 5 pass on device. Test source Ruff passes;
model_runner.py has three pre-existing Ruff findings (I001/SIM103/TRY004).
Whole-file formatting was trimmed to preserve unrelated source. git diff
--check passes. No pre-commit/Black installation.

Final source HTTP rerun: startup 38.31 s; repeated stream decode
48.344 / 53.066 / 49.358 tok/s; repeated whole-request
45.541 / 49.415 / 46.521 tok/s. All lifecycle assertions pass again.
The final generation behavior is unchanged by the extracted startup helper.
Default deployment uses launch_dflash.sh on 127.0.0.1:30007, reasoning parser
glm45, max two requests, context 8192, 0.78 static memory fraction, 1 GiB
allocator cache, trained BF16 draft and three proposals. It does not enable
native MTP simultaneously. Scripts use the remote persistent interpreter.

Commits contain only the startup fix/regression and a separate reproducible
qualification report, launch/benchmark scripts and raw results. Inherited
python/pyproject.toml edits and python/pyproject_other.toml deletion are excluded.
No push. Large weights, virtual environments and captures are excluded.
