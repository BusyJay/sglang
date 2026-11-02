# GLM-5.3-Flash NA optimization — 2026-11-02

## Result and scope

Implementation commit: `fb00133b229b03179c8148a3a89fc041a446f801`.

Base: `2a1e5aada21f139356d263ee4a0d58d2aef0784e`, branch metal, remote BlazesMacStudio:/Users/jay/workspace/sglang.
Apple M5 Ultra, 256 GB, macOS 27.2, MLX 0.32.4; TF32=0.
Target mlx-community/GLM-5.3-Flash-4bit revision fa04cbc3622b16aa7475404060b05e506d81bd4e, original mixed 4/5/6bit checkpoint.
Draft incoai/GLM-5.3-Flash-DFlash2 revision bf582e4eacc1810f76656d1811693ff6c6737d2a.
Interpreter /Users/jay/workspace/prod/common/sglang-metal/bin/python.
All implementation and experiments occurred remotely through tmux %0. User pyproject modifications and earlier experiments are excluded from commits. No dependencies installed or system wired-memory parameters changed.

The adopted mechanism is a default M5 BF16 vocabulary-head NA path for 8–16 rows, N154880/K4096. It uses MPP tensor views to read checkpoint weights, 16 K SIMD groups in one threadgroup, FP32 accumulation and local rounded top1. It writes only tile winners, retaining first-index ties and first-NaN behavior. Non-unit column strides load cooperative registers directly; weights are never materialized into a full contiguous copy. The public registered API remains gemm.mlx_dense_argmax_rows. The helper is private.

This accelerates DFlash block size 8 (7 proposed tokens) verification and similar covered batches. The existing block size 4 remains unchanged. Ordinary AR and short 1–4 row verification have no demonstrated NA speedup. This is a completed head mechanism, not completion of all GLM NA opportunities.

## Full-model results

Final production source, identical resident target/draft, 2050-token context, 128 output tokens, three prompts, two ABBA groups per prompt. Count consumed tokens after prefill / sum per-token decode seconds. First 16 decode events are separately excluded for steady statistics. Raw order and times are in final_generation_ab.json.

| Prompt | Baseline tok/s | Candidate tok/s | Change | Baseline steady | Candidate steady |
|---|---:|---:|---:|---:|---:|
| Sky explanation | 51.078 | 51.239 | +0.31% | 49.142 | 49.391 |
| Fibonacci | 78.053 | 79.280 | +1.57% | 75.415 | 76.564 |
| Chinese overfitting | 61.679 | 62.569 | +1.44% | 60.612 | 61.518 |

All output hashes and draft acceptance/rejection counts match within each prompt across arms. Peak MLX memory: A204136434060 / B204136215798 bytes, effectively equal. Swap stayed 1333.50 MiB. The engine uses the normal recommended process wired limit and 1 GiB allocation cache consistently. No claim of statistical significance beyond these bounded ABBA samples.

Prior prototype generation: two contexts, three prompts, 96 tokens, two ABBA groups; same output hashes and acceptance counts. These are exploratory results, not replacement for final-source measurements. Full verification of the fused prototype at 8 rows: short .057400→.056889 s, 2050 .067764→.067108 s, tokens and internal states equal. Internal state equations are unchanged by the adopted head path.

## HTTP and lifecycle

http_validation.json and http_baseline.json: same real checkpoint, DFlash block size8, chunk256/context8192/pool17024/max-running2, deterministic seed1234, 17 sequential cases each plus two concurrent requests and SSE abort. Cold/prefix tokens and logprobs, seeded sampling, stop, post-abort signatures all agree between versions. Concurrent signatures also saved. Both servers exited cleanly, no resident server remains.

Single-run HTTP decode medians (two streams per prompt) A→B: sky68.749→68.939, Fibonacci80.806→80.119, Chinese88.584→88.632 tok/s. These mixed sub-1% differences are not a demonstrated HTTP throughput improvement. They use shorter natural prompts than the final engine ABBA.

Initial concurrent single-shot 1.876→2.258 s was investigated with repeated clean services and seven identical flushed-cache batches. Warm six-trial medians: A1.871183 / B1.870516 s. First batches A3.487095 / B3.460853 s include compilation/layout costs. No stable concurrency regression observed; no claimed concurrency speedup.

Memory limitation: candidate first full HTTP service increased global swap1333.50→1578.50 MiB; the later baseline began at1578.50 and ended1577.38. Warm concurrent runs A1580.44→1580.94, B1569.38→1580.44 MiB. Global swap includes OS activity and different process-start states; these runs do not prove zero service memory regression. Engine A/B peak/active memory is separately recorded. Long context/large concurrency memory remains unvalidated.

## Why earlier NA candidates did not improve inference

* Reusing prefill BM32 paired NAX for 3–16 row quantized shared gate/up was slower than shared-QMV. Reducing to16 rows/64 threads did not solve it. The implementation still decodes broad weight tiles and pays cooperative/register work with too few useful rows; this is an implementation result, not a Metal batch limit.
* Direct quantized cooperative loaders and external split-K improved some16-row cases but did not beat QMV for common3–8 rows. Original checkpoint quantization was retained. Relative operator L2 around5e-4 came from the T-decoded-weight / NAX accumulation path vs affine QMV algebra, not proof of hardware incompatibility.
* BF16 KDA output native matrix microbenchmarks looked faster, but complete verification was slightly slower and maximum state relative L2 rose3–11% despite equal target tokens. This candidate was not adopted; recurrence/routing amplification is not fully localized, and no tolerance was relaxed.
* Continuous NAX packed-pointer and stride specialization made synthetic288-expert5/6bit gate/up18–21% faster with exact outputs. Actual checkpoint prefill audited eleven4bit NAX calls (2048 expert assignments) at256 input tokens; 5/6bit changes do not cover that path. Full256/1024 prefill stayed approximately .328/.834 s with exact states. 4bit byte/word/layout changes lacked stable gains. These candidates remain experiment scripts, not default-off production switches.
* Manual per-element dense cooperative loading was slower; native MPP tensor weight views avoided that work. Fragment and split-K experiments identified the head as the useful geometry. The adopted head local reduction avoids full-logit writes and shares weights across8–16 rows.
* Whole-model gain is smaller than the head operator gain because expert/KDA/attention and draft work dominate. No microbenchmark times were summed or converted to throughput.

Remaining mechanisms: routed down reuse under real sparse expert distributions; efficient3–4-row quantized NA; QK–softmax–PV fusion for indexed sparse attention (existing QK/PV already use NA under eligible low-precision policy); stricter localization of KDA numerical divergence. Full attention and dense prefill already have MLX NA paths; not every GLM call benefits automatically. Multi-device TP, video/vision quality, broad generation evals and32K memory matrix were not run for this head change.

## Validation and source identity

Final dense device test file:8 tests pass. New test covers8/9/16 rows, non-contiguous input/weights, tail output tiles, ties and NaNs. Existing seven tests preserve singleton and ordinary reduction contracts.
Namespace117 pass/1 fail: AMX lazy import case srt/utils/common.py requires missing Triton; unrelated to head, dependency not installed.
Registered test lint, Ruff F401/F821/F823/UP037, Ruff format, isort, git diff --check pass. No pre-commit executable or clang-format available; full pre-commit not run.
AST/import checks are not numerical evidence.

During testing a stale contiguous base in the strided path was fixed. Non-unit column-stride tensor-view reads still failed the device oracle; direct cooperative register reads resolved it while preserving NA computation and storage. These were implementation defects, not established platform limitations.

manifest.json records final source and every saved experiment SHA256/size. baseline/mlx_dense_rows.py is the exact original public operator for diagnostic A/B, never imported by production. Prototype scripts mutate bindings only in isolated experiment processes with independent graph stores; they are not serving interfaces. Candidate prototypes are retained to reproduce attribution and are removed/replaced when a better validated mechanism wins.

## Reproduction

From remote repository:
```sh
export PYTHONPATH=python MLX_ENABLE_TF32=0
PY=/Users/jay/workspace/prod/common/sglang-metal/bin/python
$PY test/registered/kernels/ops/gemm/test_mlx_dense_rows.py
$PY benchmark/mlx/na_optimization_20261102/final_generation_ab.py --tokens 128 --repeats 2 --output benchmark/mlx/na_optimization_20261102/final_generation_ab.json
$PY benchmark/mlx/na_optimization_20261102/http_validation.py --mode dflash --count 7 --tokens 64 --output benchmark/mlx/na_optimization_20261102/http_validation.json
```
Run complete-model processes sequentially. For serving the larger verified batch:
```sh
DFLASH_BLOCK_SIZE=8 bash llm-online/start.sh
```
This head change does not establish block size8 is optimal versus block size4; both models/request acceptance and throughput matter. Existing deployment defaults are preserved.

No patch archive created. Commits are remote only, not pushed. SHA256 excludes weights/environments/captures; raw scripts and JSON are within this directory.
