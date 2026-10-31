# GLM5-Next MLX decode fusions

The `affine_fused` default builds on `affine_all`: it keeps BF16 residuals,
local FP16 affine FFNs and compiled decode, then enables mHC, KDA and MoE
kernels. `affine_all` remains available as the pre-fusion reference.

## Implementation

- mHC fuses the four-stream Sinkhorn normalization, collapse and branch
  RMSNorm. Its FP32 mix projection and residual matrix product still use
  MLX, preserving the hardware matmul's reduction and rounding.
- KDA fuses short convolution, SiLU, Q/K normalization, vector-gated delta
  recurrence and gated RMSNorm for one sequence and 1-8 tokens. It retains
  separate convolution caches and an FP32 recurrent state. Dense or quantized
  gate projections are computed before the fused body.
- MoE fuses sigmoid router logits and corrected top-k selection, then the
  affine expert down projection, FP32 routing-weighted reduction and shared
  expert addition. Down projections support 4/5/6/8 bits with matching affine
  metadata. In particular, this checkpoint's five-bit down projections engage.
- Shape, dtype and quantization guards retain the existing MLX paths outside
  the supported contracts. Weight arrays remain explicit compiled inputs;
  weight replacement does not retain stale kernel constants.

The mHC kernel preserves separately rounded multiply/add operations and the
softmax reciprocal/product boundary. Contracting the mix into an FMA caused
real-model divergence. A cancelling-stream regression test reproduces that
failure (nonzero branch output instead of zero) on the contracted expression.
RMSNorm also rounds normalization before multiplying its learned weight.

The KDA and MoE Metal arithmetic is adapted from oMLX `decode_kernels.py` at
`5c6cbd6314cadba33f0db3f90f27dce6620831b4` (Apache-2.0), with the SGLang/MLX
normalization, cache layout and FFN dtype restoration boundaries preserved.
The callable APIs live in the attention, layernorm and moe operator groups,
with lazy MLX/MPS registry metadata.

## Full-model measurements

Apple M5 Ultra / 256 GiB, MLX 0.32.4, mlx-lm 0.32.2, Python 3.12.16.
Existing local `mlx-community/GLM-5.3-Flash-4bit` revision:
`fa04cbc3622b16aa7475404060b05e506d81bd4e`.
No requantization, model download, dependency install or OS limit change.

Same loaded weights, 16-token prefill, 32 forced decode tokens, two opposite
profile orders, process wired limit matching the existing runner (240 GiB).
Each rate is tokens divided by summed decode latency, excluding loading.

| Profile | Round 1 tok/s | Round 2 tok/s |
| --- | ---: | ---: |
| `affine_all`, reference | 20.45 | 20.54 |
| `affine_hc` | 27.06 | 26.84 |
| `affine_kda` | 20.59 | 20.55 |
| `affine_moe` | 21.17 | 21.22 |
| `affine_fused`, default | 29.11 | 28.99 |

The combined path improves decode by about 42%. KDA alone is within whole-model
noise; its warm production-geometry layer benchmark improves by about 4%, so it
is retained. Warm kernel latency improvements: mHC 1.94x, router 1.22x and
expert down/combine 1.12x. These microbenchmarks keep weights resident/hot and
make no DRAM-bandwidth claim.

All profiles' full-vocabulary logits are bitwise equal to `affine_all` for
both 32-position panels: KL=0, top-1 agreement=100%. An instrumented real
forward confirms 34 KDA and 90 mHC fused calls, with all 90 mHC outputs equal
to their reference. This is a bounded parity check, not a general task-quality
or long-context guarantee.

## Serving and checks

Serial uncached HTTP streaming, 16 input / 32 output tokens:
27.91, 28.10 and 27.73 decode tok/s, versus the prior 19.94 tok/s.
Warm TTFT is about 111 ms. Only the existing `SGLANG_USE_MLX=1` backend gate
was set: no optimization-profile or TF32 override was needed.

The chunked-prefix smoke (40 input / 16 output, chunk size 32) hits 32 tokens
on repeated requests and 40 on suffix extension. All four token/logprob panels
are identical to the previously recorded `affine_all` HTTP results. Existing
cold/hot prefill segmentation differences are unchanged.

Validation: 148 tests pass, with 55 subtests. The unchanged AMX lazy-import
probe for `srt/utils/common.py` fails in its fresh subprocess because this
Apple environment has no Triton; no dependency was installed to mask it.
Kernel/runtime parity, cache/offset checks and registry coverage pass.
Ruff lint/format, isort, CI registration parsing and whitespace checks pass.
All new code lines are at most 88 characters.

The manual warm-latency benchmark is
`test/manual/kernels/bench_glm_mlx_fusions.py`; it accepts `--output PATH`.
The independent profiles permit the same single-feature ablation on a local
checkpoint. Detailed local timing and raw panels are retained under
`benchmark/mlx/engine_comparison_20261030/`, particularly
`fusion_profiles_native_residual.json`, `fusion_micro_final.json`,
`fusion_server_stream.json` and `fusion_server_prefix.json`.
