# Small-verify W4A16 split-K candidate

Base: `08f8bf8512` (current HEAD when this work started). Prefill remains W4A4;
main decode/verify and MTP draft remain W4A16. This change targets the main
cached MoE, not the resident draft model.

NVIDIA's [FP4-Local model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
explicitly sets `VLLM_B12X_MOE_FP4_FORCE_A16=1`,
`B12X_W4A16_A4_PREFILL=1`, and an MTP speculative configuration in its
Blackwell usage examples. Those examples use B12X/vLLM kernels; they do not
establish performance equivalence to this SGLang Triton implementation.

## Implementation

- Split the reduction dimension across independent CTAs; reduce FP32 partials
  before SwiGLU and intermediate BF16 rounding.
- Fetch each packed weight byte once and interleave the two E2M1 nibbles.
- Fetch one E4M3 scale per 16 weights and broadcast it in registers.
- Use 16 output rows per CTA instead of the baseline's 32.
- Fuse down-projection split reduction, router weighting and top-k combine.
  No atomics, device sorting, host route reads or whole-expert expansion.
- Retain the CUTLASS cache layout, global/input scale cancellation and
  `[up, gate]` ordering. Reader/fill event scopes are unchanged.

The candidate is limited to 2–3 input tokens and at most 32 routes. Other
shapes retain the baseline. Scratch for 30 routes and split-K=4 is approximately
1.80 MiB (two FP32 partial planes and the BF16 intermediate), versus 0.33 MiB
for the baseline. These are invocation-local buffers; no new resident weights.

## Validation and activation

SM120 offline compilation passed for 20/30 routes and split-K=1/2/4/8, both
projections and both epilogues (32 compile cases), using Triton 3.8.0.
Python syntax and whitespace checks passed. This host has no CUDA GPU or torch;
GPU numerical checks and speedup measurements have **not** run. Defaults remain
on the baseline until the GPU comparison identifies a winning configuration.
Split-K changes floating-point reduction order and must pass numerical checks.

Run from the repository root in the existing SM120 SGLang environment:

```bash
python3 tools/qwen38_cache_tests/w4a16_verify_compile_check.py
python3 tools/qwen38_cache_tests/w4a16_verify_gpu_check.py \
  --iterations 100 --output /tmp/w4a16-verify-splitk.json
```

The GPU script tests an independent dense BF16-weight reference, nonuniform
scales, nonunit input scales, disjoint/overlapping/duplicate/invalid expert
routes, graph replay after route/input mutation, and 1/4-token fallback.
Each candidate is timed with the same inputs as the baseline. Synthetic resident
latency excludes cache planning, expert fills and service scheduling; it must
be followed by same-workload service measurement. Compile the script with the
deployment's Triton version as well, rather than relying solely on 3.8.0.

Choose the split with a consistent gain for both 2 and 3 tokens and all routing
patterns, then set it in the actual service environment, for example:

```bash
export SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=4
```

This is an example setting, not a measured winner. `0` restores the baseline;
`1` tests the compact packed/scale loading and fused epilogue with no K split.
Do not modify the archived deployment scripts in this directory to deploy.

Cross-token weight tile reuse and a small-M Tensor Core kernel remain separate
candidates. This patch does not claim either: split-K addresses the current
serial reduction and memory-access layout without expert-grouping overhead.
