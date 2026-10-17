# FP4-Local MTP W4A4 candidate, 2026-10-18

Base: `581b2fa446eab9b1ae341492adc1bc4aff24d281`.
Independent candidate; do not apply the old 00–12 optimization series again.
Only FP4-Local's resident NVFP4 MTP expert method changes. Target prefill,
decode/verify precision, expert-cache capacity and FP8 MTP are separate.

Local SM120 validation is now complete; see
[the GPU and full-model A/B report](mtp_w4a4_20261018/README.md).
The original duplicate-route graph failure and redundant scale allocations
have been fixed. 314 CPU tests, 66 GPU tests (no skips), 32/512-expert kernel
checks and four fresh-worker A16/A4/A4/A16 rounds passed. After the capacity
fix, an additional fresh A4 worker passed three short and two 32K requests,
four known-answer checks, and a separate 218015-token prompt plus 128-token
generation. A4 KV capacity is now 221248 tokens, matching A16 at 6390 slots.
The final 32K TTFT mean was 4.33s versus the A16 baseline's 6.11s; sustained
decode was similar. **W4A4 is now the default**, with explicit W4A16 rollback.
Simple quality smoke checks passed; full quality evaluation has not been run.

## Selection and rollback

Set the environment in the actual **model-worker container**, then restart:

```bash
export SGLANG_QWEN38_MTP_ACTIVATION=w4a4
```

For rollback, set `SGLANG_QWEN38_MTP_ACTIVATION=w4a16` and restart. An absent
variable defaults to W4A4 after the local validation above. Changing a
host shell variable does not update an already-running Docker container.
Startup prints `Qwen38 MTP expert activation: w4a4` or `w4a16`.
Do not switch inside a live worker: A4 prepares its weights at model load.

## Implementation

Checkpoint loading explicitly retains [gate, up], independent of the global
draft backend. A4 swaps to CUTLASS [up, gate] after loading, one expert at a time.
The packed weight allocations stay resident without whole-layer duplication.
Scale parameters are swizzled one expert at a time back into their original
storage; CUTLASS aliases those allocations. A4 does not retain an unused
checkpoint-layout copy. The original candidate's duplicate scale tables cost
150 MiB and reduced the KV pool; this was an implementation defect, not an
inherent A4 cost. At 512 experts the fixed preparation adds 5120 resident tensor
bytes and peaks at 1.56 MiB of extra allocation. Both modes have the same
221248-token KV capacity in the tested serving configuration.

CUTLASS accepts only one GEMM1 alpha per expert. The gate alpha is used for
both halves and the up/gate ratio is folded into the route's final weight.
This preserves the intended gate/up algebra before activation quantization;
intermediate FP4 rounding differs from a hypothetical kernel with independently
scaled gate/up epilogues. It is not A16 numerical parity or calibrated W4A4
equivalence. Activation global scales are 1.0; per-block E4M3 scaling is performed
by CUTLASS. Acceptance-rate changes must be measured on actual model weights.

The existing CUTLASS MoeRunner is selected explicitly, including under a
global Triton draft backend. Standard single-rank BF16 dispatch is required.
Invalid expert IDs receive zero route weight. Repeated routes to the same
expert are merged before CUTLASS dispatch: its reverse permutation is indexed
by (token, expert), so passing duplicates corrupts intermediate routing and
breaks graph/eager agreement. A fused Triton kernel combines route weights,
applies the up/gate ratio and masks invalid/duplicate IDs. Draft extend is
chunked to at most 128 tokens per CUTLASS call to
bound workspace; long prefill memory stability still needs GPU validation.
No custom SwiGLU clamp/alpha/beta configuration is accepted.

## Original author's validation

```bash
bash tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
```

292 CPU tests passed, including 5 new scale/layout/dispatch/rollback tests.
Actual SGLang imports and ServerArgs/runner projection checks passed.
Image: `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`;
PyTorch `2.13.0+cu130`, FlashInfer `0.6.18`. Dependencies were not upgraded.
At delivery, GPU checks, real payload loading, CUDA graph replay, full-model
quality and macro benchmarks were **not_run**. The subsequent local validation
is recorded in the report linked above.

## User GPU check and macro benchmark

After applying, on SM120 in the existing serving image:

```bash
PYTHONPATH=python python3 tools/qwen38_cache_tests/mtp_w4a4_gpu_check.py \
  --experts 32 --output /absolute/results/mtp-a4-kernel.json
```

The tool fails without a GPU. It checks finite outputs and small-input graph
replay, reports A4 deviation from A16 and layer latency at 1/3/128/257 tokens.
`--experts 512` uses full resident expert geometry and extra reference copies;
run it on an idle GPU with sufficient free memory. This diagnostic is not a
quality certificate or a full-model speed result.

Use the existing service benchmark after launching each mode separately:

```bash
python3 tools/qwen38_cache_tests/bench_mtp_service.py \
  --url http://127.0.0.1:8000 --stream --tokens 1024 \
  --output /absolute/results/mtp-a16-short.json
# Restart with MTP_ACTIVATION=w4a4; run the identical command with a4 output.
```

Repeat with `--context-repeat 16000` for roughly 32K synthetic context; record
actual prompt token count rather than assuming the tokenizer's length. Keep
target precision, MTP steps, KV dtype, chunk size, slot capacity, generation
parameters and prefix/cache warmup protocol identical. Alternate fresh runs
A16/A4/A4/A16, include warmups and at least three measured runs per mode,
and keep profiler off. Save startup log, launch environment, source/image
hash, server_info, TTFT, decode/e2e throughput, acceptance length and driver
peak/free memory. Do not infer that a kernel speedup is an end-to-end gain.

Use the same prompts with greedy decoding to compare outputs or task pass
results; sampled outputs need a distribution-level check, not same-seed text
equality. Verify/rejection sampling and state rollback must remain correct.
The default was promoted after local macro/smoke/capacity validation. If acceptance falls or
latency/memory regresses, use the explicit W4A16 rollback.
