> Historical review supplied with the original patch. For the final M5 decision and defaults, see [README.md](README.md). Original result files are preserved here with the `source_` prefix; the tarball retains the original harness and bundle builder.

# GLM MLX MTP throughput review — 2026-10-31

Baseline checkout: `a7a39c8b3d` (full hash in results.json), branch `metal`.
The user's pulled commits were preserved. New candidates are uncommitted changes relative to this HEAD.

Project policy and commit conventions are now recorded in root `AGENTS.md`.
See `MECHANISM_REVIEW.md` for the follow-up audit: current software guards are
implementation coverage, not established Metal capability boundaries. Batched
attention projections and per-position KDA state outputs remain unfinished
mechanisms. The FFN candidate passes its scope explicitly without mutating
module flags.

## Findings

**The existing ~32 tok/s result is consistent with the committed M5 evidence.**
`../results/glm5_next_mtp_http_20261031.json` reports fixed step=1 at 32.18 tok/s,
versus ordinary decode 27.57; fixed step=2 falls to 28.74. The corresponding diagnosis
has 65 verification cycles and 61 correct draft tokens at step=1 (conditional first-draft
correctness 61/65), with verification occupying about 89–92% of synchronized decode time.
These are existing target-machine results, not measurements newly reproduced here.

The implementation verifies N+1 rows **serially within every layer**. Attention,
FFN and lm_head are invoked for each row. Correct draft reuse avoids subsequent
serving forwards but does not get the full benefit of batched projection work.
Existing output/cache contract panels and this review's local regressions do not
show a cache publication or rejection bug that explains the reported throughput.

A second source-confirmed issue: MLA's latent decode path calls `mx.dequantize`
on kv_b_proj every invocation when quantized. The following matmul can additionally
promote FP16 weights to FP32. The new local call counter confirms repeated dequantize
construction; this does not establish how much target-machine GPU time it costs.

## Implemented candidates

`SGLANG_MLX_GLM_MTP_VERIFY_MODE=serial|head|ffn`, default `serial`:

- serial keeps the existing per-row target arithmetic.
- head batches only the target vocabulary projection, retaining per-row final norm.
- ffn batches stateless FFN work and the vocabulary projection. Attention still
  processes rows individually and saves each position's KDA/MLA/indexer state.
  This retains exact consumed-prefix transaction boundaries without replay.

The FFN candidate now retains **native fused paths for 1..17 rows** of one sequence:

- sigmoid router and corrected top-k dispatch rows independently in Metal;
- expert down projection, FP32 weighted sum and shared expert addition use per-row
  offsets and one multirow dispatch;
- compiled routed gate/up/activation supplies the fused down input;
- dense/shared FFNs accept bounded multirow compilation inside verification;
- FFN mHC collapse/Sinkhorn/RMSNorm uses one threadgroup per row.

This is a small MTP batch contract, not cross-request batching. Existing dtype,
bit-width, alignment and grouping guards still apply. Multirow FFN enabling is
scoped to the verification call, so ordinary prefill remains on its prior path.
Attention-side mHC and KDA remain singleton in this candidate. KDA's existing
1..8-row kernel does not expose every recurrent-state snapshot; simply passing a
whole chain would break rejection publication. Extending attention projection
batching plus state snapshots is further work, not claimed complete here.

`SGLANG_MLX_GLM_CACHE_LATENT_PROJECTION=1`, default false:
retain dequantized kv_b_proj at the matmul's promoted dtype. Array identity and
quantization specification changes invalidate it. Cache is initialized on first
use, avoiding unconditional startup expansion. Memory cost is the sum over
quantized MLA layers (including MTP) of
`num_attention_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * dtype_bytes`.
This is model-weight storage, distinct from the request KV/prefix cache. No
quantization scheme or checkpoint values are rewritten.

## M4 Pro evidence and numerical limits

Persistent `.venv/mlx-dtype-audit`: MLX 0.32.4, mlx-lm 0.32.2, Python 3.14.
48 GiB machine; experiments use a real-source four-layer model with three dense
MLPs, one sparse MLA/MoE layer, MTP head, BF16 activations/projections, FP16 affine
metadata, routed/dense gate/up 4bit, down 5bit, shared 6bit. mHC weights are nonzero
random values. Synthetic routing uses 32 experts/top-k 8. Peak MLX allocation in
extended_results.json is about 174 MiB; swap remained zero. No packages installed,
full weights downloaded or wired-memory parameters changed.

`registered_results.json`: 13 MLX-only registered test bodies passed, including
old singleton numerical tests, multirow kernels, both snapshot commit paths,
serving state publication, ordinary-to-MTP resume and weight-cache invalidation.
The local harness bypasses Torch-dependent package/fixture bootstrap; actual
model/kernel modules are used. This is **not** a run of the full Torch reference
suite or HTTP server on this Mac.

For 2/3/5/9/17 rows, FP16/BF16 multirow router, down/combine (4/5/6/8bit) and mHC
kernel results are bitwise equal to concatenated independent singleton fused
kernel calls. A real three-row verifier recorded 16 HC, one router and one
fused-down calls (fusion_hits in extended_results.json).

The full tiny verifier is bitwise equal in the tested two/three-row chains.
At 5/9/17 rows it differs: max logit differences 0.01074/0.015625/0.0234375,
relative L2 about 0.00285/0.00542/0.00941. Later caches differ as FFN changes
propagate into later attention. This demonstrates why the candidate is opt-in;
**kernel parity does not establish whole-model parity at batched GEMM shapes**.
Near-tied greedy choices, sampling and logprobs may change on the real model,
including two-row shapes. The target A/B records these changes rather than
silently assuming equivalence.

Warm, alternating-order samples are saved as raw millisecond arrays. The final
extended shape panel measured serial/ffn medians 2.13/1.95 ms (2 rows), 3.17/2.86 ms
(3 rows), 5.56/4.32 ms (5 rows), 9.73/7.24 ms (9 rows), 16.56/11.49 ms (17 rows).
Earlier panels varied and two rows sometimes regressed. These bounded M4 results
are opportunities for target testing, not a M5 throughput prediction.

The latent cache changed four calls from four dequantize invocations to one;
max output difference ~8.94e-8, relative L2 ~2.98e-7. Updated metadata invalidates
correctly. The small-module timing was effectively unchanged (~0.39 ms), so
there is no local evidence of a material end-to-end gain from this candidate.
Retaining a contiguous promoted weight can change matmul layout/reduction behavior;
this candidate is also not promised bitwise exact.

Metal capture succeeded using installed Xcode/MetalToolchain. The local
`traces/verify_ffn_final.gputrace` contains two three-row verifier calls (~420 MiB).
The DOT export is largely materialized due to the model's async eval boundaries;
it is not sufficient for dispatch attribution. Captures are intentionally excluded
from the portable bundle. Kernel row-axis dispatch and fusion-hit counters prove
multirow eligibility; per-kernel duration and bandwidth on M5 remain to be measured.

## Open-source comparison and remaining work

The already present oMLX source clone is pinned to
`a4048eef24b41d885d593742e32a290f3bdb2300`. Its
`omlx/patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py` compiles bounded verification
FFNs and uses KDA input capture plus post-hoc rollback. It clones speculative
MTP head state and caps chain depth to protect pooled indexer rollback. This is
source inspection only. None of these later-main changes is attributed to rc1.
The public 41–44 tok/s result uses a different checkpoint; it is not a same-weight
baseline for this implementation.

Next priorities after same-weight A/B:

1. Batch KDA/MLA input projections while keeping sequential recurrence and exact
   per-position snapshots; quantify saved weight reads before broader kernel work.
2. Share expert weight work across repeated expert selections in a short window.
   Current multirow down launch isolates rows but does not explicitly stage common
   expert weights across rows. Top-k overlap controls the possible benefit.
3. Profile layer/phase durations and shape dispatch on M5, especially BF16 KDA,
   vocabulary head and FP32 MLA association. No evidence justifies arbitrary
   quantization changes.
4. Improve adaptive budget exploration only after verification economics change.
   The current fixed-one setting beats adaptive in the committed panel; larger
   windows currently spend much more on rejected work.

## Portable A/B

Apply `candidate.patch` on the baseline checkout with `git apply --check` then
`git apply`. The archive includes new benchmark/harness files and evidence;
no weights, dependencies or captures. Reuse the target's existing environment.

Run a model-level comparison loading the same local snapshot only once:

```bash
python benchmark/mlx/bench_glm5_next_mtp_verify_ab.py \
  --model-dir "$GLM_MODEL_DIR" \
  --output benchmark/mlx/results/mtp_verify_ab.json \
  --rows 2 3 --repeats 8 --output-length 128
```

It compares serial/head/ffn/ffn+latent cache, alternates orders, records raw
samples, cache errors, greedy/logit differences and fixed-one generation token
hashes/acceptance. The synthetic forced chains are cost controls; generation
uses a chat prompt. Expand prompt/context coverage before adopting a candidate.

For HTTP throughput and serving contracts, run baseline, FFN, then baseline
again in separate processes (one server at a time):

```bash
SGLANG_MLX_GLM_MTP_VERIFY_MODE=serial \
python benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$GLM_MODEL_DIR" --steps 0 1 2 --output-length 128 --repeats 2 \
  --check-contracts --output benchmark/mlx/results/mtp_http_serial.json

SGLANG_MLX_GLM_MTP_VERIFY_MODE=ffn \
python benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$GLM_MODEL_DIR" --steps 0 1 2 --output-length 128 --repeats 2 \
  --check-contracts --output benchmark/mlx/results/mtp_http_ffn.json
```

Repeat the first command with a new output filename. Then test the independent
latent-cache flag. Compare full-model output hashes and logprobs to serial across
files, not just within each mode's own HTTP panel. HTTP contracts can fail when
batch arithmetic changes results; do not relax them solely to report speed.

No M5 measurements of this new patch exist yet. Default serial verification and
disabled weight cache remain the recommended production settings until target
cost, memory and numerical behavior are validated.
