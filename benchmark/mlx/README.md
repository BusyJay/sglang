# GLM-5.3-Flash MLX baseline and roadmap

This directory records the initial text-only GLM5-Next port, a small reproducible
performance baseline, and the remaining optimization work. The baseline is a
starting point for comparisons, not a production capacity or model-quality claim.

## Reproduce the baseline

The initial run used Apple M5 Ultra with 256 GiB unified memory and the cached
`mlx-community/GLM-5.3-Flash-4bit` checkpoint, revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e` (about 190 GiB of checkpoint tensors,
including the unused vision and MTP weights). Framework versions and Metal device
properties are recorded in the result JSON.

Start the editable SGLang checkout in its MLX environment:

```bash
env SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 sglang serve \
  --model-path mlx-community/GLM-5.3-Flash-4bit \
  --revision fa04cbc3622b16aa7475404060b05e506d81bd4e \
  --cuda-graph-backend-decode=disabled \
  --cuda-graph-backend-prefill=disabled \
  --context-length 4096 \
  --max-total-tokens 4096 \
  --mlx-enable-sampling \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 127.0.0.1 --port 30000
```

The model-specific defaults disable radix prefix caching and overlap scheduling,
set concurrency to one unless explicitly configured, and cap prefill chunks at
256 tokens. The initial implementation requires native per-request caches.
`MLX_ENABLE_TF32=0` selects full FP32 precision for the reference baseline; it
does not disable BF16 or quantized execution globally. See
[MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

After server warmup, run from the same environment:

```bash
python benchmark/mlx/bench_glm5_next.py \
  --input-lengths 32 256 1024 --output-length 32 --repeats 2 \
  --server-mlx-enable-tf32 0 \
  --output benchmark/mlx/results/glm53_flash_m5_ultra_2026-10-29.json
```

The benchmark sends exact-length raw token inputs from a repeated English passage
to `/generate`. Requests are serial, greedy, and forced to generate the requested
number of tokens with `ignore_eos=True`. One 32-input/8-output warmup is excluded
from the summaries. No chat template or reasoning parser is timed on this route.

- TTFT is client wall time to the first SSE event containing output tokens; it
  includes HTTP, scheduling, prefill and first-token sampling. The JSON records
  the first event's token count to expose any streaming coalescing.
- Decode rate is the number of tokens after the first event divided by their
  elapsed time. TPOT is its reciprocal in milliseconds.
- End-to-end latency includes all output and HTTP completion.
- The script checks exact prompt/output lengths and zero prefix-cache hits.
- Two trials per input length provide a smoke baseline only. They do not measure
  p95 latency, saturation throughput, full-length context behavior, or quality.

## Initial results (2026-10-29)

Raw trials and medians are retained in
[the result JSON](results/glm53_flash_m5_ultra_2026-10-29.json).
All rows generate 32 output tokens and report the median of two serial trials.

| Input tokens | TTFT (s) | Decode (tokens/s) | TPOT (ms) | End-to-end (s) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.745 | 1.840 | 543.6 | 17.598 |
| 256 | 1.774 | 1.775 | 563.5 | 19.242 |
| 1024 | 6.014 | 1.505 | 664.4 | 26.611 |

The observed decode rate falls as context grows. Repeated historical KV expansion
and cache/indexer work are plausible contributors from code inspection, but no
profile has yet attributed this difference. The 1024-token case exercises four
prefill chunks; these measurements do not enter the checkpoint's >2048-token
sparse-selection regime. Startup weight loading was approximately 29-30 seconds
in the two successful server launches, separate from these warmed request timings.

## Preserved behavior and current limitations

The port preserves KDA's bounded forget gate, convolution and recurrent state;
DSA k-pool selection and causal tails; mHC stream mixing and Sinkhorn iterations;
the router, shared/routed experts and clamped SwiGLU; and the checkpoint's mixed
4/5/6-bit quantization. It does not reduce the layer count or experts per token.

Current limits and costs:

- Text input only. Vision weights and the extra MTP decoder are skipped.
- Only the NoPE, independent-indexer GLM5-Next configuration is supported.
- Prefix caching and CPU/GPU overlap are disabled. Multiple requests, if enabled,
  execute separately rather than through a native batched decode kernel.
- Every layer evaluates and waits for its output and cache arrays. This bounds
  the graph and temporary expanded KV memory but introduces frequent waits.
- MLA stores compressed latents but expands the entire cached history for each
  forward pass. It does not yet use absorbed/latent-space decode attention.
- DSA selection becomes a mask for ordinary SDPA; no sparse gather attention
  kernel skips the unselected KV computation.
- The indexer retains raw key/gate history and rebuilds pooled candidates.
- mHC and the separate projections, convolutions, gates and activations are not
  fused for this architecture. The custom clamped expert path also bypasses
  mlx-lm's token-sorting optimization for large MoE batches.
- The automatic logical token budget is capped at 32768; it is not derived from
  native-cache bytes and request-state memory. This baseline explicitly uses 4096.
- Allocation reuse is conservatively capped at 1 GiB by default on this loader
  path; `SGLANG_MLX_CACHE_LIMIT_GB` can explicitly override it. Input weight
  dictionaries are consumed while stacking experts to avoid a second resident
  checkpoint at load time.
- Full-weight validation currently covers successful HTTP generation and health
  checks. The small numerical tests cover reference parity, sparse selection,
  cached continuation, mixed-bit loading and request isolation. They are not a
  full-model quality or long-context evaluation.

## Optimization roadmap

Priorities below are hypotheses from code inspection until traces establish the
actual contribution of each operation. Keep this baseline configuration available
for before/after comparisons; do not combine optimizations into one measurement.

### P0: establish attribution and quality guardrails

- [ ] Capture Metal/GPU and CPU submission timelines for prefill and decode.
      Attribute time to MLA expansion, attention, KDA, MoE, mHC and host waits;
      record GPU active/cache memory as well as system memory headroom.
- [ ] Determine which operations actually use M5 GPU Neural Accelerators and
      whether the workload is limited by submission, bandwidth or computation.
      MLX supports these accelerators, but the current port has no kernel-level
      utilization measurements. See [Apple's MLX/M5 discussion](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- [ ] Add full-weight accuracy/logprob and realistic chat checks, including
      prefill/decode continuation and inputs crossing the DSA top-k budget.

### P1: reduce avoidable work and synchronization

- [ ] Replace per-layer blocking evaluation with bounded groups or asynchronous
      submission. Preserve load/inference memory bounds and cache side effects.
- [ ] Implement absorbed MLA decode or equivalent latent-space attention so
      historical K/V expansion is avoided. Validate numerical changes separately
      for BF16 and mixed-bit weights.
- [ ] Implement true selected-KV DSA execution with correct complete-pool and
      incomplete-tail behavior. Keep the existing dense masked path as an oracle.
- [ ] Update pooled indexer state incrementally instead of rebuilding old pools
      and copying all key/gate history on every decode step.

### P2: improve matrix utilization, fusion and serving throughput

- [ ] Fuse compatible KDA projections, short convolutions, gates and gated norm;
      retain the existing Metal recurrence kernel as a comparison point.
- [ ] Compile/fuse mHC normalization, projection, Sinkhorn and residual mixing.
- [ ] Restore expert token sorting/grouping for prefill; evaluate fused gate/up
      projections and an M5-aware mixed-bit expert path without changing routing
      or silently requantizing the checkpoint.
- [ ] Evaluate the FP32 reduced-precision path separately, preferably with local
      precision control where supported. Compare quality and routing stability;
      do not infer its benefit from peak hardware specifications.
- [ ] Add genuinely batched native-cache decode and CPU/GPU overlap.
- [ ] Integrate prefix reuse with both compressed attention and KDA snapshots,
      including exact-prefix hits, eviction, retraction and chunk boundaries.
- [ ] Size the scheduling budget from actual native-cache and recurrent-state
      memory; tune chunk size and allocation-cache reuse with measured peaks.

### P3: expand feature coverage

- [ ] Integrate MTP speculative decoding and verify cache rollback semantics.
- [ ] Add image/video support and the vision tower.
- [ ] Validate and tune longer contexts, then raise the tested serving budget.

Unified memory, packed quantized weights, fast RMSNorm/SDPA and the existing Metal
KDA recurrence are already used. There is no Core ML/ANE execution path or
Ultra-specific topology tuning. Neither fact establishes that such a path would
improve this workload; measure before choosing another backend or partitioning.
