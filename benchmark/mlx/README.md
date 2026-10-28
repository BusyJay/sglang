# GLM-5.3-Flash MLX baseline and roadmap

This directory records the initial text-only GLM5-Next port, a small reproducible
performance baseline, and the remaining optimization work. The baseline is a
starting point for comparisons, not a production capacity or model-quality claim.

## Default serving precision

Use MLX's reduced-precision FP32 fast path for ordinary serving:

```bash
env SGLANG_USE_MLX=1 MLX_ENABLE_TF32=1 sglang serve \
  --model-path mlx-community/GLM-5.3-Flash-4bit \
  --cuda-graph-backend-decode=disabled \
  --cuda-graph-backend-prefill=disabled \
  --context-length 4096 --max-total-tokens 4096 \
  --mlx-enable-sampling \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --host 127.0.0.1 --port 30000
```

The historical baseline below used `MLX_ENABLE_TF32=0`. Its results remain
unchanged; they do not describe the default serving precision above. The small
numerical reference tests also use full FP32 precision to isolate implementation
errors from reduced-precision rounding.

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

## First MLA/DSA execution optimization

The following comparison uses `MLX_ENABLE_TF32=1` on both sides, concurrency one,
256-token prefill chunks, and **16 output tokens** per request. Each row is the
median of two serial trials. The before implementation is `0585d3180b`; the after
implementation combines absorbed MLA, selected-latent DSA and asynchronous layer
groups. These are aggregate measurements, not per-change attribution. They are
not directly comparable with the earlier TF32-disabled, 32-output-token table.

| Input tokens | TTFT before/after (s) | Decode before/after (tokens/s) | Decode change |
| ---: | ---: | ---: | ---: |
| 32 | 0.748 / 0.594 | 1.860 / 1.821 | -2.1% |
| 1024 | 6.099 / 4.701 | 1.590 / 1.748 | +10.0% |
| 3072 | 18.162 / 15.291 | 1.254 / 1.695 | +35.1% |

Raw trials: [before](results/glm53_flash_m5_ultra_tf32_before_2026-10-29.json),
[after](results/glm53_flash_m5_ultra_tf32_after_2026-10-29.json). The 3072-token
case exercises actual sparse selection beyond the 2048-token budget. Short decode
has no demonstrated improvement; two trials do not establish statistical
significance, and first use of a shape may include compilation overhead.

Reproduce either implementation with the same serving flags above and:

```bash
python benchmark/mlx/bench_glm5_next.py \
  --input-lengths 32 1024 3072 --output-length 16 --repeats 2 \
  --server-mlx-enable-tf32 1 --output comparison.json
```

Fast correctness checks cover FP32/BF16 absorbed-versus-expanded attention,
512-dimensional latents, quantized KV projection, independent Transformers
pool/tail selection, batched query-specific gathers and partial asynchronous
layer groups. For two synthetic chat prompts, the full-weight before/after runs
produced identical first eight greedy tokens; the maximum selected-token logprob
difference was 0.00324. This remains a smoke check, not a quality evaluation.

The selected-latent gather also has a measurable tradeoff. An isolated BF16 MLA
probe at production dimensions (64 heads, latent rank 512, history 3072) measured
extra active-memory peaks above resident weights/inputs:

| Query tokens | Before scratch (MiB) | After scratch (MiB) |
| ---: | ---: | ---: |
| 1 | 207.4 | 14.1 |
| 256 | 381.5 | 714.5 |

The decode scratch falls, but prefill's per-query latent gather increases scratch
and can be slower inside the attention module itself. Whole-model TTFT improves
in the aggregate measurement above; this does not establish that sparse prefill
alone got faster. Query tiling or fused indexed attention remains on the roadmap.
The probe overlapped another generation request, so its recorded latencies must
not be used as isolated speedup estimates; the table uses per-process MLX memory
accounting only. [Probe data](results/glm53_mla_scratch_2026-10-29.json).

```bash
env MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_mla.py \
  --reference-ref 0585d3180b --context-length 3072 \
  --query-lengths 1 256 --repeats 3 --output mla-scratch.json
```

## Reuse from Muse Glimmer

The local Muse Glimmer adapter performs invariant weight transformations and
projection fusion at load time. Following that approach, GLM5-Next now prepares
compatible KDA Q/K/V projections as one linear operation after checkpoint loading
and quantization. Original parameter names remain available as views of the fused
storage, avoiding a second resident copy. Different quantization bits, group
sizes or modes fall back to separate projections; replacing source parameters
invalidates the prepared projection.

The clamped expert activation now plugs into mlx-lm's existing `SwitchGLU`,
restoring its expert sorting/grouped matmul path for larger token batches.
Glimmer's ordinary attention cache adapter cannot directly represent GLM's
compressed MLA and recurrent KDA state, so those retain their native caches.

Small regression tests cover FP32/BF16 and quantized fusion,
incompatible mixed-bit fallback, parameter replacement, resident storage and
sorted MoE reference parity. The full-model timings above precede these latest
changes; their end-to-end performance benefit has not yet been measured.

## Prefill speed follow-up

Dense MLA prefill (multiple query tokens, total KV length within `index_topk`)
now expands K/V once and uses SDPA at the original head dimension. This reuses
K/V across queries and avoids the slower 512-dimensional latent SDPA path for
this case. Single-token decode and sparse attention retain compressed-latent
execution. Cached dense prefill still uses bottom-right causal alignment.

mHC now compiles normalization, projection, the original 20 Sinkhorn rounds and
residual stream mixing together. It preserves FP32 coefficient calculations and
the original activation cast points. Compilation specializes on input shape;
first-use compilation is excluded from these warmed timings.

Short module probes use random weights at production dimensions, identical
inputs/weights and three alternating-order timed pairs after one warmup pair.
`MLX_ENABLE_TF32=1`; allocator cache limit is 1 GiB. The Q4 probe uses uniform
4-bit affine projections with group size 64, rather than the checkpoint's mixed
bit widths. These are module measurements, not full-model TTFT measurements.

| Module, Q4 probe | Tokens | Before (ms) | After (ms) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Dense MLA | 256 | 3.551 | 2.984 | 16.0% |
| Dense MLA | 1024 | 9.991 | 8.621 | 13.7% |
| mHC | 256 | 2.092 | 1.789 | 14.5% |
| mHC | 1024 | 2.944 | 2.556 | 13.1% |

[Q4 probe data](results/glm53_prefill_modules_q4_2026-10-29.json) and
[BF16 probe data](results/glm53_prefill_modules_2026-10-29.json) include individual
samples, source hashes and active-memory peaks. Before these edits, save the
module source or use a reference commit; the short probe can compare either:

```bash
env MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_prefill.py \
  --reference-source before-prefill.py --query-lengths 256 1024 \
  --quantization-bits 4 --repeats 3 --output prefill-modules.json
```

Ten fast regression tests pass, including cached dense/sparse MLA with quantized
KV projections and a separate Transformers mHC oracle at four streams and twenty
Sinkhorn rounds in FP32/BF16. At that stage, no full-model restart or long generation test had been
performed; the later short serving comparison is recorded below. Sparse prefill remains a separate speed
optimization target; query tiling is deferred until its latency tradeoff is clear.

## Final-position logits and KDA block experiment

The GLM model exposes `forward_last_logits`: it runs the complete trunk and
updates every layer cache, then projects only the final hidden-state row. The
MLX serving runner uses this optional method for next-token generation. Ordinary
full-sequence model calls retain full logits, and intermediate prefill chunks
continue to skip the head entirely.

A random production-size Q4 head (4096 -> 154880, group size 64, BF16 input)
measured the following warmed medians over three alternating pairs:

| Prompt positions | Full projection then slice (ms) | Final row only (ms) | Extra active peak before/after (MiB) |
| ---: | ---: | ---: | ---: |
| 256 | 3.469 | 0.501 | 75.625 / 0.297 |
| 1024 | 12.015 | 0.480 | 302.500 / 0.297 |

Maximum logit difference was 0.015625 from BF16 GEMM/GEMV rounding in this random
probe. It is not a full-model quality assessment. The server caps prefill chunks
at 256, so the 1024-row head result does not describe the default serving path.
[Head probe data](results/glm53_last_head_2026-10-29.json).

The block-parallel KDA experiment solves 32-token unit-lower-triangular systems
and propagates states once per block. It uses FP32 algebra and centered gate
factors for bounded vector log gates in [-5, 0], retains nonzero initial states,
and pads tails with zero updates. A short recurrence-only probe at 64 heads x 128
dimensions found it slower than mlx-lm's existing Metal recurrence:

| Tokens | Recurrent (ms) | Block32 (ms) |
| ---: | ---: | ---: |
| 256 | 1.467 | 2.235 |
| 1024 | 3.208 | 5.659 |

Maximum output difference was 0.000489; maximum final-state difference was
0.001205 with `MLX_ENABLE_TF32=1`. The experiment remains in the benchmark script,
not in the serving dispatch. Faster on-GPU block solving/state propagation is
still needed before enabling it. [KDA probe data](results/glm53_kda_blocks_2026-10-29.json).

```bash
env MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_head.py \
  --output head.json
env MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_kda.py \
  --output kda.json
```

Eleven fast regressions pass, including final-row head input shape, full-logit
parity, cached continuation and actual runner dispatch through the quantized
loader. Long quality tests and lengthy decode runs are deferred.

## Short serving comparison after restart

Both runs use `MLX_ENABLE_TF32=1`, 4096-token context/budget, 256-token prefill
chunks, concurrency one, disabled prefix reuse and the same synthetic inputs.
Each run warms with eight generated tokens, then measures one request per length
with only two output tokens. The earlier running server already had absorbed MLA,
selected-KV DSA and grouped asynchronous evaluation. The restarted server adds
KDA Q/K/V fusion, expert sorting, dense prefill SDPA, compiled mHC and final-row
logits. KDA still uses the original recurrent Metal kernel. This comparison
measures the combined change, not attribution to any one optimization.

| Input tokens | TTFT before/after (s) | TTFT reduction | Decode before/after (tokens/s) |
| ---: | ---: | ---: | ---: |
| 256 | 1.236 / 0.967 | 21.8% | 1.831 / 1.703 |
| 1024 | 4.612 / 3.647 | 20.9% | 1.940 / 1.711 |

[Before data](results/glm53_prefill_server_before_2026-10-29.json) and
[after data](results/glm53_prefill_server_after_2026-10-29.json). Decode was slower
in these observations; each rate uses only a single inter-token interval, so this
is not a steady-state throughput result. The focus of this short run is prefill;
long decode and quality evaluations remain pending. The updated service passed
startup and generation, including Q/K/V preparation in all 34 KDA layers.

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
- Decoder layers submit groups of four asynchronously, including their updated
  cache state; the caller evaluates the final output. Scheduler-level overlap
  remains disabled. Group size is not yet tuned from profiling.
- Single-token MLA decode and sparse attention compute in compressed latent
  space, without expanding historical KV across heads. Dense prefill expands K/V
  once for faster SDPA. Reassociation changes BF16 rounding in the latent path;
  numerical and full-weight smoke comparisons are documented above.
- Sparse DSA gathers at most `index_topk + index_kpool - 1` compressed KV rows per
  query before SDPA. It skips unselected KV in attention, but currently uses
  generic gather/SDPA operations rather than a fused indexed Metal kernel.
  Prefill gather scratch scales with query count times selected count times
  latent rank; the measured increase is documented above.
- The indexer retains raw key/gate history and rebuilds pooled candidates.
- Compatible KDA Q/K/V projections are fused at load time. Short convolutions,
  gates and gated normalization still execute as separate operations. mHC's
  internal calculations compile together, while the final block-output gate/add
  remains separate. MoE uses mlx-lm's sorting path with the clamped activation.
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

- [x] Replace per-layer blocking evaluation with asynchronous groups of four,
      including cache side effects and a final partial group.
- [x] Implement absorbed MLA for decode and sparse attention; validate FP32,
      BF16, 512-dimensional latents and mixed-bit KV projection.
- [x] Use original head-dimension SDPA for dense prefill when it is faster than
      absorbed attention; preserve cached causal alignment.
- [x] Implement selected-KV DSA execution with correct complete-pool and
      incomplete-tail behavior. Keep expanded dense masking as a test oracle.
- [ ] Tile prefill query gathers or fuse indexed latent attention to reduce
      prefill scratch and avoid materializing a per-query selected-KV buffer.
- [ ] Update pooled indexer state incrementally instead of rebuilding old pools
      and copying all key/gate history on every decode step.

### P2: improve matrix utilization, fusion and serving throughput

- [x] Fuse compatible KDA Q/K/V at load time, preserving mixed-bit fallback and
      original parameter names without retaining duplicate projection storage.
- [x] Project only the final hidden-state row for next-token generation.
- [ ] Accelerate KDA block-parallel prefill. The matrix-only Block32 experiment
      regresses against the recurrent Metal kernel and remains benchmark-only.
- [ ] Fuse short convolutions, gates and gated norm; retain the existing Metal
      recurrence kernel as a comparison point.
- [x] Compile mHC normalization, projection, Sinkhorn and residual mixing;
      preserve the twenty rounds and FP32 coefficient calculations.
- [ ] Fuse the final mHC block-output gate/add and profile remaining launches.
- [x] Restore mlx-lm expert token sorting/grouping with clamped SwiGLU.
- [ ] Evaluate fused gate/up projections and an M5-aware mixed-bit expert path
      without changing routing or silently requantizing the checkpoint.
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
KDA recurrence are already used. There is no Core ML/Apple Neural Engine execution path or
Ultra-specific topology tuning. Neither fact establishes that such a path would
improve this workload; measure before choosing another backend or partitioning.
