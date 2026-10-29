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
  A four-active-request GLM throughput probe hit an existing CPU/MPS device
  mismatch in `FutureMap.stash`; current serving measurements use the default
  single active request, with concurrent clients queued at the server.
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
- The indexer caches completed pooled keys in a geometrically grown buffer.
  Raw indexer keys/gates retain only the incomplete pool (at most three tokens
  for the checkpoint's pool size of four). Latent MLA KV still grows by
  concatenation; this optimization does not change that buffer.
- Compatible KDA Q/K/V projections are fused at load time. Short convolutions,
  gates and gated normalization still execute as separate operations. mHC's
  internal calculations compile together, while the final block-output gate/add
  remains separate. MoE uses mlx-lm's sorting path with the clamped activation.
  Quantized single-token decode checkpoints the up/gate/down projection outputs
  individually; larger batches retain mlx-lm's original execution path.
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

## GLM MLX shape and dispatch audit

The 2026-10-30 audit used MLX 0.32.4 on M5 Ultra. Absorbed MLA decode
passes Q `[1,64,1,512]` and shared KV `[1,1,L,512]`. Calling the installed
SDPA API with `force_fused=True` rejects this layout because query length times
GQA factor is 64, above the vector kernel limit of 32. D512 vector attention
also defaults to GQA=8, no array mask, and at least 1024 keys. This is a
fallback to Metal matrix operations, not CPU execution.

Equivalent layouts were benchmarked before changing dispatch: split query heads
into eight batches of eight with broadcast KV, or fold heads into 64 independent
query rows for full fused attention. Neither is enabled: at BF16 L=2048,
original SDPA took **0.0298 ms**, fused vector **0.2217 ms**, and fused full
attention **0.1845 ms**. The fused kernels are substantially slower here.
These are compiled, amortized operator measurements with resident random inputs;
they do not establish end-to-end model throughput. The backend's
[eligibility and dispatch implementation](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp)
explains the shape limits and deliberate fallback heuristics.

Quantized expert inputs were also checked against the
[MLX gather-QMM dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp):
up/gate K=4096, N=2048, 4-bit and down K=2048, N=4096, 5-bit meet fast
quantized-matvec alignment requirements. Their singleton dimensions do not by
themselves imply a slow path. KDA's vector-valued forget gate intentionally uses
mlx-lm's vector-gated recurrent kernel; substituting its scalar-gate packed
kernel would change the model equations.

One unnecessary shape cost is fixed by default: single-token sparse MLA trims
the padded tail before gathering and omits the resulting all-valid mask. All
selected complete pools are visible to that final query. Multi-token prefill
retains its per-query causal selection mask. At a pool boundary the selected
length becomes 2048 instead of 2051, allowing aligned matrix dimensions.
BF16 SDPA then measured **0.0552 → 0.0334 ms** at that boundary; with one to
three tail tokens the reduction was approximately 4–5%. This is a small
operator optimization, not a claim of substantial serving acceleration.

Reproduce eligibility checks, both fused alternatives, and all four sparse-tail
lengths with:

```bash
SGLANG_USE_MLX=1 MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_sdpa_dispatch.py \
  --output /tmp/glm-sdpa-dispatch.json
```

[Dispatch and padding measurements](results/glm53_sdpa_dispatch_m5_ultra_2026-10-30.json)
include every timing round, dtype, device, maximum output difference and native
eligibility error. Expanded-KV oracle tests cover every tail length, tail
selection disabled, quantized projections, BF16, multi-row decode and
multi-token causal continuation.

Full-weight HTTP validation compared the GLM module from `6c391abebf` with the
padding fix, using checkpoint revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`, TF32=1, one active request,
4352-token context/budget, 256-token prefill chunks, sampling enabled, and
disabled radix caching/scheduler overlap. Each input length generated 24 greedy
tokens in three timed trials after warmup; the two model processes ran
sequentially without another GPU workload. Decode rate excludes the first token.

| Input tokens | Decode before/after (tokens/s) | TPOT before/after (ms) | TTFT before/after (s) |
| ---: | ---: | ---: | ---: |
| 32 | 2.697 / 2.728 | 370.8 / 366.6 | 0.659 / 0.655 |
| 1024 | 2.723 / 2.740 | 367.2 / 364.9 | 3.621 / 3.703 |
| 4096 | 2.698 / 2.709 | 370.6 / 369.2 | 19.450 / 19.800 |

The sparse 4096-token case changed by only **+0.4%**, below the observed
trial-to-trial variation. The unchanged dense cases also drifted upward slightly;
**no measurable end-to-end throughput improvement is established**. Keep the
small padding/mask simplification enabled, without forcing slower fused kernels.
A separate 4096-input/8-output greedy continuation produced identical token IDs.
The 63 relevant GLM, attention-adapter and worker-routing tests passed, including
20 parametrized subtests.

[Serving before](results/glm53_sdpa_padding_serving_before_2026-10-30.json),
[after](results/glm53_sdpa_padding_serving_after_2026-10-30.json),
[greedy-token parity](results/glm53_sdpa_padding_serving_parity_2026-10-30.json).

## Quantized expert decode throughput

GLM single-token decode now evaluates each quantized routed projection output
before building the next projection's graph. Up/gate remain 4-bit and down
remains 5-bit on this checkpoint; routing, weights and clamped SwiGLU are
unchanged. Prefill, multi-row inputs and unquantized projections use mlx-lm's
original path. This evaluation policy is enabled by default.

Full-weight probes first established that synchronizing entire layers or simply
changing async layer-group size did not improve throughput. With identical
restored request state and token inputs, the projection-output boundaries gave
the following medians on M5 Ultra (`MLX_ENABLE_TF32=1`):

| Evaluation policy | Decode time (ms/token) |
| --- | ---: |
| Original four-layer async groups | 571.9 |
| Checkpoint up/gate/down outputs | 344.2 |
| Checkpoint projection inputs only | 622.3 |
| Checkpoint whole MoE inputs/outputs | 912.8 |

Each policy used three alternating-order trials of five decode steps, excluding
the first step's timing, with exact greedy-token parity. This is a measurement
of evaluation policy, not a kernel-level explanation of the backend behavior.
Raw evidence: [module attribution](results/glm53_decode_module_attribution_m5_ultra_2026-10-29.json),
[layer groups](results/glm53_decode_eval_groups_m5_ultra_2026-10-29.json),
[projection boundaries](results/glm53_decode_projection_boundaries_m5_ultra_2026-10-29.json).
The module-attribution rows overlap (MoE includes expert QMM); they must not be
added together. Instrumentation changes execution timing.

The change passed 63 GLM/reference, attention-adapter and worker-routing tests,
including exact FP32/BF16 expert output parity for mixed 4/5-bit projections,
clamping, sorted prefill and multi-row fallback.

Full HTTP serving was measured across 2026-10-29/30 with the cached
`mlx-community/GLM-5.3-Flash-4bit` checkpoint, revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`. Both implementations used TF32=1,
one active request, a 4352-token context/scheduling budget, 256-token prefill
chunks, sampling enabled, and disabled radix caching/scheduler overlap. The
reference GLM module is from `65bf6e3b70`; the rest of the runtime is identical.
Two timed trials followed warmup. Every request generated 16 greedy tokens with
`ignore_eos=True`; no other GPU benchmark ran alongside the servers.

| Input tokens | Decode before/after (tokens/s) | Decode speedup | TPOT before/after (ms) | TTFT before/after (s) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.718 / 2.974 | 1.73x | 582.2 / 336.4 | 0.674 / 0.687 |
| 1024 | 1.695 / 3.150 | 1.86x | 590.0 / 317.8 | 3.810 / 3.954 |
| 4096 | 1.656 / 2.979 | 1.80x | 604.0 / 335.7 | 20.058 / 20.373 |

[Serial serving before](results/glm53_projection_eval_serving_before_2026-10-29.json),
[after](results/glm53_projection_eval_serving_after_2026-10-29.json). TTFT did not
improve; the change targets decode execution. The reported decode rate excludes
the first output token.

Four concurrent HTTP clients, each sending 32 input tokens and requesting 16
output tokens, measured aggregate output throughput of **1.637 → 2.823 tokens/s
(1.72x)**. Clients queued behind the default single active GLM request; this is
not a four-active-request native batch. The rate includes prefill, queueing and
HTTP, measured as all 64 completed output tokens divided by total wall time.
[Queued throughput before](results/glm53_projection_eval_concurrent_before_2026-10-29.json),
[after](results/glm53_projection_eval_concurrent_after_2026-10-29.json).

Reproduce the client workloads against each running server:

```bash
python benchmark/mlx/bench_glm5_next.py \
  --input-lengths 32 1024 4096 --output-length 16 --repeats 2 \
  --server-mlx-enable-tf32 1 --output serial.json
python benchmark/mlx/bench_glm5_next_concurrent.py \
  --concurrency 4 --input-length 32 --output-length 16 --repeats 2 \
  --output queued-throughput.json
```

[Full-weight parity](results/glm53_projection_eval_serving_parity_2026-10-29.json)
includes identical eight-token continuations after a 4096-token input and exact
output IDs for every queued request in both trials. This is a synthetic serving
comparison, not a saturation curve or a model-quality evaluation. An earlier
attempt with four active requests failed on the baseline's
[CPU/MPS device mismatch](results/glm53_projection_eval_max4_failure_2026-10-29.json);
no concurrent throughput result is claimed for that unsupported configuration.

## Incremental pooled indexer cache

Completed indexer pools now keep their original weighted means instead of
recomputing them from all historical keys/gates every decode step. The fourth
MLA cache array stores the pooled-key buffer; its valid length comes from the
sequence offset. It participates in normal cache evaluation and state transfer.
This implementation is enabled directly on the GLM5-Next path.

Reproduce the paired production-shape probe with the same BF16 weights, raw
prefix and advancing decode inputs on each side:

```bash
SGLANG_USE_MLX=1 MLX_ENABLE_TF32=1 python benchmark/mlx/bench_glm5_next_indexer.py \
  --reference-ref 26c029a9affacc97df528d37eda7d37d90792631 \
  --repeats 31 \
  --output benchmark/mlx/results/glm53_indexer_incremental_m5_ultra_2026-10-29.json
```

Apple M5 Ultra, MLX 0.32.4, mlx-lm 0.32.2, batch one. Four warmup steps precede
31 interleaved timed decode steps. The reference is the same GLM implementation
before this change. Indexer measurements include only the indexer; MLA includes
all attention projections, index selection, selected-latent attention and cache
writes. These are single-layer probes with random weights.

| Prefix tokens | Indexer before/after (ms) | Indexer speedup | MLA before/after (ms) | MLA speedup | Indexer cache before/after (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.201 / 0.220 | 0.91x | 0.881 / 0.935 | 0.94x | 0.517 / 0.126 |
| 4096 | 0.404 / 0.363 | 1.11x | 0.739 / 0.708 | 1.04x | 2.017 / 0.501 |
| 16384 | 0.467 / 0.387 | 1.21x | 0.884 / 0.777 | 1.14x | 8.017 / 2.001 |

[Raw timings](results/glm53_indexer_incremental_m5_ultra_2026-10-29.json) include
every trial and source provenance. Sparse index sets match exactly, and sampled
MLA outputs have zero observed difference. Cache sizes count indexer arrays
after the stream, including spare pooled capacity; they exclude latent KV,
weights and allocator cache. Dense short-context steps do more pooling work
than before and show a small latency regression. This table does not establish
full-model serving speedup.

Validation: 62 GLM/reference, attention-adapter and worker-routing tests passed,
including weighted pool equivalence across growth, partial tails and cache-state
round trips, FP32/BF16/quantized MLA, chunked continuation and request isolation.

The full `mlx-community/GLM-5.3-Flash-4bit` checkpoint (revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`) was also measured sequentially before
and after. Both servers used `MLX_ENABLE_TF32=1`, concurrency one, a 4352-token
context/scheduling budget, 256-token prefill chunks, sampling enabled, and the
model's default disabled prefix caching and scheduler overlap. Only the GLM
module changed; the reference module came from the revision named above. Each
row is the median of two requests with 16 greedy output tokens after a
1024-input/8-output warmup. No other GPU benchmark ran alongside these requests.

| Input tokens | TTFT before/after (s) | Decode before/after (tokens/s) | TPOT before/after (ms) |
| ---: | ---: | ---: | ---: |
| 1024 | 3.981 / 3.916 | 1.666 / 1.714 | 600.4 / 583.6 |
| 4096 | 20.528 / 20.199 | 1.634 / 1.635 | 612.1 / 612.1 |

Serving data: [before](results/glm53_incremental_indexer_serving_before_2026-10-29.json),
[after](results/glm53_incremental_indexer_serving_after_2026-10-29.json).
The 4K decode rate is effectively unchanged. The roughly 3% short-context
difference and small TTFT differences are within the scope of a two-trial smoke
comparison, not evidence of a stable full-model speedup. The demonstrated gains
are the indexer cache reduction and isolated sparse indexer/MLA latency above.
A separate 4096-token synthetic input produced
[identical eight-token greedy continuations](results/glm53_incremental_indexer_serving_parity_2026-10-29.json).
This checks full-weight sparse continuation parity, not model quality.

## Optimization roadmap

Priorities below are hypotheses from code inspection until traces establish the
actual contribution of each operation. Keep this baseline configuration available
for before/after comparisons; do not combine optimizations into one measurement.
The current focus is measured serving throughput. Memory-only work, including
query-gather tiling and capacity sizing, is deferred while this is the priority.

### Shared MLX backend: reference review (2026-10-29)

Local reference revisions: vllm-metal `93562f1`, Strata `46c152a1`.
These priorities supplement the GLM-specific work below. Ordinary GQA/MHA pool
attention does not accelerate GLM5-Next's native compressed MLA/DSA cache.

1. **P0: direct shared-pool decode attention (in progress).** Replace the
   ordinary attention wrapper's per-request KV padding/concatenation with a Metal
   kernel reading SGLang's token-slot mapping directly. Start with one query per
   request, uniform FP16/BF16/FP32 GQA/MHA, and full-attention layers. Keep the
   existing allocator and radix ownership. Retain contiguous request caches for
   fallback and lookahead steps without scheduler-owned slots. Enable supported
   decode configurations by default; retain an explicit switch for comparisons.
   - [x] Implement the kernel, shared batch metadata and wrapper integration.
   - [x] Verify fragmented slots, unequal lengths, GQA, prefix reuse, batch
     changes and lazy write/read ordering against the existing SDPA path.
   - [x] Measure B=1/4/16 with equal and mixed lengths; record warmed attention
     latency and transient MLX memory (results below).
   - [x] Check real HTTP serving with eight concurrent requests, repeated
     prefixes and overlap scheduling on a random tiny Llama model.
   - [ ] Measure full-model serving throughput and TPOT on a representative
     GQA/MHA checkpoint to quantify end-to-end impact.
   - [ ] Extend to varlen prefill/mixed batches; evaluate M5 NAX after the
     ordinary decode path is established. Reference: vllm-metal's
     `attention/impls/sdpa.py` and `metal/paged_attention.cpp`.
2. **P1: memory accounting.** Include process footprint, request-cache mirrors,
   transient workspaces and desktop headroom; recheck available memory during
   growth. Reference: Strata `metal/src/engine/arena_budget.h`. Recalibrate its
   constants on our workloads rather than copying them.
3. **P1: bounded expert streaming.** Start with safetensors expert ranges and a
   byte-bounded cache including stacking scratch (vllm-metal `moe_offload.py`).
   Evaluate direct reads into shared Metal slots, urgent/prefetch queues and
   in-flight slot protection (Strata `metal/src/engine/stream.h`) separately.
4. **P2: compressed KV reads.** Couple packed KV storage to attention-side
   dequantization; account for scale metadata and prefill scratch. Reference:
   vllm-metal TurboQuant. Validate quality before claiming a capacity benefit.
5. **P2: cost-aware speculation.** Extend existing adaptive speculation with
   measured committed-tokens/time, including a no-draft choice. Reference:
   Strata `include/strata/spec/controller.hpp`; its measured cost constants are
   not portable defaults.

### Direct-pool decode: first implementation and measurements

The path is enabled by default; set `SGLANG_MLX_USE_POOL_ATTENTION=0` to use
the previous padded SDPA path for comparisons. Pool attention requires
the shared attention pool (radix caching enabled, no sliding-window layers).
Supported decode heads have dimensions divisible by 32 up to 256, matching
FP16/BF16/FP32 Q/K/V, and no attention sinks. Other shapes and native-cache
models retain their existing attention path. Lookahead steps use request caches
because they have no newly allocated scheduler slot; fused RoPE pool writes are
also disabled on those steps. Unsynced lookahead KV is flushed before the next
admitted pool decode. Request-cache mirrors remain allocated, so this change
does not eliminate duplicate persistent KV storage.

Reproduce the attention microbenchmark from the MLX environment:

```bash
SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 python benchmark/mlx/bench_pool_decode.py \
  --repeats 31 \
  --output benchmark/mlx/results/pool_decode_m5_ultra_2026-10-29.json
```

Measured on Apple M5 Ultra, MLX 0.32.4, mlx-lm 0.32.2, PyTorch 2.13.0.
Random BF16 attention weights use 16 query heads, 4 KV heads and head dimension
128. Each sample includes projections, RoPE, per-step slot metadata, mirror
writes and output/cache evaluation. Timings are medians of 31 interleaved trials
after warmup. Mixed batches cycle lengths 4096/2048/1365/1024. The memory columns
are one warmed sample of MLX peak active bytes minus active bytes before the
call; they exclude allocator cache and system footprint.

| Batch | Lengths | Padded SDPA (ms) | Pool (ms) | Speedup | SDPA extra peak (MiB) | Pool extra peak (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 0.266 | 0.399 | 0.67x | 0.53 | 0.54 |
| 4 | equal 4096 | 0.445 | 0.450 | 0.99x | 34.12 | 2.16 |
| 4 | mixed | 0.470 | 0.381 | 1.23x | 73.74 | 2.16 |
| 16 | equal 4096 | 0.731 | 0.626 | 1.17x | 136.47 | 8.63 |
| 16 | mixed | 1.077 | 0.543 | 1.98x | 294.92 | 8.63 |

[Raw trials and all 512-token cases](results/pool_decode_m5_ultra_2026-10-29.json)
show small-batch regressions. The default favors the measured mixed-batch and
temporary-memory improvements; the B=1 4K case adds about 0.13 ms per attention
layer. End-to-end impact still depends on layer count and the workload. These are
single-layer attention measurements, not full-model throughput or GLM results.
The [eight-request HTTP smoke](results/pool_decode_http_smoke_m5_ultra_2026-10-29.json)
produced identical greedy tokens with the feature
on/off, including a repeated-prefix request; random weights establish scheduler
and cache parity, not model quality or serving performance.
The same check passed with the environment variable unset (default pool
attention) versus an explicit `SGLANG_MLX_USE_POOL_ATTENTION=0`.

Validation: 218 selected namespace/MLX/kernel tests and 25 parametrized subtests
passed. The unfiltered run had 219 passes and one existing failure:
`test_amx_backend_probe_is_lazy[srt/utils/common.py]` imports unavailable Triton
when loading that module outside the Apple platform stubs. The same failure
reproduced using the test file from the pre-change HEAD. Formatting, import
sorting, Ruff checks and the registered-test lint passed. The full pre-commit
runner was unavailable because its virtualenv dependency was not cached.

### P0: establish attribution and quality guardrails

- [ ] Capture Metal/GPU and CPU submission timelines for prefill and decode.
      Attribute time to MLA expansion, attention, KDA, MoE, mHC and host waits;
      record GPU active/cache memory as well as system memory headroom.
- [ ] Determine which operations actually use M5 GPU Neural Accelerators and
      whether the workload is limited by submission, bandwidth or computation.
      MLX supports these accelerators, but the current port has no kernel-level
      utilization measurements. See [Apple's MLX/M5 discussion](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- [x] Audit production decode shapes against MLX SDPA and quantized-matvec
      dispatch rules. Reproduce D512 fusion rejection locally and benchmark
      equivalent fused layouts; retain faster matrix attention. Shader-level
      utilization attribution remains part of the timeline work above.
- [ ] Add full-weight accuracy/logprob and realistic chat checks, including
      prefill/decode continuation and inputs crossing the DSA top-k budget.
- [ ] Fix the GLM mixed extend/decode CPU/MPS mismatch in `FutureMap.stash`,
      then measure throughput with multiple active native-cache requests.

### P1: reduce avoidable work and synchronization

- [x] Replace per-layer blocking evaluation with asynchronous groups of four,
      including cache side effects and a final partial group.
- [x] Implement absorbed MLA for decode and sparse attention; validate FP32,
      BF16, 512-dimensional latents and mixed-bit KV projection.
- [x] Use original head-dimension SDPA for dense prefill when it is faster than
      absorbed attention; preserve cached causal alignment.
- [x] Implement selected-KV DSA execution with correct complete-pool and
      incomplete-tail behavior. Keep expanded dense masking as a test oracle.
- [x] Remove padded sparse KV rows and the unnecessary single-query mask during
      decode; restore aligned key length at complete-pool boundaries.
- [ ] Tile prefill query gathers or fuse indexed latent attention to reduce
      prefill scratch and avoid materializing a per-query selected-KV buffer
      (deferred while prioritizing throughput).
- [x] Update pooled indexer state incrementally instead of rebuilding old pools
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
- [x] Measure GLM decode evaluation boundaries and checkpoint quantized routed
      projection outputs where full-model token throughput improves.
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
