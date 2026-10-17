# Qwen3.8 FP4-Local compatibility (2026-10-17)

Target: `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision
`b9ce18a6887e18c089fdea8509a33e63429a3300`, SM120 / TP=1.
Integrated patches 11 and 12 from `qwen38-optimization-patches-20261016.tar.gz`
on top of `acd9c47fed94afe9c753631e4f88df58ca3f676f`, retaining the existing
NVFP4 transport, adaptive fill CTA policy, shared-fill overlap and PLE double
staging. The independent optimization candidates were not reapplied.

## Runtime behavior

- Main experts: CSF block scales are restored losslessly during conversion;
  the existing NVFP4 packed cache ABI and slot size are retained.
- Attention/shared projections: MXFP8 checkpoint `.weight_scale` is loaded as
  native UE8M0 `.weight_scale_inv`, including merged projections.
- The 96-row GDN a/b projection needs a padded 128-row compute view for
  FlashInfer CUTLASS. Padding is prepared at load time; logical checkpoint
  parameters and returned output shapes remain unchanged. Tuple inputs, bias
  and CUDA graph replay are covered by the real-weight check.
- PLE: bounded lazy U8/E4M3 packed rows (90 bytes per 160 dimensions), with
  double staging and GPU gather dequantization. The checkpoint's FP32
  `weight_scale_2` is applied in the gather before BF16 rounding, and missing
  or invalid global scales fail loading. An old pinned/file launch
  setting still uses indexed lazy loading for FP4 PLE. Keep PLE offload enabled.
- HC/PLE norm checkpoint weights retain FP32; fused norm emits BF16.
- MTP: packed resident W4A16 weights with separate gate/up global scales;
  real payload loading and synthetic kernel parity are checked. Full-model
  MTP quality, throughput and memory capacity remain unmeasured.
- Both target and MTP loaders accept the snapshot's `tensors/` subdirectory
  while rejecting absolute paths and `..` traversal. HF blob symlinks work.

`SGLANG_QWEN38_CACHE_W4A16` defaults to **0**, overriding patch 12's test
default. Setting it to `1` enables the optional main-expert SIMT W4A16 path;
it does not control MTP. `SGLANG_QWEN38_FP32_NORM_FUSION` defaults to **1**.

The 5090 D synthetic main-MoE comparison (32 experts, 2560/640, top-k 10)
measured W4A16/W4A4 latency ratios of 1.11 / 2.24 / 7.98 / 30.85 for
1 / 4 / 16 / 64 tokens. These are layer diagnostics, not service throughput.
W4A16 stays available for investigation but is not the serving default.

## Local validation artifacts

Evidence is under `fp4_local_20261017/`:

- `cpu.log`: 267 CPU tests and real SGLang import/config/runner checks.
- `checkpoint.json`: full snapshot index coverage and 128 packed PLE pairs.
- `mtp_payload.log`: all 4,637 MTP tensors read and validated, including
  FP8 block-scale validation on CPU.
- `kernels.json`, `microbench.json`: MTP numerical/graph checks, exact packed
  PLE parity, cross-stream lazy gathering and 512-expert MTP microbenchmarks.
  The real PLE check covers both ends of all 128 shards, invalid rows and
  reversed lookups across streams under a 1 MiB host cache budget, including
  the actual global factor `3.324236240587197e-5`.
- `components.json`: native MXFP8, FP32 norm and FP32/BF16 GDN checks.
- `dense.json`: real checkpoint shared gate/up and GDN a/b/QKVZ loading,
  byte equality before postprocess, numerical reference and padded graph checks.
- `cached_w4a16.json`: A16 reference, cache hit/miss/eviction/arena,
  threaded/sync runtimes, verify graph and shared-fill overlap; A4 comparison.
- `nvfp4_gpu_regression.log`: existing PLE staging, BF16 GEMV and shared-fill
  GPU tests to cover the previous model's paths (6 passed).
- `process-isolation.json`: two actual checkpoint layers converted in separate
  child processes, parity passed and all exported bytes matched the original.
- `pack.json`: final manifest/ABI validation, 48-layer / 432-case conversion
  parity (maximum absolute error zero) and 672 tensor-boundary byte samples.

The GPU/image environment is RTX 5090 D, PyTorch `2.13.0+cu130`, local image
`swallow-sglang:v0.5.21-qwen38-capacity`. No dependencies were upgraded.

## Pack and launch settings

The new pack directory is separate from the old NVFP4 pack:

```text
/home/jay/workspace/prod/cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/
```

`provenance.json` and `runtime-source.patch` record the exact converter source
as the base commit plus an uncommitted patch fingerprint.
The final runtime source is recorded separately after the small-N and PLE
global-scale corrections.
The pack manifest's `sglang_commit` must be used verbatim in loader config.

For the existing container cache mount and launch script, the prepared settings
are in `fp4_local_20261017/launch.env`. They select the new model/revision/pack,
keep `load_format=qwen38_nvfp4_cache`, 5140 total slots, MTP off and lazy PLE.
Slots and context capacity require full-model acceptance; no capacity increase
is assumed from checkpoint size alone. Component validation did not change
deployment; a subsequent service rollout is recorded below.

Full conversion with parity now uses at most four layers per subprocess
(`--layers-per-process`, default 4), bounding backend CUDA allocations across
48 layers. A child failure stops conversion before assembly. Existing export
provenance and per-layer parity records are checked before final assembly.
The local validation resumed completed exports using isolated batches with the
same saved converter source. Neither the old pack nor its checkpoint is reused.

Long-context quality and the maximum slot budget have not been accepted by
these component checks.

## Subsequent service deployment

On 2026-10-17 the user's Compose service was switched to FP4-Local with W4A16,
5140 slots, FP32 GDN and MTP off. Startup, text/image requests, OpenAI tool
calling and SSE streaming passed. The 4080 retains the original Flash-Next
BF16 encoder, so this is a mixed language/vision deployment. NVFP4 and 27B
presets were also actually started and passed text/image acceptance.

Model switching and rollback are managed by
`/home/jay/workspace/prod/homelab-llm/llmctl`; instructions and evidence are in
`/home/jay/workspace/prod/homelab-llm/models/README.md` and
`models/acceptance/deployment-20261017.json` under that deployment directory.
Each preset pins the local image and uses either a frozen source snapshot or
the original 27B image source. The component validation's `launch.env` retained
its W4A4 experimental setting; the deployed `models/fp4-local.yml` selects A16.

## Phase-specific routed-MoE activation (2026-10-17)

The deployed FP4-Local preset now sets `SGLANG_QWEN38_CACHE_W4A16=1` and
`SGLANG_QWEN38_CACHE_PREFILL_W4A16=0`: CUTLASS W4A4 prefill, W4A16 decode
and verify. The new variable defaults to the old global flag for compatibility.
Selection uses the explicit execution phase carried through the eager graph
boundary, not batch size. A one-token prefill uses A4; multi-token verify uses
A16. The same cache layout and serialized expert pack are reused.

`fp4_local_20261017/phase_activation.json` records the real-runner parity suite:
cache misses, hits, evictions, prefill arena, single-token prefill and breakable
CUDA graph replay (decode and multi-token verify). CPU tests check independent
mode selection and reject invalid settings. Quality equivalence to all-A16
is not claimed: prefill now deliberately quantizes activations to FP4.

The SIMT W4A16 path was inappropriate for large prefill: its 16-token chunks
and per-route reductions caused a 20–31x layer-level slowdown at 64 tokens.
An experimental grouped Tensor Core kernel was also checked (10.89x over SIMT
at 8192 tokens), but is not wired into the deployed runtime. Its source and
same-weight parity/latency results are archived with `grouped_prefill*` artifacts.

End-to-end diagnostic after deployment: uncached 16384-token synthetic input,
4 output tokens, 2.298 s wall time. Logged 8192-token prefill chunks were
3737.79 / 8028.35 tokens/s. The prior all-A16 16K checks took ~53 s with a
different slot count and repeated token ID, so this is not a controlled A/B
throughput result. Current user settings (6140 slots, no explicit KV cap) were
preserved. A recoverable allocator OOM mapping retry was logged; memory stability
is not established by this successful request.

## Pinned PLE, shared scripts and MTP/EPD (2026-10-17, subsequent update)

FP4 PLE now honors pinned/file/lazy selection and defaults to pinned. Compressed
weight/scale planes are streamed into 90-byte rows, with bounded copy scratch;
the GPU gather applies the FP32 global scale directly. The real table is
320001536 rows / 26.82 GiB. Lazy remains explicitly selectable. Small pinned
GPU checks prove byte-exact output versus lazy, invalid IDs, preallocated output,
cross-stream access and native CUDA graph replay (`pinned_ple.json`).

Both production FP4 presets use `sglang-fp4-local.sh`; A4 only passes `--decode-a4`.
The legacy A4 entry is a forwarding shim. All settings are maintained once.

MTP loading audits now support read-only parameter loader properties and restore
callbacks after both success and failure. The pure-text guard is removed while
other EPD constraints remain. A real steps=2 service passed remote-ViT image
requests, >8192-token chunked image prefill, prefix-cache reuse, image-at-boundary
and multi-iteration decode. Graph memory is reported for target verify and draft
phases. Startup uses 5340 slots / mem_fraction_static=0.90 to leave scratch for
MTP prefill; token capacity 235456 retains the configured 220000 context.
No full quality or speedup claim is made. The production acceptance record is
`models/acceptance/fp4-pinned-mtp-epd-20261017.json` under the deployment root.

## MTP performance investigation (2026-10-17)

See `mtp_perf_20261017/REPORT.md` for service logs, CPU/GPU traces, the
verify/mailbox path difference, short-request throughput and its limitations.
The shared production launcher keeps MTP=2 enabled and observation logging
off. See `mtp_perf_20261017/VERIFY_FIX.md` for the deployed verify mailbox fix. `llmctl profile start|stop|summary`
controls bounded CPU/GPU profiling without restarting the service.
