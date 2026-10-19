# Qwen3.8 cached NVFP4 verification

Use a local SGLang v0.5.21 image chosen explicitly by the caller:

```bash
tools/qwen38_cache_tests/docker_cpu.sh "$SGLANG_IMAGE"
```

For the image currently installed on the development host:

```bash
sg docker -c 'tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21'
```

`sg docker` refreshes group membership for this command when the running app
still has its old supplementary groups. It is unnecessary after a new login.
The script uses `--pull never`, `--network none`, a read-only source mount and
no GPU. It preserves the image's `HF_HOME`, sets offline flags, and neither
starts a server nor downloads any model files.

The container runs small `unittest` CPU fixtures, imports the changed SGLang
modules, parses the saved public ModelOpt config, checks that the custom MoE
creates no expert parameters, and verifies CUTLASS runner payload wiring with
a mocked execution call. The mock does not prove kernel numerical parity.

For a Python environment with CPU PyTorch and NumPy:

```bash
python3 tools/qwen38_cache_tests/run_cpu.py
```

Without PyTorch, dependency-free tests still run, and tensor tests are skipped.

The public model metadata in `flash-next-plan/model-metadata/` is pinned to
revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`. Only README, config and
quantization metadata were saved. No checkpoint weights were downloaded.

## Runtime pack v2

Export an already transformed resident SM120 ModelOpt FusedMoE layer using
`export_runtime_layer(layer, output / 'layer-000')` from
`sglang.srt.model_loader.qwen38_nvfp4_pack.export`. Call after the existing
`ModelOptNvFp4FusedMoEMethod.process_weights_after_loading()`. This helper
exports six per-expert tensors plus scalar float32 `w13_input_scale_quant` and
`w2_input_scale_quant`. SwiGLU parameters are derived from the loaded model's
runner configuration using the same helper as the resident CUTLASS path.

The local checkpoint converter in `qwen38_nvfp4_pack.convert` constructs and
transforms one real resident layer at a time, then invokes this exporter. Its
execution and numerical parity still require SM120 validation. CPU fixtures
use `synthetic=True`, which must never be used for a production pack.

Assemble exported layers using the existing `build.py` CLI. The builder streams
raw files in 1 MiB chunks. Packs from v1 are rejected because their input scales
are missing; rebuild them from transformed layers.

The runtime allocates cache and optional arena before SGLang sizes its KV pool.
The default sync mode uses a bounded one-expert-tensor staging buffer and synchronous transfers.
Routing with more unique experts than cache slots selects the full-layer arena;
other routing uses the cache. CUDA compute is synchronized before any reuse.
This implements a correctness reference path, with per-layer CPU routing
synchronization. Async events, JIT routing, full model execution, long-context
and MTP are not validated. Speculative decoding is rejected at startup.

GPU continuation, conversion and canary commands: [RUN_ON_GPU.md](RUN_ON_GPU.md).

## Cache metrics and 112-slot baseline

`full_model_canary.py` now copies the latest `Qwen38ExpertCacheStats[...]` JSON
line from `server.log` into `report.json` as `cache_stats`. When the server
emits decode throughput lines, the canary also records a `decode_throughput`
summary with `avg_token_per_second`, `max_token_per_second` and
`last_token_per_second`.

For the current 5090 + `--mem-fraction-static 0.90` deployment profile, the
working high-cache baseline is `cache_slots_per_layer=112`. See
`CACHE_TUNING_112.md` for the measured hit-rate and throughput numbers.

The later 96-slot, long-context patch-by-patch comparison is recorded in
[OCT13_PATCH_COMPARISON.md](OCT13_PATCH_COMPARISON.md). These two workloads
have different context lengths and cache settings, so compare their results
within each report.

Bulk prefill transport, bounded large staging buffers, and synthetic validation:
[BULK_TRANSPORT_OPTIMIZATION.md](../../flash-next-plan/BULK_TRANSPORT_OPTIMIZATION.md).
Async/device/threaded modes default to four buffers of up to 64 MiB each;
canary flags `--staging-buffers` and `--staging-buffer-mib` configure the ring.

Optional device/threaded arena-to-decode seeding: `--warm-decode-cache` (off by
default). Implementation and GPU comparison guidance:
[DECODE_CACHE_WARMUP.md](../../flash-next-plan/DECODE_CACHE_WARMUP.md).

Per-layer capacities and profiling-only route capture / CPU replay:
[LAYER_CAPACITY_REPLAY.md](../../flash-next-plan/LAYER_CAPACITY_REPLAY.md).

Checkpoint directory or cached Hugging Face repo ID (no downloads), with updated
canary/conversion/Docker commands:
[OFFLINE_MODEL_ID_SUPPORT.md](../../flash-next-plan/OFFLINE_MODEL_ID_SUPPORT.md).

Representative pre-listener startup warmup: `--warmups qwen38_offload` on the
Python HTTP server, or canary flag `--offload-startup-warmup`. See
[STARTUP_WARMUP.md](../../flash-next-plan/STARTUP_WARMUP.md).

## Segmented decode graph (opt-in)

The canary accepts `--decode-graph --runtime-mode threaded`. Prefill remains eager;
decode uses batch-1 `breakable` graphs, with routed MoE/host transport replayed
eagerly at every boundary. Details, limitations and GPU validation steps are in
[DECODE_GRAPH.md](../../flash-next-plan/DECODE_GRAPH.md). The small GPU suite includes synthetic graph
replay versus resident CUTLASS parity; it never downloads weights.

## Remote ViT on another GPU (EPD)

Use `--language-only --encoder-urls ...` on the offload language worker, not
`--language-model-only`; the existing encoder uses `--encoder-only` and the normal
loader on its own GPU. Canary accepts `--encoder-urls ... --image /local/image.png`.
See [REMOTE_VIT_OFFLOAD.md](../../flash-next-plan/REMOTE_VIT_OFFLOAD.md), including
the Docker networking requirement and two-patch merge order.

## Encoder-only selective loading

The Qwen4Exp EPD encoder uses visual-only safetensors reads (index shard
selection, header fallback, and name filtering before `get_tensor`). This avoids
language tensor materialization and whole-shard GPU staging. Missing visual
parameters now fail startup. See `flash-next-plan/REMOTE_VIT_OFFLOAD.md` for
encoder memory/cache settings and checkpoint format limitations.

同卡 MTP（默认关闭）的配置与验收入口见 [MTP.md](MTP.md)。

专家缓存总预算可用 `cache_slots_total` 指定；canary 对应 `--total-slots`。原 105 slots/层等价于总预算 5040。该字段与每层容量选项互斥，adaptive 在总预算内分配层容量。

Prefill 直接读取已命中的 cache、仅将缺失专家加载到 arena：
[PREFILL_DIRECT_CACHE.md](PREFILL_DIRECT_CACHE.md)。threaded runtime 下默认开启（需要 `prefill_reuse_cache`，同为默认值），写 `"prefill_direct_cache": false` 回退。

配套的 prefill 计算/加载重叠、KV stages 容量调整保留热点和 decode 提前计算：
[CACHE_OVERLAP_AND_RESIZE.md](CACHE_OVERLAP_AND_RESIZE.md)。

是否真的拆分由 `cache_overlap_policy` 决定（默认 `auto`，只在实测有净收益的档位拆分；`always` 用于强制对照）。decode 提前计算没有实测盈利区间，仍默认关闭：
[动态路径选择复验](cache_overlap_20261019/adaptive/REPORT.md)。
