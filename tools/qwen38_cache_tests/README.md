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
