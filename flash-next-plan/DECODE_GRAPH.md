# Flash-Next offload: segmented decode CUDA Graph

## Scope

An opt-in first implementation uses SGLang's `breakable` backend. GPU computation
between routed-MoE boundaries is captured; each routed MoE (cache route/plan,
worker enqueue, mmap/staging/H2D, CUTLASS and last-use events) runs eagerly on every
replay. This is **not** a full-model single CUDA Graph, nor a captured CUTLASS MoE.
The aim is to reduce the launch overhead outside routed MoE without freezing the
expert mapping or changing the existing worker protocol. Speedup is unmeasured.

Current supported configuration: threaded runtime, TP/DP/EP/PP=1, one running
request, decode capture batch size 1, no torch.compile, eager prefill. Same-card NEXTN with 2/3/4-token verify and adaptive cache is also supported; see `tools/qwen38_cache_tests/MTP.md`.
Existing slot count/layout and optional decode-cache seeding are unchanged.
PLE's UVA gather kernel stays inside captured GPU segments; file-prefetch advice
is skipped during capture as before. No model weights are downloaded.

## Enable

Keep your existing model, pack and memory settings. Set `runtime_mode` to
`threaded` in `--model-loader-extra-config`, then use:

```bash
--cuda-graph-backend-prefill=disabled \
--cuda-graph-backend-decode=breakable \
--cuda-graph-bs-decode 1 \
--cuda-graph-max-bs-decode 1
```

Batch-size options may be omitted: the offload hook resolves them to `[1]` and 1.
JSON graph configuration is supported with the same restrictions. Unsupported
full/tc_piecewise decode or enabled prefill graphs fail early, rather than
capturing stale host state. Default remains eager; rollback is
`--cuda-graph-backend-decode=disabled`.

NVIDIA breakable graphs require `cuda-python` in the runtime (as does the existing
breakable backend); this change does not install dependencies or alter images.
Capture adds startup time and graph memory. Existing `--warmups qwen38_offload`
can then exercise representative requests before serving.

The local/full-model canary accepts `--decode-graph --runtime-mode threaded` and
forwards the graph flags. `docker_model.sh` forwards these after `-- canary`.

## Correctness details

- The eager boundary receives tensors explicitly, not a captured dispatch
  closure. The backend weak-refs pool intermediates and owns a fixed output bridge.
- Capture uses a shape-only zero-output stub. Upstream captured route kernels
  have not run yet, so no transfer is allowed to inspect capture-time route IDs.
- Warmup and every replay execute real cache planning, slot remapping and
  transport, followed by CUTLASS. Fresh output is copied into the fixed bridge.
- Computed events still cover the real eager MoE reads before slots/arena may be
  overwritten. Worker errors propagate; full capture guards remain intact.
- This first step does not remove per-layer host waits, miss bytes, or MoE launch
  overhead. Capturing CUTLASS separately would require separating prepare/execute/
  completion while preserving slot lifetime; it is deliberately not enabled here.

## Validation

CPU tests exercise the actual breakable decorator and quantization boundary with
stand-ins: no capture-time transport, changed routes/activations on each replay,
fresh remapping/output bridge, propagated errors, supported configuration and
canary CLI forwarding. These are protocol tests, not GPU execution evidence.

The GPU small suite now includes
`test_threaded_breakable_decode_cutlass_parity`: synthetic NVFP4 weights, real
ModelOpt transform, cached-vs-resident CUTLASS comparison with input/route changes,
miss/hit, forced eviction and eager arena followed by graph replay. It asserts a
fresh runtime invocation for each replay. Run on SM120; no downloads:

```bash
python3 tools/qwen38_cache_tests/run_gpu.py \
  --require-sm120 --output /tmp/qwen38-graph-small.json
```

Then compare full-model canaries with identical deployment/cache settings,
contexts, probe prompts and token counts, once eager and once `--decode-graph`.
Check visible answers (including arithmetic), finite outputs, no worker timeout,
actual decode graph use in server logs, warm decode token/s and startup time.
Do not judge speed using first-request latency alone. Full-model capture,
attention/PLE integration and performance require the user's GPU machine; this
workspace has no PyTorch/CUDA/weights and cannot validate them.
