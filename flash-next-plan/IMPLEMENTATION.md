# Qwen3.8 Flash-Next Offload Implementation Status

Last updated: 2026-10-09

Baseline: SGLang `v0.5.21`, commit
`e00930c5489053f26d86b179cee0d087f846acbb`

Branch: use the actual target checkout branch.

## Implemented

### Load format and startup constraints

- Added `--load-format qwen38_nvfp4_cache`.
- Added `Qwen4ExpForConditionalGeneration` to the language-model-only allow-list.
- The Qwen3.8 load format is selected before the generic ModelOpt loader, so a
  serialized `modelopt_mixed` checkpoint cannot accidentally allocate all 512
  routed experts on the GPU.
- Startup resolution forces:
  - language-model-only mode;
  - `flashinfer_cutlass` MoE backend;
  - CUDA graph disabled;
  - shared-expert fusion disabled;
  - TP/DP/EP equal to 1.

### Dedicated checkpoint loader boundary

- Added strict Qwen3.8 model-shape validation for the planned 48-layer,
  512-expert, top-k=10, hidden-size=2560, MoE-intermediate-size=640 model.
- Added a single-threaded safetensors iterator that checks tensor names before
  calling `get_tensor()`.
- Routed expert tensors and visual tensors are skipped before materialization.
- Dense, shared-expert, PLE and other language-model tensors continue through
  the existing Qwen4-Exp `load_weights` path.
- The custom ModelOpt wrapper delegates dense/PLE quantization to the original
  `ModelOptMixedPrecisionConfig` and replaces routed `FusedMoE` construction
  with a parameter-free marker method.
- The loader refuses unsafe fallback behavior rather than allocating the full
  routed expert checkpoint on GPU.

### Runtime expert-pack format

- Added an ABI-bound manifest containing checkpoint revision, SGLang commit,
  FlashInfer backend, SM target and model shape.
- Added strict layer/tensor coverage, dtype, shape, range, overlap and
  per-expert divisibility validation.
- Added a read-only mmap reader with whole-layer tensor and per-expert tensor
  byte views. It does not create a complete pinned-memory copy.
- Added CPU builder, validator and inspector tools. The builder consumes
  already transformed per-layer tensor files and never downloads weights.

Expected per-expert transformed tensors per layer (v2 additionally stores two
scalar input quantization scales):

```text
w13_weight
w2_weight
w13_blockscale_swizzled
w2_blockscale_swizzled
g1_alphas
g2_alphas
```

### Cache control path

- Added a per-layer decay-LFU policy with LRU tie-breaking.
- Added generation counters, active-slot protection and protection underflow
  checks.
- Added list and tensor ID-remap helpers.

## Synchronous Runtime Continuation (2026-10-09)

Implemented without checkpoint weights or GPU:

- Runtime pack v2 includes both scalar float32 input quantization scales. v1
  packs are rejected rather than silently assuming unit scales.
- Added a transformed-layer exporter with storage checks and bounded per-expert
  streaming; the existing CPU builder now streams files and validates before
  replacing its payload.
- Added fixed-address per-layer decode caches and a shared full-layer prefill
  arena. Overflow in the unique routing set selects the arena automatically.
- Added bounded writable staging, tensor byte copies, global-to-slot ID mapping,
  padded route (-1) preservation, transfer failure invalidation, and memory
  accounting. Cache and arena memory is reserved before KV-pool sizing.
- Implemented the required `get_scaled_act_names()` delegation so the wrapper
  can be instantiated under the actual v0.5.21 base class.
- Bound cached quant methods to the runtime and prepared the same six-scale
  FlashInfer CUTLASS payload as the resident ModelOpt implementation. Decode and
  arena runners have their own expert capacities. Router scaling follows the
  resident path, and SwiGLU overrides use the resident materialization helper.
- Fixed LFU decay to age all residents; cache capacity failure is checked before
  mutating mappings. Main-model checkpoint filtering now retains MTP experts.
- Startup rejects non-single-rank TP/DP/EP/PP, nonstandard A2A, and speculative
  decoding. Loader rejects devices other than SM120.
- Saved public README/config/quantization metadata, pinned to revision
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`. Public shape and per-layer NVFP4
  declarations agree with the implementation. No model weights were downloaded.

### Validation

The CPU suite covers pack export/build/mmap round trips, hit/eviction copies,
fixed storage addresses, cross-layer arena reuse, weighted route selection,
padded/invalid IDs, disabled-arena overflow, transfer/kernel failures,
reservation, memory accounting, policy aging and public metadata.

Host CPU environment: 23 tests passed, zero skips (CPU PyTorch/NumPy installed
in a temporary virtual environment). See `tools/qwen38_cache_tests/README.md`
for the configurable offline container entry point and current image command.
Container results are recorded in `flash-next-plan/cpu-test-result.json`.

### Remaining GPU Work

The synchronous CUDA branch and CUTLASS invocation are implemented but are
**not GPU-validated**. The current runtime synchronizes routing to CPU and waits
for CUDA compute before buffer reuse. It is a reference implementation, not
an asynchronous production path.

Still required on SM120 with real weights:

1. Create a resident ModelOpt layer from checkpoint shards, invoke the original
   post-load transform, export v2 tensors, and verify the actual runtime layout.
2. Compare cached CUTLASS outputs against fully resident ModelOpt for fixed
   routes, decode misses and arena prefill. CPU route parity is not NVFP4 parity.
3. Validate full server loading/generation and actual cache + PLE + FP8 KV memory.
4. Implement JIT/device routing, transfer stream, staging ring and ready/last-use
   events after synchronous numerical parity.
5. Validate chunked 8K/32K prefill, 128K/220K contexts and MTP; speculation remains
   disabled until its separate loader and verify-window behavior are checked.

Relevant new continuation files:

```text
python/sglang/srt/layers/moe/qwen38_cache_runtime.py
python/sglang/srt/model_loader/qwen38_nvfp4_pack/export.py
tools/qwen38_cache_tests/test_cpu_runtime.py
tools/qwen38_cache_tests/docker_cpu.sh
tools/qwen38_cache_tests/run_container_cpu.py
flash-next-plan/model-metadata/
```

## Patch 2: Experimental GPU Continuation (2026-10-09)

- Fixed the real Qwen FusedMoE prefix (`.mlp.experts`), verified with the actual
  FusedMoE class and public ModelOpt config in the container.
- Added a local-only, single-layer-at-a-time checkpoint converter, complete
  projection/scale index validation, resume provenance checks, pack assembly and
  optional fixed-route resident CUTLASS comparisons for each exported layer.
- Added CUDA JIT route remap and a 513-int histogram/invalid-route summary.
  The SM120 extension compiled and loaded in the CPU-only v0.5.21 container.
  No CUDA kernel was executed.
- Added opt-in `runtime_mode=async`: bounded pinned staging ring, H2D stream,
  staging completion events, compute-ready waits, coarse last-use protection
  across streams and explicit synchronization for cleanup.
- Added CUDA storage/event tests, synthetic real NVFP4 loading/transform/parity,
  configurable offline Docker entry points and a full-model/long-context canary.
- Added strict consumer weight/alpha/view layout checks before allocation,
  zero-filled cache storage, PLE host offload and single-request resolution.

The second patch is **GPU-unvalidated**. CPU regression, module imports and JIT
cross-compilation have passed; actual conversion, H2D events, numerical parity,
server startup and long-context performance require the target machine. The
async policy still synchronizes a small routing summary to the host. MTP and
CUDA graphs remain disabled. See `tools/qwen38_cache_tests/RUN_ON_GPU.md`.

Patch 1 preserves the earlier CPU-tested snapshot. It includes the previous
agent's implementation but has no model-startup guarantee; patch 2 includes the
prefix audit fix. For GPU testing apply both patches in order.

### Device-policy continuation

Added opt-in `runtime_mode=device`: GPU decay-LFU/LRU decisions, expert-to-slot
map, generations, parallel routing histogram and final ID remap. Host reads
only the bounded transfer/active-slot plan; a per-layer host wait still exists.
Decode ready/last-use events are tracked per slot; prefill arena has separate
events. Metadata uses a separate stream, failures poison the runtime, and
capture is explicitly rejected. Added GPU CPU-policy trace comparisons and
CUTLASS parity, and strengthened the parity trace to force real cache eviction.
Both CUDA routing and planner extensions cross-compiled/loaded for SM120;
actual execution remains GPU-unvalidated. CUDA Graph and completely host-free
miss transport remain future work, as described in RUN_ON_GPU.md.

### Background transport continuation

Added opt-in `runtime_mode=threaded`: main thread enqueues CUDA policy/remap,
D2H transfer plans, a compute-side ready-event wait, and compute; a background
worker dispatches mmap/staging/H2D transfers and reports readiness through a
per-job CUDA event plus a host flag; a separate compute-retirement thread waits
for completion before releasing job capacity. No per-layer plan readback runs on
the main inference thread. Queue backpressure and startup/cleanup still wait.
Conservative token-count bounds choose cache vs arena without GPU readback.
Transport failures raise into the caller and poison the runtime so partially
copied entries cannot become hits on retry. Added two CPU worker protocol tests
and target-GPU threaded storage/parity tests. The initial design gated compute
with driver MemOps (cuStreamWaitValue32/cuStreamWriteValue32): on real GPUs
those cross-thread driver calls wedged the transfer stream until timeout, while
every configuration without them passed, so the gate was replaced by
semantically equivalent pure PyTorch events (`QWEN38_READY_TIMEOUT_SECONDS`,
default 120 s, bounds the readiness wait). Graph replay is unsupported.

### Archive preflight

Added host Docker supervision with configurable deadlines, live logs, scoped
container removal and JSON results. GPU reports preserve skip/failure details,
and required-SM120 mode fails on skips. GPU fixtures now register failure-path
cleanup for worker lifetimes. CPU tests/container imports passed; supervisor
success/timeout tests and a real Docker smoke job passed. GPU behavior remains
unvalidated.

### Activity watchdog refinement

GPU small jobs now have a 5-minute hard cap. Model jobs use a 60-second
continuous quiet-log / unchanged raw cumulative CPU/I/O / sampled GPU 0% heuristic, with at least 30s confirmation
rather than a default 12-hour cap. Missing metrics do not count as inactivity.
The supervisor monitors redirected server.log as well as stdout and records
activity samples. This detects suspected idle stalls, not busy deadlocks.


### Script and archive review

Current CPU regression: 57 tests plus container imports/config/runner checks.
Fixed missing-I/O inactivity handling, cleanup failure exit codes, SIGTERM
cleanup, frozen queued configuration, recovery locking and saved Docker
identity cleanup. Canary now waits for /ready and preserves its report on
cleanup errors. Current manuals use the watchdog wrappers; historical scaffold
entry points are retired. GPU execution and same-session wake integration
remain target-machine validation work.


Watchdog hardening: automatic kill remains default; 60s observation plus 30s
confirmation requires unchanged raw CPU/I/O counters and zero sampled GPU.
Logs, any work, unknown metrics and sampling gaps reset observation. Partial
stdout counts as activity. Tests cover bursty work, resets, missing accounting,
confirmation recovery and default kill/warn lifecycle. Local Docker raw CPU
stats were readable, but cumulative block IO was absent; idle kill correctly
remains unavailable in that case. GPU telemetry is unvalidated.
