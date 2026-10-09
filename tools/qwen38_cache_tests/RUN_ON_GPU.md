# Apply and test on a GPU host

Both patches target SGLang commit
`e00930c5489053f26d86b179cee0d087f846acbb` (`v0.5.21`).

```bash
git apply --check /path/to/patches/01-cpu-validated.patch
git apply /path/to/patches/01-cpu-validated.patch
git apply --check /path/to/patches/02-gpu-unvalidated.patch
git apply /path/to/patches/02-gpu-unvalidated.patch
```

Patch 1 is the saved CPU-tested reference, including the prior agent's changes.
Its CPU results do not prove model startup. Patch 2 is incremental and includes
subsequent audit fixes, conversion, async runtime and GPU launch/test tooling.
Use both for GPU testing. Do not apply them to the development worktree again.

Choose the **local** image and device explicitly; no script assumes a tag:

```bash
export SGLANG_IMAGE='your-local-sglang-v0.5.21-image'
export QWEN38_GPU_DEVICE=0
```

The GPU scripts use offline flags, `--pull never`, `--network none`, and preserve
`HF_HOME`. They require the installed image to contain all Python/CUDA/FlashInfer
dependencies, including any FlashInfer kernel cache needed for runtime. If a
kernel dependency is absent, prepare the image beforehand; these scripts do not
fetch it. Local checkpoint paths must already contain configuration, tokenizer,
index and shards. These direct test wrappers do not stop an existing GPU service. The optional
GPU-switch workflow below handles stop/test/restore/wake.
Ensure the selected GPU is available before starting these tests.

## 1. CPU regression and JIT compilation

```bash
tools/qwen38_cache_tests/docker_cpu.sh "$SGLANG_IMAGE"
```

The repository's JIT loader can compile the routing op without a GPU:

```bash
docker run --rm --pull never --network none \
  -v "$PWD:/workspace/sglang:ro" -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 "$SGLANG_IMAGE" tools/qwen38_cache_tests/compile_cuda.py
```

## 2. Small GPU tests

For CUDA H2D/cache/arena/event/JIT routing tests on SM89 or SM120:

```bash
tools/qwen38_cache_tests/docker_gpu.sh \
  --image "$SGLANG_IMAGE" --device "$QWEN38_GPU_DEVICE" \
  --results /absolute/results/gpu-small
```

On SM120, the same suite also runs synthetic NVFP4 checkpoint shards through
real FusedMoE weight loading and the original ModelOpt transform, exports a
pack, and compares sync/async/device/threaded cached CUTLASS against resident CUTLASS. To demand
SM120 rather than silently skip those comparisons:

```bash
tools/qwen38_cache_tests/docker_gpu.sh \
  --image "$SGLANG_IMAGE" --device "$QWEN38_GPU_DEVICE" \
  --results /absolute/results/sm120 -- --require-sm120
```

These are tiny fixtures, not a model. Memory checks require 1–2 GiB free and
assert less than 1 GiB allocated peak. SM89 cannot validate the SM120 CUTLASS
pack, but can run generic transfer/event and routing tests. `gpu-small.json`
reports skips; an all-skipped run exits with failure.

## 3. Convert local checkpoint, first one layer

Use a fresh output directory and the revision that identifies your local
checkpoint. The checked public revision is
`fc694b54fb0174e0913e6adf86691ef85a4ead47`.

```bash
tools/qwen38_cache_tests/docker_model.sh \
  --image "$SGLANG_IMAGE" --device "$QWEN38_GPU_DEVICE" \
  --checkpoint /absolute/checkpoint --pack /absolute/pack-one-layer \
  --results /absolute/results/convert -- convert \
  --checkpoint-revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --sglang-commit e00930c5489053f26d86b179cee0d087f846acbb \
  --layers 0 --parity --runtime-mode sync
```

The converter reads only selected layers' routed expert tensors. It validates
all 12 projection/suffix entries per expert before allocating a layer, calls
SGLang's real weight loader and post-load transform, and optionally performs
fixed-route comparisons. Inspect `layer-000/parity.json`.

Repeat in a different fresh directory with `--runtime-mode async` for event/JIT
parity. Keep sync as the baseline until both pass.

For all 48 layers and a server-ready manifest, omit `--layers` and add
`--assemble`. `--parity` checks each layer before release. Expect roughly a
complete transformed export plus a complete assembled pack on disk, and one
resident layer plus one-layer parity cache/arena in GPU memory. The converter
never creates the full model. `--resume` checks matching conversion provenance
and tensor file sizes; it cannot skip a requested parity run. Remove/recreate
an incomplete layer directory before retrying that layer.

## 4. Full-model canary and long contexts

After SM120 layer parity passes, use the assembled pack directory:

```bash
tools/qwen38_cache_tests/docker_model.sh \
  --image "$SGLANG_IMAGE" --device "$QWEN38_GPU_DEVICE" \
  --checkpoint /absolute/checkpoint --pack /absolute/assembled-pack \
  --results /absolute/results/canary-sync -- canary \
  --runtime-mode sync --contexts 8192 32768
```

The canary starts its own localhost server, sends deterministic generation
requests, saves `server.log`, per-context JSON and `report.json`, and shuts down
only the process group it created. It forces text-only, TP/DP/EP/PP=1, no graph,
no speculation, one active request, FP8 KV and chunked prefill. The expert cache
and arena are reserved before KV allocation; PLE is explicitly offloaded to
host memory. The pinned PLE table is separate from the expert staging ring.

After short contexts pass, run `--contexts 131072 225248`; 32 output tokens fit
within the default 225280 context limit. This tests generation completion and
memory capacity, not semantic recall quality. Try `--runtime-mode async` only
following sync validation. Saved server info and logs help distinguish startup,
KV memory, attention/QSA, PLE and expert-cache failures.

## Known boundaries

- `runtime_mode=sync` is the default. `async` uses a bounded pinned ring and a
  transfer stream, ready/last-use events, JIT ID remap, and a 513-int D2H routing
  summary. CPU cache policy still synchronizes that summary once per layer.
- Last-use protection currently serializes runtime invocations across streams;
  this limitation applies to `async`. The `device` and `threaded` modes below
  implement CUDA cache policy and per-slot lifetimes. CUDA Graph remains unsupported.
- Actual checkpoint conversion, CUDA copies/events, CUTLASS numerics, full model
  startup, FP8 KV and long contexts have **not run** on this development machine.
  CPU fixtures and cross-compilation do not prove any of those results.
- MTP remains rejected. The draft loader/pack coverage and PLE speculation
  constraints require a separate end-to-end design and GPU validation before
  enabling NEXTN. Vision and generic parallel/offload modes remain outside scope.
- A pack contains backend-specific transformed bytes; rebuild for changed
  SGLang/FlashInfer/runtime layout. The manifest stores revision identifiers, not
  a cryptographic proof that local checkpoint files match those identifiers.

## Device cache policy and per-slot pipeline

`--runtime-mode device` is now available in conversion parity and canary.
Run the same fresh-output conversion with `--parity --runtime-mode device`,
then repeat the canary with `--runtime-mode device` after sync parity succeeds.
GPU tests include CUDA policy vs CPU decay-LFU/LRU trace comparisons, per-slot
transfer/arena tests across compute streams, and real CUTLASS parity. Numerical
parity now explicitly fills past cache capacity before testing eviction.

The device policy owns lookup, keys, frequency, ages, generation and routing
remap on CUDA. Host dispatch reads a bounded plan (status/counts/miss pairs/
active slots) to schedule mmap-to-pinned-ring-to-GPU copies. Each decode slot
has ready and last-use events; transfers wait only on the overwritten slot's
last reader, while compute waits on the required slots. Shared prefill arena
has its own ready/last-use events. A failed operation poisons this runtime so
partially copied entries cannot become hits on retry.

A CPU synchronization for the plan remains because ordinary mmap expert bytes
are moved through host-issued CUDA copies. The CUDA planner runs a bounded
single-thread LFU/LRU decision after the parallel histogram; it is not tuned.
Host event orchestration and metadata summary transfer remain eager. The
older limitations above about CPU cache *decisions* and whole-runtime last-use
serialization apply to `async`, not `device`.

CUDA Graph is still disabled. Capturing host-generated transfer decisions would
freeze one invocation's plan. A safe all-hit graph needs dynamic ID remapping,
a residency/overflow guard before replay, and stable pinned slot generations
for the captured compute; misses must execute eagerly or use a separately
designed device transport. Device mode explicitly rejects capture. Neither
all-hit graph integration nor graph miss fallback is implemented yet.

## Background transport without per-layer main-thread CUDA synchronization

Use `--runtime-mode threaded` for conversion parity and the canary. This mode
requires CUDA driver stream memory operations (cuStreamWaitValue32/v2 and
cuStreamWriteValue32/v2). Startup probes them and fails if unavailable; there is
no silent synchronous fallback. GPU tests now include threaded transport and
CUTLASS parity. Start with one-layer `--parity --runtime-mode threaded`.

The caller queues device policy/D2H plan work, a stream memory wait, a failure
guard, and CUTLASS compute. A background thread waits for plans, stages mmap
bytes into the bounded pinned ring, dispatches H2D and writes the ready signal.
A separate retirement thread waits for compute completion and retains gate/
plan storage until all consumers finish. The caller does not read the GPU plan
or wait on per-layer CUDA events. Queue capacity is bounded (64 jobs), so a
producer that outruns transfer experiences backpressure; startup and explicit
close/synchronize also wait. Host allocation, Python launch and backend internals
may still add latency. This is not a guarantee of zero host stalls.

Since runner capacity is a host choice, routing with `tokens * top_k > slots`
uses the full arena conservatively, even if its actual unique set might fit.
Other routing uses CUDA policy. One layer's metadata follows its previous compute
completion to keep policy/workspace lifetimes valid. Decode slot reuse and the
shared arena follow last-use events. Stream memory waits consume no SM spin
kernel; the worker transfers weights then releases compute with a barrier.
Transfer failures release gates with an error value; a CUDA guard asserts before
MoE reads invalid bytes. A fatal driver/context error can prevent a wait from
being released, so failed target-machine tests need process-level timeout and
termination (as provided by the canary supervisor).

Ordinary mmap still requires CPU transport. “Asynchronous” here means no
per-layer CUDA synchronization on the main inference thread, not no CPU worker.
Neither CUDA stream memory operations nor this transport have executed on the
development host. CPU worker/retirement protocol tests and SM120 guard/planner
cross-compilation passed. CUDA Graph integration remains unsupported: replay
would not enqueue background jobs or create fresh transfer gates.

## Timeouts, logs and failure cleanup

GPU/model Docker wrappers are supervised on the host using Python 3. GPU small
jobs default to a 300-second hard deadline; conversion/canary jobs default to
no total deadline and use the activity watchdog instead.
Set wrapper `--timeout SECONDS` (before `--`) or `QWEN38_TEST_TIMEOUT_SECONDS`.
Each job creates a unique container name, saves a live output log and
`docker-job.json`, and forcibly removes only that container on timeout/interrupt.
Timeout exits with 124; SIGTERM/SIGINT trigger cleanup and exit with 130. A cleanup error makes an otherwise successful job fail; check the named container
manually if the Docker daemon itself fails. No existing GPU service is stopped.

`gpu-small.json` includes individual skip reasons and failure/error tracebacks.
`--require-sm120` treats any skipped test as failure, so a resource skip cannot
be mistaken for completed SM120 validation. SM89 runs may skip CUTLASS cases.
Use a distinct results directory per run to retain summary history.

## Activity watchdog (archive defaults)

The supervisor polls every 5 seconds. Automatic kill is the default
(`--stall-action kill` / `QWEN38_STALL_ACTION=kill`). At 60 seconds of observed
inactivity it saves diagnostics; it continues observing for at least 30 more
seconds before killing only its own container. Thus the default idle kill is
about 90 seconds plus sampling overhead, not a single quiet-minute decision.

Kill requires unchanged **raw cumulative container CPU time and block-I/O byte
counters**, zero sampled selected-GPU utilization, and quiet output throughout.
Docker's rounded CPU percent/BlockIO strings are diagnostic only. CPU work
between polls increases cumulative time and resets the window. Any new log,
I/O or CPU progress, counter reset, unknown/non-finite metric, or excessive
sampling gap or stale/repeated Docker stats snapshot restarts observation and confirmation. Partial stdout without a
newline counts as activity; wrappers set PYTHONUNBUFFERED=1 and the canary's
redirected server.log is watched. Supervisor diagnostic messages do not count
as test progress.

Raw stats use the Docker CLI's local Unix-socket endpoint. Remote endpoints,
missing permissions or missing cumulative I/O accounting are **unknown**, not
zero; idle kill is disabled until all required metrics are available. This
host's Docker returned no cumulative block-I/O counters during the smoke test.
Check activity.json on the target host. The hard timeout still applies when
metrics are unknown. Do not claim automatic idle detection passed from a run
with unknown metrics.

Use `--idle-seconds SECONDS`, `--confirm-seconds SECONDS` (minimum 30) and
`--stall-action warn` before the wrapper `--`; corresponding environment
variables are QWEN38_IDLE_SECONDS, QWEN38_CONFIRM_SECONDS and
QWEN38_STALL_ACTION. Warn mode records observations without killing.
Small GPU tests retain their 300-second **hard timeout regardless of activity**;
this produces status=timeout, distinct from suspected_stall. Cold JIT may exceed
that budget: prepare the image/kernel cache or explicitly raise --timeout for
that run. Model jobs have no wrapper hard timeout by default, but canary has
startup/request deadlines. Set --timeout to bound model jobs when desired.

The evidence remains a heuristic: GPU utilization is sampled and device-wide,
busy deadlocks are not detected, and external I/O waits can resemble inactivity.
Diagnostics (container state/process snapshot, activity.json and history) are
saved before kill; keep them to diagnose any false positive. Do not silently
increase thresholds or retry a suspected_stall without examining the evidence.


## Fire-and-forget with a regular SGLang Docker service

Use `flash-next-plan/sglang-gpu-switch-workflow/config.example.json` and its
README for the external persistent supervisor. Set the local test image,
regular container name, health URL, workset paths and an actual same-session
wake command. The wake executable is a machine-specific integration, not
included in the archive. Start the supervisor once from a normal terminal.
Codex submits `dispatch`, confirms acceptance and ends its turn. The supervisor
then stops the regular container, runs these watchdog-protected wrappers,
cleans the test container, starts the regular container and waits for
`/health_generate`. It publishes `latest-test.json` before invoking the wake
command with zero additional delay. Test failure/stall follows the same restore
and wake path; failed restoration records an error and does not invoke wake.
Keep the outer `batch_timeout_seconds` at 0 so cleanup remains owned by the
inner watchdog. Use `docker_model.sh` as the batch command for model jobs.


Before model tests, mount complete local tokenizer/config/index/shard files.
For `--ple-backend file`, pass a container-visible writable directory, e.g.
`--ple-dir /workspace/results/ple`; a host path is not visible unless mounted.
Canary checks `/ready` before sending the first generation request. Test results
prove execution and nonempty generation only, not semantic quality or speed.
