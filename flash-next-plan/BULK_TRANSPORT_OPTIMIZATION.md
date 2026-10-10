# Bulk expert transport implementation

Date: 2026-10-11. Baseline: `3a740e7863`.

## Scope

The async/device/threaded runtimes now load the prefill arena by whole tensor,
split into bounded contiguous H2D chunks, rather than iterating over all experts.
Decode still uses the existing policy, slot mapping, and overwrite dependencies;
the six tensors of each missing expert share staging pages and completion events.
Their destinations remain separate: this change does **not** claim one H2D per
decode expert, nor less weight traffic.

- A configurable pinned ring replaces the expert-sized staging ring. Defaults:
  four buffers, up to 64 MiB each (256 MiB total). Each buffer is capped at the
  largest required whole tensor, so synthetic fixtures stay small.
- `staging_buffer_bytes=0` selects the former expert-tensor-sized buffer bound.
  This is a small-buffer comparison mode, not the old per-expert prefill algorithm.
- mmap bytes copy directly into a writable memoryview of the pinned buffer;
  the transient `bytearray` and its additional host copy are removed.
- Ring completion events are reused **only after CPU synchronization**. They
  are blocking events. Persistent slot/compute readiness uses distinct events,
  never a re-recorded ring event.
- Shared arena last-use tracking is maintained in the threaded worker, including
  same-layer reads, so changing transfer granularity preserves cross-stream safety.
- Poisoned threaded runtimes reject new work. Fatal CUDA failures still require
  process-level timeout/termination.
- `h2d_copy_calls` and `staging_batches` expose transport operation counts alongside
  transferred bytes. These count host dispatch, not successful GPU completion.

The sync reference runtime retains per-expert copying. No change to CUTLASS,
pack format, routing, cache capacity, GPU weight allocation, or scheduler.
Per-layer capacity allocation, arena-to-decode warming, hit/miss split execution,
and route/plan/remap fusion are **not** implemented in this change. Subsequent
optional arena-to-decode warmup is documented in [DECODE_CACHE_WARMUP.md](DECODE_CACHE_WARMUP.md).

## Expected operation count (not a measured speedup)

The old prefill path dispatches `512 * 6 = 3072` copies per layer. The new count
is approximately `sum(ceil(tensor_bytes / buffer_bytes))`, with possible extra
splits where tensors share a staging-page boundary. Event count scales with pages,
not experts. A roughly 1.34 GiB layer occupies about 22 pages at 64 MiB; H2D count
is of the same order, versus 3072. Total transferred bytes remain unchanged.

Decode retains separate destination copies, but removes intermediate bytearrays
and reduces six ring events per expert to roughly one staging-page event plus an
independent readiness event. GPU timing must establish actual performance gains.

## Validation

Local host has neither PyTorch nor CUDA. No weights were downloaded.

CPU tests exercise the actual transport methods through byte-buffer/CUDA-event
stand-ins: whole-tensor arena loads, uneven byte chunks, shared pages, ring wrap,
immutable ready events, mmap-view release, and partial dispatch failure. Worker
protocol tests additionally check arena last-use publication and poison rejection.

Run the complete CPU suite:

```bash
python3 tools/qwen38_cache_tests/run_cpu.py
```

Local result: 72 tests, 61 passed, 11 skipped because PyTorch is unavailable.
Python AST syntax checks and `git diff --check` also passed. The skipped tests
include the real-tensor CPU reference tests; the byte-buffer transport and worker
protocol tests ran, not merely source-text checks.

On a configured GPU host, run the synthetic suite (no checkpoint needed):

```bash
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output results/bulk-gpu.json
```

The suite checks all six expert tensors across independent streams, uses an
11-byte ring to split float32 storage at odd boundaries, and asserts the fixture's
arena copy/page counts. Existing synthetic NVFP4 CUTLASS parity exercises the new
prefill path in async/device/threaded modes.

Then run the existing local-weights canary, keeping checkpoint, slots, chunk,
contexts, and output length fixed. New flags:

```text
--staging-buffers 4 --staging-buffer-mib 64 --cache-stats-log-interval 1024
```

Compare 16/64/128 MiB buffers and baseline commit `3a740e7863`, recording TTFT,
decode latency, H2D count/bytes, CPU utilization, and Nsight Systems stream timeline.
Do not treat the last periodic stats line as a guaranteed final request snapshot.
GPU correctness and performance results are pending; none are claimed here.
