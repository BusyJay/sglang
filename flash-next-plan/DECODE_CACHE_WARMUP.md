# Optional arena-to-decode cache warmup

Date: 2026-10-11. Builds on the bulk transport work in the same worktree.

## Implementation

Enable `warm_decode_cache: true` in the cache loader JSON, or pass
`--warm-decode-cache` to `full_model_canary.py`. Supported only in device/threaded
mode with the prefill arena. Default is **off**, pending GPU performance evidence.

For each arena invocation, select up to `slots_per_layer` experts from that chunk's
positive route histogram, descending by frequency with expert-ID tie breaking.
Only the chunk being processed is considered, not the entire request history.
Thus later chunks replace earlier seed sets. Already-resident entries are not
preserved; this intentionally simple heuristic is a baseline for GPU comparison.

The six transformed expert tensors are gathered directly from the GPU arena into
the existing decode-cache allocations with `torch.index_select(..., out=...)`.
No CPU expert-weight read, H2D expert-weight copy, extra full weight tensor, or
pack-format change is required. Small index/metadata tensors are temporary CUDA
allocations; this is not a CUDA Graph path.

The seed operation clears lookup/keys and initializes selected slots, ages,
generation and clock coherently. Frequencies start at one invocation, **not** at
raw prompt counts, preventing long prompts from effectively pinning seed experts.

## Ordering

- Seed overwrite waits for previous decode slot readers.
- Arena fill, seed weight gathering, and policy writes use the transfer stream.
- A distinct warm-ready event is recorded after weight and policy writes.
- Subsequent metadata/routing work for that layer waits for warm-ready.
- The caller receives arena readiness only after warmup is enqueued. The shared
  arena cannot be replaced before the seed reads and current MoE compute finish.
- The threaded worker reads a bounded expert-count histogram (513 int32 values
  for the real model), rather than only the invalid-route counter, when enabled.
- Failures enter existing device poisoning / threaded worker-error paths; partial
  seeds are never advertised as a successful ready event.

Counters: `warmup_runs`, `warmed_experts`, `warmup_d2d_bytes`. Warmup does not count
as a decode hit or miss and does not increment H2D transferred bytes.

## Tradeoff and validation

This aims to reduce cold decode misses, not guarantee a higher steady-state hit
rate. Prompt-hot and decode-hot experts may differ. Copying 112 experts per layer
adds meaningful D2D traffic; this first implementation deliberately includes
warmup in readiness, so it can increase TTFT. Compare total request latency and
first-token/decode timing, not hit rate alone. Multi-chunk prompts repeat warmup.

CPU tests validate selection, deterministic ties, small/empty active sets, actual
warmup dispatch/policy initialization through mocks, worker publication order,
and warmup failure propagation. Local full suite: **78 tests, 67 passed, 11 skipped**
because PyTorch is unavailable. No model weights were downloaded.

Added synthetic GPU cases (pending execution):

- all-six-tensor parity with warmup across independent streams in device/threaded;
- warm layer 0, replace arena with layer 1, then verify layer 0's seeded decode
  uses correct weights with **no additional H2D bytes or misses**;
- existing bulk transport and synthetic NVFP4 CUTLASS cases remain in the suite.

Run on an already configured SM120 host, without model weights:

```bash
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output results/warmup-gpu.json
```

Then compare the same local-weights canary with and without `--warm-decode-cache`,
keeping slots, staging size, context, chunk size and completion length fixed.
Measure TTFT, early/steady decode throughput, total latency, H2D bytes and D2D bytes
for real prompts and multi-chunk inputs. Keep the feature disabled if its total
cost exceeds avoided misses. GPU correctness and speedup are not claimed locally.

Remaining optimization directions: per-layer capacity allocation from routing
replay, tail/profile-informed seeds, and fused decode route/plan/remap. These are
not implemented here.
