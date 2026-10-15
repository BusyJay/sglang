# Qwen3.8 decode transport

The `threaded` runtime enables `decode_device_gate` and
`decode_hit_fast_path` by default. Set `"decode_device_gate": false` in
`--model-loader-extra-config` to use the inline D2H path; also set
`"decode_hit_fast_path": false` to compare with the original worker path.

For single-token decode outside adaptive observation windows:

- The GPU planner updates cache policy and remaps routed expert IDs. Hits need
  no plan D2H, CPU confirmation, staging job, or retirement job.
- Misses publish a plan into mapped pinned memory. A native CPU poll releases
  the GIL; the worker copies expert bytes into a shared pinned payload and
  publishes progress after each expert.
- A pre-enqueued GPU kernel waits for each missing expert and reads its payload
  into the cache. The background staging worker submits no CUDA operations,
  allowing synchronous host reads to finish without a CUDA context-lock cycle.
- GPU completion events protect mailbox/payload reuse across layers and streams.
  Cache migration drains readers and refreshes destination descriptors.

Prefill, multi-token batches, adaptive observation windows, and one warm call
after each layer-capacity change use the ordinary transport. The default
observation window/interval is 512/4096 steps. Full CUDA Graph capture of this
runtime remains unsupported; segmented decode graphs keep MoE boundaries eager.

The mailbox uses one CPU polling core and an additional pinned payload for at
most one top-k expert set. `max_pending` bounds outstanding completion events.
`QWEN38_READY_TIMEOUT_SECONDS` defaults to 120; zero disables the readiness
deadline. Host failure flags and a bounded device wait prevent failed fills
from exposing incomplete expert weights.

## Profiling and metrics

CUDA profiler start/stop must first drain work with `torch.cuda.synchronize()`.
That binding releases the GIL, allowing Python staging to finish before entering
the GIL-holding cudart profiler binding. Otherwise Nsight/CUPTI synchronization
can deadlock against a GPU kernel waiting for Python staging.

Check `decode_device_gate_invocations`, `decode_device_gate_hits`,
`decode_device_gate_copy_calls`, and `decode_device_gate_copy_bytes` in cache
statistics. Device hit counts are collected at synchronization/logging boundaries.
`transferred_bytes` includes mapped-payload reads; `h2d_copy_calls` counts only
ordinary DMA submissions. In Nsight, `mailbox_fill_kernel` includes staging waits
and host-payload reads, so its duration is not pure compute time. Periodic
statistics D2H is expected even when per-layer plan D2H has disappeared.

GPU tests cover LFU/SLRU parity, cross-stream readers, prefill, sampling,
repartitioning, pageable host reads, default/disabled paths, and timeout failure.
The profiler-boundary CPU regression checks worker completion before cudart;
pending-miss start/stop was also exercised in three consecutive Nsight ranges
on a separate GPU. Production throughput comparisons require matching request
routes, cache budgets, and observation phases.
