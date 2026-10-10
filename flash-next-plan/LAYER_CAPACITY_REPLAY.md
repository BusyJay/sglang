# Per-layer capacity and offline decode replay

Date: 2026-10-11. Third optimization stage, building on bulk transport and optional
arena warmup. GPU validation remains pending; no weights were downloaded.

## Runtime

`layer_cache_slots` is an optional JSON list with exactly one integer per layer.
The production loader requires each value in [10, 512]. Omitting the list retains
the uniform `cache_slots_per_layer` setting (default 56).

Every layer now uses its own capacity for cache allocation, CPU/CUDA policy,
transfer-plan parsing, conservative threaded arena selection, CUTLASS runner
capacity, and warm seed size. The shared full arena is unchanged. Memory budgeting
uses the sum of actual per-layer capacities; the shared host plan is sized for the
largest layer. The existing CUTLASS boundary already keys runners by capacity.

Example loader extra config fragment:

```json
{"cache_slots_per_layer": 112, "layer_cache_slots": [96, 128]}
```

The two-element example is for a two-layer fixture; real Qwen3.8 requires 48
entries. A larger individual layer is not permission to increase total GPU memory:
keep the sum within the previously measured budget and verify startup headroom.

Canary accepts `--layer-cache-slots capacities.json`, where the file contains only
the list. Do not pass the full replay report as that file.

## Profiling-only route capture

Use `--route-trace-path /mounted/results/routes.jsonl --route-trace-limit 10000`
on the canary, or the equivalent loader options. The directory must already exist
and be writable **inside the server/container**. Use a new path for each run: the
runtime appends JSONL, and the limit applies per runtime instance.

Each record contains `layer` and a `routes` matrix. Capture is opt-in and copies
GPU route IDs synchronously to CPU, with host file writes. It is deliberately
unsuitable for throughput/latency benchmarking. No prompt text is recorded.
Prefill records may be large and consume the invocation limit; size the limit to
include the intended decode window. Capture multiple representative workloads.

## Replay and allocation

```bash
python3 -m tools.qwen38_cache_tests.replay_cache \
  --trace results/routes.jsonl --output results/allocation.json \
  --layers 48 --experts 512 --capacities 56 80 96 112 128 \
  --budget 5376
```

Default replay includes only one-token invocations, skips larger prefill/batch
records, drops -1 padding and processes unique expert IDs in ascending order,
matching device planner visitation. Every layer must have decode samples. For
multi-request decode set `--max-tokens` deliberately; this filter does not infer
SGLang forward mode from the matrix alone.

The tool reuses the current CPU decay-LFU reference, builds each layer's miss
curve, and uses dynamic programming to find minimum replay misses over the supplied
candidate capacities with total slots **at most** the budget. It also reports
uniform-capacity miss totals and per-layer sample counts. A candidate that cannot
hold a traced active set is rejected rather than silently modeling arena traffic.

Assumptions/limits:

- Equal bytes per slot across layers, enforced by current runtime layout checks.
- Cold-start decode replay, no prefill warming, persistent pins or oracle eviction.
- CPU floating point may differ slightly from CUDA float32 policy tie decisions.
- Candidate-grid optimum, not an optimum over every capacity or GPU latency.
- Optimizing a captured workload can overfit. Use held-out workload captures and
  validate the resulting allocation against the uniform baseline on GPU with
  trace disabled. Report H2D bytes and latency, not only replay hit rate.

## Validation

Added no-torch tests for capacity validation, trace filtering, replay cold/hit
counts, and exact allocation against exhaustive search including non-concave miss
curves. Real-tensor CPU tests cover heterogeneous arena/cache selection, memory
budget, and bounded trace output (skipped locally because PyTorch is unavailable).
GPU synthetic suite adds device/threaded heterogeneous capacities, including
warmup and cross-stream all-six-tensor parity.

Local suite: **85 tests, 72 passed, 13 skipped** (PyTorch unavailable). AST syntax
checks and `git diff --check` pass. Synthetic GPU and real-checkpoint performance
results are pending.
