# Smaller expert pools and lower-overhead adaptation

This experiment keeps LFU as the default and adds GPU SLRU, reusable pinned
transfer plans, compact route sampling in the existing plan D2H, and a standalone
CPU analysis process. Continue using the FlashInfer per-capacity runner patch.
GPU execution and throughput must be validated on the target machine.

## Suggested comparisons

Keep prompts, decode length, concurrency, graph settings, warmup, and memory
settings identical. Disable `route_trace_path` when measuring speed. Measure both
the sampling window and the unsampled steady-state interval. Record decode
tokens/s, per-layer hit/miss deltas, transferred bytes, and rebalance pauses.

| Experiment | Canary options added to the existing command |
| --- | --- |
| Original capacity, LFU baseline | `--runtime-mode threaded --slots 112 --cache-policy lfu` |
| Sampling overhead only | Above plus `--adaptive-cache --adaptive-cache-mode sample` |
| Sampling and process analysis, fixed layout | Above plus `--adaptive-cache --adaptive-cache-mode observe` |
| Full allocator | Above plus `--adaptive-cache --adaptive-cache-mode rebalance` |
| Smaller static LFU | `--runtime-mode threaded --slots 96 --cache-policy lfu` |
| Smaller static SLRU | `--runtime-mode threaded --slots 96 --cache-policy slru --cache-protected-fraction 0.7` |
| Smaller adaptive SLRU | Above plus `--adaptive-cache` |

For SLRU also try protected fractions 0.5 and 0.8. Reducing uniform slots from
112 to 96 saves approximately 2 GiB of expert weights across 48 layers. SLRU is
an experimental alternative; workload-dependent routing can favor LFU.

The GPU parity tests are in `tools/qwen38_cache_tests/test_gpu_cache.py`. Run
the normal small GPU suite before full-model benchmarks. In particular the new
tests compare CPU and GPU SLRU replacement decisions, exported original expert
IDs, and all six expert tensors across eviction and a capacity migration.

## Loader configuration

Static smaller pool, with no analysis process:

```json
{
  "cache_slots_per_layer": 96,
  "runtime_mode": "threaded",
  "cache_policy": "slru",
  "cache_protected_fraction": 0.7,
  "adaptive_cache": false
}
```

Optional online adaptation:

```json
{
  "cache_slots_per_layer": 96,
  "runtime_mode": "threaded",
  "cache_policy": "slru",
  "cache_protected_fraction": 0.7,
  "adaptive_cache": true,
  "adaptive_cache_window": 512,
  "adaptive_cache_interval": 4096,
  "adaptive_cache_mode": "rebalance"
}
```

`adaptive_cache_interval` counts **unsampled** single-token decode calls per
layer between contiguous windows. Zero samples continuously. The first window
starts immediately; defaults collect 512 calls then skip 4096. Prefill/batched
calls break the observation sequence. A window with dropped samples is discarded.
Analysis warms shadow policies with the first min(128, window/4) calls and scores
the remaining contiguous calls. Candidates restart between windows because the
intervening routes were not observed. Cold-start estimates and workload changes
can still bias the recommendation. The default cadence replaces the original
per-token shadow-policy replay and its persistent two-window cooldown.

`sample` collects windows but does not start the analysis subprocess. `observe`
evaluates and logs proposals without applying them. `rebalance` applies proposals.
The process imports neither torch nor the SGLang package; transfer/compute remain
in the serving process. The serving coordinator serializes one complete window
at a time, so this removes continuous GIL contention rather than all host work.

SLRU uses the existing per-slot frequency storage as a segment flag. A miss
enters probation, a hit promotes to protected, and overflow demotes the oldest
protected resident. Eviction prefers the oldest inactive probation resident;
if every probation resident is active, an inactive protected resident may be
evicted. All active experts remain protected from overwrite. Prefill warmup seeds
probation. Shrinking the layout trims protected membership to the new bound.
The policy is supported in the threaded runtime only.

Online allocation retains the pool allocated at startup; it does **not** return
unused weight storage to the KV allocator. To save memory, start with fewer slots
or a smaller explicit layout. Rebalancing remains globally synchronized and can
cause a pause. New capacities can also incur first-use runner initialization.

## Offline minimum-size layout

Capture a representative trace separately using the existing route trace options.
Trace capture synchronizes CUDA and must not be enabled for speed measurements.
Then, from the repository root:

```bash
python3 -m tools.qwen38_cache_tests.replay_cache \
  --trace /absolute/path/routes.jsonl \
  --output /absolute/path/slru-report.json \
  --layout-output /absolute/path/slru-layout.json \
  --cache-policy slru --protected-fraction 0.7 \
  --capacities 64 80 96 112 128 \
  --reference-slots 112 --max-extra-misses-per-token 2
```

This finds the smallest candidate-grid layout with no more than two additional
expert misses per model token relative to uniform 112-slot **LFU** on the same
trace. It requires equal single-token sample counts in every layer. The report
includes reference misses, extra misses/token, capacity curves and used slots.
Pass the layout with `--layer-cache-slots /absolute/path/slru-layout.json`, or
set `layer_cache_slots` in loader JSON. Use LFU in the replay command to obtain
an LFU layout. Existing fixed-budget `--budget` allocation remains supported.

The estimate is cold-start replay, not a latency model. First measure how much
extra decode time a few extra misses cost on the target system, then select an
allowance matching the desired throughput loss. Validate on held-out prompts.

## Validation

CPU unit tests cover SLRU scan resistance, promotion/demotion, active-set safety,
window boundaries, standalone analysis and minimum-slot allocation. The local
dependency container supports compilation for SM120 without a GPU. A successful
compile verifies the binding/build, not GPU numerical parity or throughput.
