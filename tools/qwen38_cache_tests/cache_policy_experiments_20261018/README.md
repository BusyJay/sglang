# Reproducible cache-policy and fill-cost experiments

Local GPU reports: [eviction policies](LIVE_REPORT.md) and
[capacity allocation with real migration](CAPACITY_REPORT.md). The capacity
supplement fixes LFU, trains on a common prefix, migrates the actual uniform
cache, and measures held-out transport plus migration cost with five repeats.

Production behavior is unchanged. Three separate tools are supplied:
record target verify routes; replay policies on CPU with isolated fill-cost
estimates; replay actual planner/fill with the real pack on GPU. A fixed route
tape removes different free-generation/MTP acceptance trajectories from the
expert workload comparison. It does not reproduce the full serving model.

## Record on an isolated worker

Use one target rank and one request stream, with the normal MTP settings.
On the recording worker only, prepend the overlay to PYTHONPATH before starting
Python (adapt repository paths to your existing image):

```bash
export QWEN38_ROUTE_RECORD_PATH=/results/verify-routes.jsonl
export QWEN38_ROUTE_RECORD_SWEEPS=2048
export QWEN38_ROUTE_RECORD_ARM=/results/verify-routes.armed
export PYTHONPATH=/workspace/sglang/tools/qwen38_cache_tests/cache_policy_experiments_20261018/record_overlay:/workspace/sglang/python:/workspace/sglang
# Start the normal server using your existing launcher and send ONE request.
```

Use a fresh, initially absent arm path. After the isolated server is ready,
create that file immediately before sending the single recording request.
This excludes synthetic verify calls made during graph setup. The route tape
must still be absent before arming. Never expose this worker to business traffic.

The overlay wraps only the target threaded runtime entry point and records
`execution_phase=verify` before remapping. Original per-token ID order and
duplicates, layer order, sweep number and shape are retained. It does not record
router weights or model activations because this experiment executes transport,
not MoE math. Each completed sweep is published together; an incomplete final
sweep is absent. Existing output files are refused. Run a fresh recording worker
and output path per request; do not concatenate requests as continuous cache
history. Multiple target processes writing the same path fail. No production
module or launcher is edited. Remove the overlay and controls before any timing.
CPU route readback synchronizes CUDA and perturbs the recording worker's timing.

A `.meta.json` sidecar identifies geometry and recording limitations. Save the
request input IDs, output metadata/MTP acceptance, exact model/pack revision,
source commit, image ID, GPU/driver, server argv and SHA256 of manifest and tape
beside it. The tools hash tape and calibration/manifest inputs in their reports.
The route sequence is reproducible even if the recorded accepted-prefix state
is unavailable, but that is insufficient for full-model state replay.

## CPU algorithm and objective comparison

From the repository root (no torch/GPU needed):

```bash
python3 -m tools.qwen38_cache_tests.cache_cost_replay \
  --trace /results/verify-routes.jsonl \
  --fill-samples tools/qwen38_cache_tests/launch_fill_experiments_20261018/results/fill-prefill-overlap.json.gz \
  --budget 6390 --warmup 128 --output /results/cache-cost.json
```

For the existing request-marked tape, additionally pass `--case short`, `raft`,
`long-0` or `long-1`. The tool rejects mixed requests, partial/gapped sweeps,
unsupported phases and invalid IDs. New tapes have no case markers and require
no `--case`. Compare LFU, LRU (SLRU protected=0), SLRU protected=0.5/0.7/0.8.
Active experts cannot be evicted in any policy.

The same near-uniform layout uses all 6390 slots (133/134 per layer).
Candidate layouts are chosen on the first half after 64/128 warmup sweeps;
the second half is held out. Each layout replays the full training prefix at
that capacity before scoring validation, to avoid cold-start comparison.
Capacity objectives: sum of misses or sum of per-call calibrated fill latency.
The latter accounts for miss-count distributions, including all-hit launch
cost, rather than evaluating latency at the average miss count. Missing counts
inside the measured range use explicitly reported linear interpolation; outside
that range raises an error. `--variant mapped_adaptive` and
`--metric wall_ms_per_call` allow sensitivity studies; wall samples include
per-call synchronization and must not be treated as service timing.

Report fields include held-out misses/hit rate, all-hit call counts, miss-count
histogram, coalescible source+destination runs, per-layer results and estimated
fill cost. Estimated fill cost pools synthetic address layouts and excludes
real compute overlap, policy/planner cost, cache migration and unrecorded work.
It is neither measured model latency nor a throughput prediction. The tool does
not implement a new latency-aware eviction policy or change live adaptive.
Unsampled residual slots are not assigned fictitious gains; unused slots are
reported (these archived runs all use exactly 6390).

## Real-pack GPU transport replay

Stop the serving worker while running this standalone test to avoid memory and
GPU contention. It allocates the production-size expert cache. Use the existing
nightly image with source mounted and the usual PYTHONPATH (without overlay).
First run exact payload checks, then run separate timing processes:

```bash
python -m tools.qwen38_cache_tests.cache_transport_replay \
  --trace /results/verify-routes.jsonl --manifest /data/pack/manifest.json \
  --budget 6390 --warmup 128 --rounds 1 --correctness --correctness-sweeps 16 \
  --output /results/policy-parity
python -m tools.qwen38_cache_tests.cache_transport_replay \
  --trace /results/verify-routes.jsonl --manifest /data/pack/manifest.json \
  --budget 6390 --warmup 128 --rounds 3 --output /results/policy-timing
```

Each round randomizes candidate order with a recorded fixed seed and starts a
fresh process per policy. Each worker starts with an empty cache and executes
the identical warmup prefix, then measures the identical remaining route tape.
All policies retain actual GPU planner/remapping, fill and runtime events;
miss plans intentionally differ because they are the algorithm's outcome.
Timing synchronizes only at interval boundaries. Correctness mode compares
all routed payload tensors with the source pack and is excluded from timing.
It checks every layer and tensor for the first 16 sweeps after the complete
warmup prefix by default; raise `--correctness-sweeps` for a larger payload audit.
To compare layouts for one policy, use `--worker --policy lfu --layout layout.json`
with fresh output files; the declared budget must exactly match the layout.
Archive the per-run reports and compare wall time, miss counters and bytes.
Native mailbox counters may describe device fills separately from worker DMA;
inspect complete before/after snapshots rather than only transferred_bytes.

This is TRANSPORT ONLY: no B12X compute, shared expert, attention, GDN or draft
execution is timed. Prepared-launch optimization therefore cannot be evaluated
with this harness. No full-model graph segmentation or state restoration is
claimed. The transfer package initially contained CPU import/syntax checks only.
Local GPU results are now documented in [LIVE_REPORT.md](LIVE_REPORT.md). Cold compilation or
shape paths not touched by warmup can still enter the interval; inspect logs.

## First offline results on archived real verify routes

Trace: `verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz`, older
3-token NVFP4 run, with only target verify routes. Calibration: this week's
FP4-Local native fill microbench, mapped CTA8. This cross-run comparison is a
geometry-based preliminary estimate; re-record the current production workload
before selecting deployment settings. Four requests are a small correlated
sample and do not establish policy superiority across workloads.

Warmup 64 sweeps, train first half, score second half, total budget 6390.

| Request | Fixed LFU held-out misses | Fixed LRU vs LFU | Fixed SLRU(0.8) vs LFU | LFU allocated by training misses vs fixed |
|---|---:|---:|---:|---:|
| short | 27,275 | +0.15% | -1.29% | -8.02% |
| raft | 29,923 | +14.41% | +5.10% | -11.69% |
| long-0 | 30,532 | +9.92% | +0.57% | -10.81% |
| long-1 | 38,539 | +8.84% | +4.92% | -8.21% |

LFU is strongest on three of these fixed-layout held-out requests; SLRU has a
small edge on short. Per-layer layout choice matters more here than switching
policy. Current deployed online adaptive is NOT the near-uniform control, so
these reductions are not an extra gain demonstrated over current production.

Replacing miss minimization with the calibrated-time objective changes some
layouts, but held-out surrogate time differences vs miss minimization are only
about -0.15%..+0.13% across tested policies/requests. Equal-sized experts and a
roughly linear miss-cost curve make the objectives close in this experiment.
Do not deploy a more complex adaptive solely from those tiny differences.

A future objective should price UNHIDDEN fill/ready latency plus planner and
migration costs, with enough predicted gain to repay migration over a defined
horizon. Address fragmentation, total bytes and source/target runs matter for
DMA, but current mapped fill still partitions work per expert. Instrumentation
and calibration must reflect that actual transport. Overlap opportunity belongs
to the model dependency chain, which this transport-only replay omits.

CPU validation: existing 334 regressions and four new recording/cost invariants passed.
The original transfer was CPU-only; subsequent real-pack GPU replay and payload
validation are documented in [LIVE_REPORT.md](LIVE_REPORT.md).

CPU regressions, complete result JSONs, chosen LFU layouts and a small transfer
package accompany this experiment. No GPU performance result is claimed here.
