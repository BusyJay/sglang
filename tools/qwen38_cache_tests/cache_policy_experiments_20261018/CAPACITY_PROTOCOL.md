# Capacity-allocation GPU supplement, 2026-10-18

This experiment fills the missing GPU allocation comparison in LIVE_REPORT.md.
It does not rerun eviction-policy selection: all candidates use LFU and exactly
6390 expert slots across 48 layers. Current `adaptive` fill and prepared launch
are enabled by the standalone shell script, never YAML.

## Predeclared design

- Reuse the exact coding and raft route tapes from the previous recording.
  Each contains 1024 complete verify sweeps, including original token order and
  duplicates. MTP acceptance and routing are fixed inputs.
- Select layouts using only sweeps 0–511; exclude the first 128 sweeps from
  training miss/cost scores. No layout tuning uses sweeps 512–1023.
- Four algorithms: uniform (six layers at 134, remaining layers at 133), current
  production allocation worker, patch DP minimizing misses, patch DP minimizing
  measured fill latency (`mapped_adaptive`, median GPU milliseconds).
- Production uses its own default candidate grid extending to 256; both patch
  objectives share [96,112,128,133,134,144,160,192,256]. Grid differences are part
  of these implementations, so this is not a pure optimizer-only comparison.
- All GPU candidates reserve the same adaptive pool and metadata, start empty
  with the **uniform layout**, and replay exactly the same 512-sweep prefix.
  At the split, invoke the real `_maybe_rebalance` path for selected layouts.
  This preserves production's frequency/age retention, actual D2D relocation,
  metadata restoration and mailbox refresh. Do not rebuild a target-layout
  cache and call it migrated state.
- Then time all remaining 512 sweeps, including any transient refills and first
  use after migration. Report migration wall time separately and add it to the
  held-out wall time for the primary inclusive comparison against uniform.
- Each candidate is a fresh process. Five repeats per workload and algorithm,
  shuffled in each block with seed 20261019; no timing outlier removal or early
  stopping based on results. Report every time, CV and descriptive paired
  log-ratio t(4) intervals conditional on these two tapes.
- Before timing, independent correctness processes check all six tensors for
  all active experts of every layer over 16 sweeps immediately after migration.
  Readback is excluded from performance runs. Check equal prefix counters and
  deterministic held-out counters, full budget, target layouts and copy bytes.
- Keep adaptive allocation buffers enabled for uniform too. Disable the
  sampling observer in the experimental runtime subclass for every candidate;
  train layouts in advance, prevent asynchronous reselection and invoke the
  actual migration synchronously at the same boundary.
- CPU static-layout held-out replay is a diagnostic prediction, not a required
  equality check for migrated GPU caches: retained residents and ages differ.

## Scope and restoration

This measures planner/transport on the real FP4 pack, without model compute,
attention, draft execution or overlap. It cannot establish service tokens/s.
CPU layout analysis is offline and excluded from the primary GPU timing;
background analysis latency and observer overhead need a serving experiment.
The migration is deliberately applied at a controlled split rather than at a
background worker's variable completion time.

Reuse the existing experiment-window authorization. Stop only homelab-llm,
leave ViT running, and restore the original service in `finally`. Verify health,
launcher hash and unchanged ViT start time. Never edit the production script.
Archive raw JSON/logs, layouts, CPU worker request/reply, training reports,
source hashes and a reproducible summary with the report.
