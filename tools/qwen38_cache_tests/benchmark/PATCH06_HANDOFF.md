# 06 L2 cache — ready for GPU testing, code review pending

Current authorization (2026-10-13): prepare 06 → review → 05 → review → 01 →
combination/profile tools before GPU testing. This supersedes the historical
GPU-baseline-before-06 gate; historical Stage A review and validation are retained.
Defaults remain `l2_mode=off`. No GPU correctness or speedup has been measured.
This patch depends on reviewed a2 `55b13ab866d71b610599c5e92feb921003f737a1`,
with r2 configuration/statistics fixes on r1 `083a5b1f5dbb34a67596fe303ace0391e2bdc861`.
R3 fixes the boundary lock on r2 `8fee09d118de4d3d56f0cc58409032ebfd69ea6b`.
The performance code base is 02+03+04 `384c83da44a4334dcaa74c474195ab960126e029`.
It introduces no 05 packed scatter or 01 small-batch submission.

## Configuration and ownership

All settings are loader extra config. `l2_mode` is `off`, `fixed`, or `dynamic`.
`off` builds no L2/policy and keeps the original decode transfer plan size/path.
The legacy non-off `SGLANG_QWEN38_ELASTIC_L2` is rejected. Enabled serving requires
single rank, no schedule overlap, speculation, preemption or disaggregation, and
QSA full MHA with the paged allocator/VMM owner; invalid layouts fail explicitly.

`fixed` requires nonempty `l2_fixed_seed: [[layer, expert], ...]`. Initial attachment
performs one explicit pack→L2 warmup, checks capacity and completes transfers
before publishing storage. The seed remains copied in L2 after promotion; there
are no decode demotions and no automatic reseeding following KV reclamation.
This warmup cost belongs outside timed intervals; record startup and boundary
stats. Shared replay prefill seed is separate and identical for every candidate.

`dynamic` uses `l2_min_score=2.0`, `l2_margin=0.1`,
`l2_demotion_budget=2` (global per ordered model decode sweep, reset layer zero),
`l2_prefill_budget=8` (per layer/chunk). These are unvalidated experiment values.
Admission is decided once, low-score victims use a bounded shortlist refreshed
per epoch, and promotion sources in the current plan are protected from eviction.
L1 victims copy directly to L2 before their slot is overwritten, on the same
transfer stream after previous slot readers. Hit copies remain resident in L2.
No fusion/swap kernel is introduced: direct ordered demotion avoids scratch and
cross-stream source lifetime risks; scratch counters are explicitly zero.

Only decode payload names are stored. With resident alpha enabled, alpha rows are
omitted from L2, decode uses existing alpha scatter and prefill fills alpha from
immutable resident tables. Cached tensor views survive slot reuse and are dropped
before relevant aliases unmap. Slabs preserve VA for unaffected slots on partial
reclaim. Current scheduler borrowing drains jobs and CUDA at capacity boundaries;
allocator excludes free tails before aliasing and removes aliases before returning
KV pages. A high occupied page (including cached/session/shared KV) prevents
lending its containing region. Exceptional alias rollback never advertises an
uncertain tail as free. Runtime shutdown drains work and removes L2 aliases;
physical handles remain owned by KV until its own teardown.

Periodic snapshots use CPU counters only; duplicate occupancy is null.
Snapshots include capacity/entry count, row/occupied bytes,
promotion/demotion/prefill-hit/prefill-admission/fixed-seed counters, score/budget
rejections and view construction count. Explicit `boundary_stats_snapshot()`
holds the submission lock, joins jobs and retired compute, checks worker errors
and calls the inherited `_synchronize_locked()` drain helper before reading device keys for duplicate occupancy. Replay
uses it before starting and after ending the timed interval. No periodic snapshot
calls `.cpu()`, `.item()` or CUDA synchronization. Full-model periodic logs
therefore report duplicate occupancy as null, not zero. Fixed `fixed_seeded`
is historical completed warmup count; current entries may be lower after reclaim. Profile replay enables
opt-in host view/policy/staging/plan wait/ready wait intervals, stream plan-D2H,
transfer including reader waits, and compute callback intervals. Nested intervals
overlap and cannot be summed as exclusive costs. Transport replay has an empty
compute callback; actual MoE gaps/parity require the full model GPU run.

## Reproduction

CPU container and offline routing JIT compile/load (no kernel execution):

```bash
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
docker run --rm --pull never --network none \
  -v /data/workspace/sglang:/workspace/sglang:ro -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 lmsysorg/sglang:v0.5.21 \
  tools/qwen38_cache_tests/compile_cuda.py
```

On GPU, run from patched repo; use the Stage A GPU handoff for checkpoint/tokenizer,
input preparation, Raft answer freezing, real trace collection, pack hashing,
config inspection and normal/profile separation. Suite version is now
`oct13-06-r3`: do not resume a2 results or reuse them as current validation.

```bash
export PYTHONPATH="$PWD/python"
python3 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output /absolute/new/gpu-small06.json
python3 -m tools.qwen38_cache_tests.benchmark prepare-inputs \
  --suite tools/qwen38_cache_tests/benchmark/suite06.example.json \
  --checkpoint "$CHECKPOINT" --output "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --correctness --results "$BENCH_DIR/06-replay-correctness"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --results "$BENCH_DIR/06-replay-normal"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --profile --results "$BENCH_DIR/06-replay-profile"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/06-model"
```

The sample matrix includes baseline/off overlap-on, off overlap-off, fixed and
dynamic overlap-off, same 96 slots/layer and chunk=4096. Replay has an explicit
8 GiB dedicated VMM physical backing proposal, not borrowed KV: record its effective
aligned size; if capacity/OOM fails, preserve failure and create a new comparison
suite rather than silently shrink it. Fixed seed is 10 experts/layer; all candidates
use the same additional prefill route seed. This is an experiment, not a recommended
machine profile. Full model instead uses real free KV tail and may reject fixed
seed for insufficient free space. Real traces and long context KV/session/cached
prefix checks are mandatory before any deployment recommendation.

GPU tests are prepared for extended victim export plan/reference equivalence,
L2 hit/miss alpha-on/off payload parity, prefill/decode transition, capacity-zero
reclaim, physical KV alias overwrite, partial shrink and stable KV graph addresses.
Long-context full-model correctness, real MoE SM120 parity, model graph replay,
allocator admission under real scheduling, performance, instrumentation overhead
and tuning are all `not_run`. Existing tests do not certify those outcomes.
No validated acceleration profile is produced. 05/01/combination tools remain
subsequent review-gated work.
