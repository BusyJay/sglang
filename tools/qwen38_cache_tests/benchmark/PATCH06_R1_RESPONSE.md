# 06 r2 response to R1 code review

Review object: `083a5b1f5dbb34a67596fe303ace0391e2bdc861`.
Source review: `PATCH06_R1_REVIEW.md` (reviewer-owned shared document).
This revision fixes R1/R2 only; 05/01 remain subsequent review-gated work.
GPU tests remain deferred, defaults remain `l2_mode=off`.

R1: `l2_requested()` checks the existing context publication predicate before
calling `get_model()`. Unpublished configuration returns off; published config
is parsed as an object and validates mode/score/budget without broad exception
handling. CPU/device/type filtering in `MHATokenToKVPool` precedes the helper.
The regression starts a fresh Python process, checks actual unpublished context,
calls the real helper, and constructs a real CPU pool while the helper is forbidden.
Other cases use the actual context's scoped publication API for off, fixed,
dynamic, malformed JSON, non-object JSON, invalid mode and negative budget.

R2: ordinary `stats_snapshot()` uses only host counters and reports duplicate
occupancy as null. `boundary_stats_snapshot()` is a separate explicit API that
holds the submission lock, joins job/retirement queues, checks worker errors,
synchronizes then reads device keys. Replay requests this snapshot before timing
and after the elapsed interval. Periodic logging frequency remains unchanged.
The regression forces `_record_invocation → _maybe_log_stats → log_stats →
stats_snapshot` every layer, forbids keys `.cpu()`/`.item()` and CUDA synchronization,
and checks counter logging continues. The explicit snapshot test verifies drain
order before reads and duplicate output. Full-model ordinary logs do not expose
exact duplicate occupancy; GPU boundary/actual scheduling validation is not run.

Reproduction: `tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21`.
Final container run: 211 tests passed, zero skipped; imports/ModelOpt wiring passed.
Host benchmark subset: 31 passed. Offline SM120 routing JIT compiled/loaded;
kernels not executed. Raw evidence is in `/data/workspace/qwen38-06-r2-review-20261013/`.
`PATCH06_R1_VALIDATION.json` retains the earlier validation record. Suite version
is `oct13-06-r2`; r1 results cannot be resumed under revised semantics.
Current GPU correctness/performance/profile overhead remain `not_run`.
