# Stage A review response — oct13-a2

Date: 2026-10-13 Asia/Shanghai. Revision parent:
`6194db936af5867d6074093f441b738b5097fb54`. The original
STAGE_A_REVIEW.md is preserved. This response records implementation/CPU evidence;
it does not declare review approval or a completed GPU baseline. No 06/05/01.

| Review | Change | CPU evidence |
|---|---|---|
| R1 | `execute_interval` normal mode only brackets wall time with boundary sync/counters; `ReplayProfile` is constructed exclusively for `--profile`. All replay modes explicitly force stats_log_interval=0. Profile results use diagnostic_wall_s, wall_s=null, separate fingerprint/mode, no speed ratios | `test_replay_benchmark_has_no_profile_events` checks normal/profile branch and event/record counts; `test_replay_profile_never_enters_speed_summary` exercises actual matrix/summary. No GPU performance claim |
| R2 | `subprocess_environment` is one minimal allowlist policy for service/import probe/replay worker. Unknown inherited variables are removed, offline/PYTHONPATH forced, forwarded execution settings fingerprinted and recorded in effective child env. CUDA sync/connections, allocators and explicit SGLANG_QWEN38_ELASTIC_L2 included | `test_effective_environment_fingerprints_execution_settings` checks changed CUDA/cache/allocator/connections reject resume, unrelated/secret/unlisted switches removed; `test_effective_server_environment_is_full_policy` checks server artifact; worker matrix fixture checks policy |
| R3 | Existing replay units validate parent fingerprint/round/candidate/order/pair/mode/trace/warmup and successful result fields. Zero-exit worker must supply valid status/version/mode/trace/candidate/range/env/counters/config/wall. Invalid new reports become failed units. Foreign units fail closed. Orphan reports are preserved and never reused | `test_replay_resume_rejects_foreign_and_malformed_units`, `test_replay_worker_success_requires_valid_report_and_recovers_orphan`, `test_successful_worker_report_validation_rejects_wrong_metadata` exercise real replay matrix and validators |

Nonblocking clarifications also handled: warmup help/report state "repeat first N
trace events, then measure the entire trace [0,end)"; no claim of warmup-prefix
exclusion. SSE header wait now uses the request timeout without an implicit
30-second cap; remaining deadline bounds subsequent reads. Delayed-header and
partial-stream deadline tests pass. Sample mem fraction=0.8 remains explicitly
unverified; GPU startup failure must create a new frozen suite/control group.
Transport payload correctness and model semantic acceptance remain distinct.

Validation commands:

```bash
python3 -m unittest tools.qwen38_cache_tests.test_cpu_benchmark -v
python3 tools/qwen38_cache_tests/run_cpu.py
python3 -m compileall -q tools/qwen38_cache_tests/benchmark
python3 -m tools.qwen38_cache_tests.benchmark replay --help
```

Results: 29 benchmark tests passed; full CPU suite 180 tests, 160 passed,
20 skipped for absent torch/numpy. CPU mocks verify orchestration/protocol only.
CUDA compilation, real tokenizer/model/trace, GPU payload/MoE correctness,
performance and normal-vs-profile instrumentation overhead remain `not_run`.
Use GPU_HANDOFF.md for new profile command and complete smoke/baseline workflow.

Package script continues to accept only benchmark-directory files and the exact
benchmark test path. It now copies the original review and this response into
the package. No unrelated README/planning/old zip is included in the patch.
A newly generated package records the new target SHA/tree and hashes; original
package/results remain available and cannot be resumed as a2 evidence.

Next gate: review the R1–R3 revision, then GPU smoke/correctness and frozen formal
baseline, including normal/profile overhead evidence. Return whole raw result
bundle for baseline review before starting stage B (06).
