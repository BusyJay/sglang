# Result schema: oct13-a2

JSON serialization rejects NaN/Infinity. Times are seconds unless named `_ms`.
Missing metrics are `null`, never zero. Status is `ok` or `failed`; validation
status can be `passed` or `not_run`. This protocol does not approve deployments.

| Artifact | Required content and meaning |
|---|---|
| `run.json` | `fingerprint` (SHA256 of canonical `identity`), suite and machine config, Git SHA/tree/diff/hash/source hashes, runtime software/CPU/GPU/environment, model revision/JSON fingerprints and file signatures, manifest/payload hashes, saved input hashes, selected mode and schedule |
| `schedule.json` | Ordered `{round,case,candidate,order,pair}` entries; round is an independent process run, pair is round + case |
| `units/rN-case-candidate/unit.json` | Status/error, fingerprint, actual argv/import paths, requested loader, warmup and measured requests, cache state, telemetry before/after, all periodic cache snapshots |
| `effective_server_info.json` | Raw `/server_info`; required loader and selected resolved server fields must match suite assertions before any request |
| `import_probe.json` | New interpreter return code/stdout/stderr; imported SGLang/runtime must be under repository `python/sglang` |
| `server.log` | Entire owned server stdout/stderr, including startup failure |
| `request-N.json`, `warmup-N.json` | Input/output token IDs and counts, output hash/text, finish reason, errors, TTFT/TPOT/duration, raw SSE events with receipt times and cumulative/new token counts; warmup is excluded from summary |
| Request `cache_statistics` | Before/after periodic snapshots and snapshots observed during request; `exact_delta=null`, explicit alignment gap. These are NOT exact request increments |
| `summary.json`, `report.md` | Per case/candidate/session turn raw sample references, n/median/nearest-rank p95/stdev/min/max, missing/failed units, paired TPOT ratios and exploratory bootstrap interval, output equality flags |
| `routes.raw.jsonl` | Separate trace-mode runtime capture, original ordered layer/routes records; NEVER use collection latency as benchmark |
| sealed trace + `.meta.json` | Layer, monotonic token-row offset, route matrix. Hash, kind (`real`/`synthetic`), pack/model identity, event count, collection provenance/truncation; reject incomplete layer sweeps |
| replay `rN-candidate/unit.json` | Pair/order/fingerprint/argv/worker exit/error and raw result; `worker.log` retained even on failure |
| replay `replay.json` | Real payload bytes, runtime source/config, initial empty cache + warmup prefix, synchronized wall interval, profile-only per-layer GPU readiness and host dispatch intervals, per-token spans, aligned before/after counter deltas, correctness evidence |
| replay `summary.json` | Every raw unit, wall statistics and paired wall ratios; correctness runs have `wall_s=null` and cannot supply speed claims |
| candidate profile | `status=candidate`, `approved=false`, full candidate/baseline config, provenance, result path/hash, selection limitations |

TTFT: request transmission start to receipt of first nonempty generated token
chunk. TPOT: first-to-last generated token receipt time divided by N-1; N<2
is unavailable. Tokens coalesced in one chunk share the receipt time. The raw
chunk-size evidence must be reviewed; this is client observation, not GPU token
execution timing. Duration includes EOF/DONE receipt and serialization overhead.
Length policy `fixed` requires ignore_eos and exactly max_new_tokens; `eos`
accepts an early stop or the configured cap, retaining its actual finish reason.
A failed/incomplete request contributes no valid performance metrics to summary,
but its raw partial output is retained. Later missing session turns are not
invented. p95 at n<100 is explicitly unstable. Paired bootstrap with five rounds
is exploratory evidence; no automatic threshold approval in stage A.

Counter deltas in GPU replay include only named monotonic counters, not derived
rates or capacity gauges. CUDA events bracket the compute stream's readiness
wait; host timestamps bracket dispatch. Neither is a substitute for synchronized
whole-replay wall time or real MoE latency. Normal replay performs no MoE and no
per-copy synchronize. Correctness separately reads all routed payloads and
compares with the mmap reference; its copies/checks are not benchmark timings.

Resume requires identical code, input, environment and config fingerprint;
completed failures remain failures. An interrupted full-model unit is written as
failed. A replay interrupted before committing its unit is incomplete and can
be rerun in a fresh child process. Output-directory advisory locking prevents
concurrent writers. After code/config/input changes, start a new results directory.

Revision a2 (R1–R3): normal replay has empty `gpu_intervals`,
`host_dispatch_intervals`, `per_token`, and no per-layer timing events. Replay
`--profile` has mode `profile`, `wall_s=null`, `diagnostic_wall_s` and instrumented
intervals, excluded from paired speed ratios. Correctness is also separate.
All replay modes force and record `effective_runtime.stats_log_interval=0`.
`warmup_policy` repeats the prefix; `measured_event_range=[0,event_count]` always
measures the whole trace. Parent identity and every worker/unit record the mode,
trace hash and warmup association; a2 cannot resume a1 results.

`machine_identity.effective_child_env` and unit/worker effective environments
use the same explicit minimal allowlist in core.py. Unknown inherited variables
are removed, forced offline/PYTHONPATH settings are captured, and credentials
are not inherited. CUDA synchronization/connections, allocator and listed cache
switches are fingerprinted. Forwarding new variables requires a reviewed
allowlist change. Successful worker reports require status/mode/version/candidate/
trace/warmup/range/environment, boundary counters/runtime config and valid wall
fields; successful correctness requires `passed`. Existing units also require
fingerprint/round/candidate/order/pair/mode/trace/warmup. Foreign or incomplete
units fail instead of entering statistics. A zero exit with absent/invalid result
is stored as failed. Completed failures remain failures. Orphan results from
interruption are preserved under `replay.orphan-N.json` before fresh execution.

Campaign additions (`qwen38-campaign-v2` / `oct13-final-r3`): plan freezes full
matrix, case weights and gate values, with digest. Stage suites contain subset
and plan/prior-selection binding. Selections retain all raw artifact paths/hashes,
failed units, per-case ratios, paired independent-round geometric TPOT ratios
and deterministic bootstrap confidence. Candidate/holdout selection never alone
certifies correctness. Profiles include canonical config/fallback, full plan,
recursive screen/tune/holdout evidence, correctness seal and machine/code/model/
input binding. Profile fingerprints and every referenced raw artifact are
rechecked on validated loading. Baseline_pending/candidate/baseline states cannot
load as validated acceleration; explicit fallback remains available.
GPU small reports additionally record execution_kind/capability/code/environment.
Synthetic CPU protocol tests produce no deployable validated profile.

Final r2: campaign/selection/correctness/profile schemas are v2; suite version is
`oct13-final-r3`. Run identity requires `input_artifacts` with exactly the same
keys as `inputs`: each entry contains absolute `path` and JSON file byte `sha256`.
Canonical digest(token list) is not this byte hash. Selection reopens and hashes
every input file, then validates all request/warmup tokens and rebuilds measured
multi-turn history using actual outputs and repeated tail stripping. Missing
locator, changed bytes, wrong tokens or raw warmup evidence reject selection.
All referenced files must remain accessible at their bound paths for loading.
Launch receipt config contains assertions reflecting canonical profile < machine
< explicit argv. Every execution mode checks assertions, loader JSON string/dict,
and model/revision. Explicit overrides remain unvalidated.

Final r3: suite version oct13-final-r3, v2 evidence schemas unchanged. CLI options
use ServerArgs declarations for type/arity and exact production pure parsers;
encoder_urls is always a list, Optional[int] mamba sizes are numeric. Complete
list groups override atomically. Normal experiment receipt is
experimental_unvalidated and does not require a machine profile; validated
loading retains all real GPU evidence gates. User-authorized actual defaults are
dynamic/packed/packed/group2 with L2 overlap-off. Every benchmark candidate must
explicitly declare feature modes, preventing implicit default drift.
