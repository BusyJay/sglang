# Final prepared patch series and staged GPU acceptance

All code is prepared; GPU execution remains not_run. Code/package review is a
separate gate from GPU correctness/performance acceptance. User-authorized experimental defaults are dynamic L2, packed decode/prefill
and small direct/fallback batches of 2. See EXPERIMENTAL_DEFAULTS.md for ordinary
startup without a profile and per-feature shutdown. The benchmark old baseline
remains explicitly L2-off/direct/group1/overlap-on. User authorized preparing patches before GPU
on 2026-10-13; historical GPU-before-development gates are superseded.

Recommended series: benchmark a2, consolidated 06 r3, 05, 01, final tools/argv fix,
experimental defaults, delivery metadata.
package.json lists exact implementation base/trees. At a2 skip benchmark; at 06
skip benchmark/06; at 05 apply 01/tools; at 01 apply tools/defaults/delivery. Check hashes and
each `git apply --check`. No historical defective r1/r2 L2 patches are replayed.
Performance reference 02+03+04 is 384c83da44a4334dcaa74c474195ab960126e029;
implementation base is d1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f. On the final tree,
all toggled comparisons are ablations, not separate 04+single-patch trees.

## Prepare and verify

Use an idle SM120 GPU and already available local image/checkpoint/pack. Set
CHECKPOINT, BENCH_DIR (new directory), MACHINE, SUITE, TRACE, TRACE_META to absolute
paths. Use machine.example.json and edit local revision/image/paths/legal PLE and
remote ViT settings. Never silently change slots/chunk. Follow GPU_HANDOFF.md for
Raft answer freezing and tokenizer/session boundary inspection. Use a clean final
source checkout with fixed dependencies/environment; code sources and dirty diff,
GPU UUID/driver, software, minimal child environment, model revision/metadata/file
signatures, full pack hashes and ALL input byte hashes and absolute artifact locators must stay identical across stages.
The v2 run identity includes `input_artifacts` (path/byte SHA256), cross-checked
with `inputs`. Keep those files available when selecting, sealing and loading.
Changing a locator requires a fresh campaign; old v1 results cannot be promoted.
Measured first-turn and warmup IDs must equal frozen file IDs. Later turns must
equal prior actual input + actual answer with all configured trailing strip IDs
removed + frozen suffix. Warmup raw JSON, output/count/finish and SSE timing are
verified too; warmup never enters paired performance scores.
Checkpoint weights are identified by revision+metadata+size/mtime, not full weight
hash; operator can retain a separate weight digest. Timing uses client SSE token
boundaries, not pure prefill kernel time. Profile is separate from normal runs.

```bash
export PYTHONPATH="$PWD/python"
python3 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output "$BENCH_DIR/gpu-small.json"
python3 -m tools.qwen38_cache_tests.benchmark prepare-inputs \
  --suite tools/qwen38_cache_tests/benchmark/suite.combinations.json \
  --checkpoint "$CHECKPOINT" --output "$BENCH_DIR/inputs"
export SUITE="$BENCH_DIR/inputs/suite.json"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark campaign plan \
  --suite "$SUITE" --gates tools/qwen38_cache_tests/benchmark/gates.example.json \
  --output "$BENCH_DIR/plan.json"
python3 -m tools.qwen38_cache_tests.benchmark campaign profile \
  --plan "$BENCH_DIR/plan.json" --output "$BENCH_DIR/baseline-pending.json"
```

GPU small report includes actual CUDA capability, code and machine provenance.
Inspect real MoE parity, payload/alpha/ring/slot/migration, fixed/dynamic L2 KV
borrow/return and graph tests; long context/session allocation checks additionally
require real model cases. No CPU/mock or synthetic trace certifies a machine profile.
Keep all GPU test failures and skipped results. `--require-sm120` rejects any skips.

32 unique configurations cover all binary feature combinations, fixed/dynamic L2,
decode/prefill packed separately, direct group 1/2/layer, overlap-on deployment
baseline and overlap-off control. Packed decode uses single-expert descriptors;
its grouping variants are intentionally not duplicated in screening. Direct
staging groups still issue target DMA separately and can increase DMA calls on
page splits. Neither reduced batch counts nor zero descriptor H2D proves benefit.
Scratch/table memory consumes KV sizing budget; compare actual KV capacity.

First run the frozen deployment baseline across all cases and review its noise,
correctness and effective configuration before the candidate screening stage:

```bash
python3 -m tools.qwen38_cache_tests.benchmark campaign stage \
  --plan "$BENCH_DIR/plan.json" --stage baseline --output "$BENCH_DIR/baseline-suite.json"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$BENCH_DIR/baseline-suite.json" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/baseline"
```

## Cost-controlled stages

The prepared suite has tune and holdout token files. Do not regenerate inputs
between stages. Plan freezes weights and gates BEFORE seeing candidate results.
Defaults: screen shortest tune case/1 round, tune all tune cases/3 rounds/top 3, holdout 5 paired process rounds.
Holdout requires separate measured files AND distinct byte hashes from tune.
If a shortlist is empty, later stages retain baseline. Profile remains baseline.
Raw failed/incomplete samples remain; missing units or foreign fingerprints fail
closed. Per-case median TPOT/TTFT/duration regression limits are 3%, weighted
geometric TPOT ratio gain >=3%, paired round bootstrap 95% upper ratio <=0.97.
Five pairs may be noisy: evidence insufficient means baseline, never guessed gain.

```bash
python3 -m tools.qwen38_cache_tests.benchmark campaign stage \
  --plan "$BENCH_DIR/plan.json" --stage screen --output "$BENCH_DIR/screen-suite.json"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$BENCH_DIR/screen-suite.json" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --results "$BENCH_DIR/screen"
python3 -m tools.qwen38_cache_tests.benchmark campaign select \
  --plan "$BENCH_DIR/plan.json" --stage screen --results "$BENCH_DIR/screen" --output "$BENCH_DIR/screen-selection.json"
python3 -m tools.qwen38_cache_tests.benchmark campaign stage \
  --plan "$BENCH_DIR/plan.json" --stage tune --previous "$BENCH_DIR/screen-selection.json" --output "$BENCH_DIR/tune-suite.json"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$BENCH_DIR/tune-suite.json" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/tune"
python3 -m tools.qwen38_cache_tests.benchmark campaign select \
  --plan "$BENCH_DIR/plan.json" --stage tune --previous "$BENCH_DIR/screen-selection.json" \
  --results "$BENCH_DIR/tune" --output "$BENCH_DIR/tune-selection.json"
python3 -m tools.qwen38_cache_tests.benchmark campaign stage \
  --plan "$BENCH_DIR/plan.json" --stage holdout --previous "$BENCH_DIR/tune-selection.json" --output "$BENCH_DIR/holdout-suite.json"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$BENCH_DIR/holdout-suite.json" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/holdout"
python3 -m tools.qwen38_cache_tests.benchmark campaign select \
  --plan "$BENCH_DIR/plan.json" --stage holdout --previous "$BENCH_DIR/tune-selection.json" \
  --results "$BENCH_DIR/holdout" --output "$BENCH_DIR/holdout-selection.json"
```

Existing run/replay `--resume` verifies full identity; never change files in place.
Collect and seal REAL route trace using Stage A commands, same code/machine/pack/
input directory. Trace collection/profiling have separate results and no speed
ratios. Correctness replay must use identical candidate configs to holdout suite.
Run replay with `--correctness` on holdout-suite.json to produce replay-correctness.
Run independent normal/profile replay and full model profiling in separate dirs
for overhead, first H2D, final weights ready/MoE start and token wall diagnostics.

## Profile sealing, loading and fallback

Before sealing, a reviewer inspects holdout generated reasoning, token/session
boundaries, real model/long-context/KV behavior and output divergences. Prepare
semantic-review.json: `status="passed"`, nonempty `reviewer`, `notes`,
`selection_sha256` from core.digest(holdout-selection), and exact mapping
`reviewed_request_artifacts={path:sha256}` for EVERY request artifact in selection.
This human attestation alone is insufficient: the sealer also requires self-bound
SM120 CUDA test results, real trace/payload correctness and independently rebuilt
screen/tune/holdout metrics from raw stream events. Reports are local artifacts,
not cryptographic hardware attestation; maliciously forged test provenance cannot
be detected. Operator/reviewer must trust execution environment, not handwritten
GPU flags. Changing approved/status never bypasses reconstruction and hashes.

```bash
python3 -m tools.qwen38_cache_tests.benchmark campaign seal-correctness \
  --plan "$BENCH_DIR/plan.json" --selection "$BENCH_DIR/holdout-selection.json" \
  --gpu-small "$BENCH_DIR/gpu-small.json" --replay-results "$BENCH_DIR/replay-correctness" \
  --semantic "$BENCH_DIR/semantic-review.json" --output "$BENCH_DIR/correctness-seal.json"
python3 -m tools.qwen38_cache_tests.benchmark campaign profile \
  --plan "$BENCH_DIR/plan.json" --selection "$BENCH_DIR/holdout-selection.json" \
  --correctness "$BENCH_DIR/correctness-seal.json" --output "$BENCH_DIR/machine-profile.json"
python3 -m tools.qwen38_cache_tests.benchmark campaign check \
  --profile "$BENCH_DIR/machine-profile.json" --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark campaign launch \
  --profile "$BENCH_DIR/machine-profile.json" --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --receipt "$BENCH_DIR/launch.json"
# Add --execute only to start a new managed foreground instance; existing port fails.
# Explicit baseline fallback is available even for candidate/pending profiles:
python3 -m tools.qwen38_cache_tests.benchmark campaign launch \
  --profile "$BENCH_DIR/baseline-pending.json" --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --fallback-baseline --receipt "$BENCH_DIR/fallback-launch.json"
```

Server priority is profile argv < machine.deployment_args < explicit argv.
The final command has one occurrence per flag. Repeatable
`--server-override=--chunked-prefill-size=2048`, or the separate-value pair
`--server-override=--chunked-prefill-size --server-override=2048`, replaces both
profile and machine values. Boolean options use flag form, for example
`--server-override=--disable-overlap-schedule` (no `=false` form). Types and arity come from actual ServerArgs field declarations, not old
assertion values. Optional scalar/list and boolean positive/negative actions
are supported; `--encoder-urls URL1 URL2` is a complete list group. Custom pure
parsers are compiled from exact local production functions; other custom
actions need actual CLI dependencies and fail clearly on an unsupported host.
Loader JSON `--loader-overrides` overrides profile loader settings. Explicit
changes are always `explicit_override_unvalidated`; machine changes to existing
profile assertions are `machine_override_unvalidated`. Update the suite and
rerun GPU evidence before claiming these settings validated. All modes, including
baseline fallback and overrides, check final server assertions plus normalized
JSON string/dict loader and model/revision before recording successful startup. Managed model/revision/port/manifest/trace options
cannot be overridden. Illegal L2 overlap rejects launch. Source defaults are final
fallback. Profile fingerprints must match current binding or validated application
fails; no silent cross-machine reuse. Receipt records effective configuration at
startup; mismatch stops only this newly launched process. No production restart.
Candidate profiles cannot load as accelerated defaults; fallback is explicit.

Return entire inputs/plan/stage suites/raw outputs/logs/trace/tests/profile/receipts,
complete command history and unexecuted checklist. Missing benefit keeps baseline.
All GPU checks and validated profiles in THIS delivery are not_run/not_generated.

Final r3 fixes FINAL_R2_REVIEW type/arity and enables experimental defaults;
final r2 fixed FINAL_R1_REVIEW R1–R3; r1 evidence is preserved in FINAL_R1_VALIDATION.json. CPU protocol tests simulate hardware/report values only inside temporary directories for positive sealing/loading and fake execution. They are not actual GPU acceptance and their files are deleted; no deployable validated profile is included.
