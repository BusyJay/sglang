# User-authorized experimental defaults, oct13-final-r3

The user requested all optimizations enabled for testing and accepts manually
disabling or reverting them. This overrides historical default-off gates. Actual
Qwen38 threaded runtime constructor and loader omissions now select:

| Setting | Experimental default | Disable/control |
|---|---|---|
| runtime_mode | threaded | sync/async/device require explicitly disabling threaded features |
| l2_mode | dynamic | off; fixed requires explicit l2_fixed_seed |
| packed_decode | packed | direct |
| packed_prefill | packed | direct |
| decode_submit_experts | 2 | 1 per expert; 0 whole layer candidate |
| disable_overlap_schedule | true when L2 requested | L2 off permits overlap-on |

Default packed policies retain byte/copy thresholds and direct fallback. Their
immutable descriptors and DMA operate per expert. Group2 affects direct/fallback
submission and does not merge packed descriptors across experts. Prefill submission
granularity is unchanged. No correctness or speed benefit is claimed before GPU
execution; no validated machine profile is supplied.

`resolve_experimental_config` fills omitted loader settings and preserves explicit
values. Qwen38 startup publishes the resolved loader JSON and resolves overlap-off
when L2 is enabled, logging the effective configuration. The registered fields
permit these resolution decisions. KV allocation checks the resolved Qwen38 load
format; unrelated models and unpublished standalone pools do not request L2.
Unsupported rank/speculation/disaggregation/KV layouts fail with an L2-off option;
the runtime never silently disables the requested features.

Ordinary startup needs no validated profile. Use absolute machine/input paths;
prepare inputs as in FINAL_GPU_HANDOFF.md. `$SUITE` is a prepared suite and
`$MACHINE` points to actual local checkpoint/pack/revision/container settings.

```bash
# Dry-run, review command/effective assertions; no profile required.
python3 -m tools.qwen38_cache_tests.benchmark campaign experiment \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --receipt "$BENCH_DIR/experimental-launch.json"
# Add --execute to own and start a new foreground test server on the free port.
python3 -m tools.qwen38_cache_tests.benchmark campaign experiment \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --receipt "$BENCH_DIR/experimental-execute.json" --execute
# Full old-path control, overlap-on unless machine explicitly requests off.
python3 -m tools.qwen38_cache_tests.benchmark campaign experiment \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --loader-overrides '{"l2_mode":"off","packed_decode":"direct","packed_prefill":"direct","decode_submit_experts":1}' \
  --receipt "$BENCH_DIR/old-path-launch.json"
# Whole-layer direct candidate; explicit and unverified.
python3 -m tools.qwen38_cache_tests.benchmark campaign experiment \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --loader-overrides '{"packed_decode":"direct","decode_submit_experts":0}' \
  --receipt "$BENCH_DIR/layer-launch.json"
```

`campaign experiment` starts from actual shared defaults, not a validated profile.
It records `experimental_unvalidated`, `GPU=not_run`, merges machine and explicit
options, and checks all effective fields plus loader JSON and model/revision.
`campaign launch` remains the evidence-checked machine-profile path. These are
different claims: experimental startup is authorized; validated promotion still
requires real GPU correctness, paired independent holdout and semantic review.

The normal `docker_model.sh ... -- canary` entry now defaults threaded/dynamic/
packed/packed/group2 and adds overlap-off. It checks actual loader and overlap
before generation and retains `/get_server_info` in report.json. Individual
canary switches are `--l2-mode`, `--packed-decode`, `--packed-prefill`,
`--decode-submit-experts`; fixed mode also requires `--l2-fixed-seed FILE`.

```bash
# Use known legal local image/model/pack paths as in docker_model.sh.
tools/qwen38_cache_tests/docker_model.sh --image "$IMAGE" --device "$GPU" \
  --checkpoint "$CHECKPOINT" --pack "$PACK" --results "$RESULTS" -- canary
# Explicit legacy experiment and fallback:
tools/qwen38_cache_tests/docker_model.sh --image "$IMAGE" --device "$GPU" \
  --checkpoint "$CHECKPOINT" --pack "$PACK" --results "$RESULTS" -- canary \
  --runtime-mode sync --l2-mode off --packed-decode direct --packed-prefill direct \
  --decode-submit-experts 1
```

Benchmark baselines explicitly bind threaded/L2off/direct/direct/group1/
overlap-on and all frozen key settings. Every suite candidate must declare the
four feature parameters. Matrix keeps L2off overlap-off control; packed groups
are canonically represented as 2 since packed DMA still stays per expert. Single
patch/combination comparisons remain same-tree ablations. The standard suite
includes a named experimental-defaults candidate; ordinary testing chooses the
experiment entry, whereas campaign stages deliberately measure their baseline
and candidate matrix.

Defaults patch can be reverted independently of reviewed 06/05/01 algorithms
and argv fix. Old package/docs are preserved as history. GPU payload/async/ring/
reclaim/KV/session/graph/performance remain not_run. The added default-constructor
GPU test uses real borrowed VMM storage and compares payloads, then reclaims;
it is prepared but unexecuted. Existing legacy transport tests now explicitly
choose off/direct/group1 so future defaults cannot change their intended cases.
