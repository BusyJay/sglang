Current oct13-final-r3 test defaults are user-authorized dynamic/packed/packed/group2. See EXPERIMENTAL_DEFAULTS.md for ordinary startup and shutdown; historic off-default guidance below applies to explicit baselines. GPU not_run.

> 01 update: see PATCH01_HANDOFF.md for submission groups and packed limitations.

> 05 update: see PATCH05_HANDOFF.md for packed controls, GPU tests and matrix.

> Execution update, 2026-10-13: user authorized preparing all patches before GPU testing.
> Historical baseline-before-06 gate below is superseded. See PATCH06_HANDOFF.md
> and ../OCT13_BENCHMARK_AND_PATCH_PLAN.md. This file retains Stage A commands/history;
> current suite version is oct13-final-r3 and all GPU evidence remains not_run.

# Stage A GPU handoff — benchmark only

Implementation base is `d1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f`.
Performance base is `384c83da44a4334dcaa74c474195ab960126e029`
(02+03+04). This patch changes only benchmark tools/docs/tests, no 06/05/01,
no deployment defaults. `package.json` records exact base/target trees and patch
hash. The previous RUN_ON_GPU.md patch application instructions refer to older
packages; do not apply those old patches for this handoff.

Apply the single patch to a clean checkout of its exact implementation base:

```bash
sha256sum -c SHA256SUMS
git switch --detach d1d85bf29b855e511d2a96f9a7f6cec73fc2bb1f
git apply --check 01-stage-a-benchmark.patch
git apply 01-stage-a-benchmark.patch
```

Run inside an already prepared GPU container with all local dependencies,
checkpoint, tokenizer, PLE files if needed, and assembled pack available. No
model downloads, production restarts or implicit GPU-switch workflow. Mount the
patched checkout read-write and run from its root. Use Python >=3.10. Choose an
idle SM120 GPU; reserve enough disk for whole raw result trees. Record immutable
local image identity in machine config. For generic GPU parity first use the
existing `run_gpu.py` runner below (SM120 required for real NVFP4 parity).

```bash
export PYTHONPATH="$PWD/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python3 -m unittest tools.qwen38_cache_tests.test_cpu_benchmark -v
python3 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/run_gpu.py --help
```

For the complete small-test report, run:

```bash
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 \
  --output /absolute/new/experiment/gpu-small.json
```
 The existing
container wrapper is also available: `tools/qwen38_cache_tests/docker_gpu.sh`
with `--image`, `--device`, `--results`, and `-- --require-sm120` (see existing
RUN_ON_GPU.md). Do not interpret any CPU fixture as GPU parity.

Copy machine.example.json to an external experiment directory. Set absolute
checkpoint/manifest paths, model revision matching pack, actual image digest,
port and legal deployment args. For the deployment requiring remote ViT use
`--language-only --encoder-urls <existing URLs> --encoder-transfer-backend
zmq_to_scheduler` instead of `--language-model-only`; text requests contain no
images. Include required PLE directory args for your deployment. Add every
additional resolved launch field to `expected_effective` if it changes the
experiment. The sample asserts graph=breakable/disabled, overlap on, one active
request, FP8 KV, PLE lazy, mem fraction=0.8, context=65536 and 96 slots/layer,
chunk=4096; these are startup proposals, not measured recommendations.

```bash
export BENCH_DIR=/absolute/new/experiment
export CHECKPOINT=/absolute/local/checkpoint
mkdir -p "$BENCH_DIR"
cp tools/qwen38_cache_tests/benchmark/machine.example.json "$BENCH_DIR/machine.json"
# Edit machine.json now to record real local paths, revision/image and deployment args.
python3 -m tools.qwen38_cache_tests.benchmark prepare-inputs \
  --suite tools/qwen38_cache_tests/benchmark/suite.example.json \
  --checkpoint "$CHECKPOINT" --output "$BENCH_DIR/inputs"
export SUITE="$BENCH_DIR/inputs/suite.json"
export MACHINE="$BENCH_DIR/machine.json"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
```

Preparation uses the local tokenizer/chat template and saves exact token arrays.
Long contexts are synthetic repeated text (28,672 and 51,200 input tokens);
tune/holdout filler differs. Short input plus 2048 fixed output tokens isolates
long decode. The user-supplied Raft session asks for overall flow then arbiter
voting/membership without ordinary replication/commit. It uses EOS with 8192
output cap per turn, greedy temperature=0, sampling_seed=42. Save and review
full reasoning/replies; natural output length can vary. It is not a guarantee
that every request reaches the cap, and fixed seed does not fix generated routes.
The second turn appends the rendered assistant/user chat suffix, stripping a
terminal EOS once. Inspect these boundaries with the actual tokenizer before
freezing the suite. No external tool declarations are supplied to the model.

Smoke, then freeze a baseline first answer for prefill comparisons:

```bash
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --case raft-session --rounds 1 --results "$BENCH_DIR/preparation-session"
# Inspect request-0.json text/output_ids/finish_reason and request-1.json.
python3 -m tools.qwen38_cache_tests.benchmark freeze-answer \
  --suite "$SUITE" --inputs "$BENCH_DIR/inputs" \
  --source-unit "$BENCH_DIR/preparation-session/units/r0-raft-session-baseline04/unit.json" \
  --output-suite "$BENCH_DIR/inputs/frozen-suite.json"
export SUITE="$BENCH_DIR/inputs/frozen-suite.json"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
```

The frozen case reuses the exact first answer + arbiter prompt in a fresh
process. Its TTFT includes prefill plus first decode and client overhead; it is
not pure kernel prefill time. Inspect EOS/template boundaries and semantic
answers. Preparation artifacts are separate from benchmark. Freezing adds an
input, so start a NEW result directory afterwards; old fingerprints must fail.
If the first answer is capped, its incomplete content remains recorded and must
be reviewed before selecting that input. Further replay of an answer can be
added as another saved-token case, without changing the measured inputs in place.

A process-restart case starts after the configured model offload startup warmup;
expert cache is therefore not empty. Warm case adds explicit warmup request.
Prefix/KV and expert cache persist between session turns. Host page cache is
uncontrolled; no privileged cache eviction. PLE and remote-encoder behavior is
recorded. Never report a single generic "cold" result for these different states.

Collect a real trace independently (short case limits recording cost), seal and
check its hash/boundaries, then GPU correctness before timed transport replay:

```bash
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --case short-decode --rounds 1 --mode trace --results "$BENCH_DIR/trace-collection"
python3 -m tools.qwen38_cache_tests.benchmark seal-trace \
  --suite "$SUITE" --machine "$MACHINE" \
  --source-unit "$BENCH_DIR/trace-collection/units/r0-short-decode-baseline04/unit.json" \
  --output "$BENCH_DIR/routes.jsonl"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$BENCH_DIR/routes.jsonl" --trace-meta "$BENCH_DIR/routes.jsonl.meta.json" \
  --correctness --results "$BENCH_DIR/replay-correctness"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$BENCH_DIR/routes.jsonl" --trace-meta "$BENCH_DIR/routes.jsonl.meta.json" \
  --results "$BENCH_DIR/replay-baseline"
```

Trace collection may include startup warmup; all ordered sweeps are retained and
source logs/unit identify coverage. Cap is 240000 events. A complete capped prefix
can be replayed but is marked truncated; partial sweeps are rejected. Full-model
trace writes synchronize route IDs and must never supply latency claims. The
GPU replay keeps real pack shapes/payload and initial empty cache; default extra
warmup prefix is zero. To measure hot replay use a new directory and an explicit
`--warmup-events` count divisible by 48 and within the saved trace. This repeats the first N events, then measures the ENTIRE
trace from event zero; `measured_event_range=[0,event_count]` is saved. It does not
exclude the warmup prefix from the measured work or claim production steady decode.
It is
transport-only, with no real MoE compute/graph/overlap; wall time includes dispatch,
staging, handshakes and one final synchronization. Normal replay records only complete wall boundaries; it creates no per-layer
timing events or host marks. Periodic runtime stats logging is explicitly disabled
(`stats_log_interval=0`), with before/after boundary counters retained. The actual
runtime value is saved and checked in worker results. Full-model periodic logs
remain as configured in the frozen loader and are included in its request cost.
Independent replay profiling records auxiliary CUDA readiness intervals and host
marks. These are not copy times. Correctness compares all routed payload bytes/alpha against
mmap reference, reading them separately; those times are excluded from performance.
Use existing SM120 resident/cached NVFP4 parity to check actual MoE correctness.

If real trace collection is blocked, create a synthetic trace for smoke only:

```bash
python3 -m tools.qwen38_cache_tests.benchmark synthetic-trace \
  --suite "$SUITE" --machine "$MACHINE" --tokens 16 \
  --output "$BENCH_DIR/synthetic.jsonl"
```

Use the replay command with that trace and matching metadata. Synthetic routing
cannot independently select a deployment default.

Run the frozen full-model baseline (five fresh process rounds per case) only
after smoke and correctness pass. If any config cannot start, retain logs and
create a new suite/control group; do not silently change slots/chunk/memory.
Random candidate ordering and case ordering are deterministic and recorded;
with baseline only, repeat noise can be reviewed without speedup claims.

```bash
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --results "$BENCH_DIR/model-baseline"
# After interruption, exact identity is required; completed failed units are retained.
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --results "$BENCH_DIR/model-baseline" --resume
python3 -m tools.qwen38_cache_tests.benchmark summarize \
  --results "$BENCH_DIR/model-baseline"
# Independent profiling: never merge these metrics into benchmark results.
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --case raft-answer-prefill --rounds 1 --mode profile --results "$BENCH_DIR/profile"
python3 -m tools.qwen38_cache_tests.benchmark candidate-profile \
  --results "$BENCH_DIR/model-baseline" --candidate baseline04 \
  --output "$BENCH_DIR/baseline-candidate.json"
```

Candidate export is diagnostic (`candidate`, `approved=false`). Validated profile
selection/loading/default rollout belongs to stage E after reviewed GPU evidence;
this stage does not promote configs. Model requests and replay workers use the same minimal child environment built
by `subprocess_environment()`. Only explicitly allowlisted execution/cache/path
settings are inherited; all other inherited variables are removed (including
unlisted SGLANG switches and credentials). To use a new execution toggle, add its
exact name to the allowlist and rerun tests in a new results directory. Recorded
and fingerprinted effective variables include CUDA_LAUNCH_BLOCKING,
CUDA_DEVICE_MAX_CONNECTIONS, CUDA module/allocator settings and
SGLANG_QWEN38_ELASTIC_L2. Changing them rejects resume. Offline flags and repo
PYTHONPATH are forced consistently and recorded; unrelated variables do not
change effective conditions. See core.py for the exact allowlist. This audits a
future L2 variable, and does not implement L2.

Mode, effective args, actual imports, model
revision, pack full payload hash, input hashes, software/hardware and environment
are fingerprinted. Full pack hashing each invocation may take minutes, outside
measurement. Model weight identity uses revision + metadata hashes + file
size/mtime signatures, not a full checkpoint content digest; record an independent
checkpoint digest if supplied by the operator. GPU clocks/temperature/processes
are sampled before/after; unstable external activity needs review. Only whitelisted
environment variables are captured, no credentials.

Return the entire experiment directory (inputs, source preparation, sealed trace,
all result modes, logs, raw output IDs, timing events, failed/incomplete units,
summary, candidate profile), machine config and complete commands. Add a GPU
validation checklist with CUDA compilation, SM120 MoE parity, replay payload
correctness, model semantic checks, EOS/template boundary inspection, baseline
noise and effective config verification. Label all unexecuted items `not_run`.
Do not copy historical throughput numbers into these results. Review must freeze
measurement gates and independent sample requirements before GPU performance acceptance.

Stage A revision after R1–R3 review: old a1 results/packages are preserved, but
cannot be resumed or treated as a2 evidence. Worker reports are validated for
version/mode/trace/candidate/warmup/environment, required counters and wall field.
Existing units additionally validate parent fingerprint, round/order/pair and
status. Invalid units fail closed; malformed/missing worker reports are stored as
failed units and excluded from summary. Interrupted workers without a committed
unit run again, preserving their report as `replay.orphan-N.json`.

Run replay profiling in a NEW result directory, alongside a normal replay with
identical trace/warmup to quantify instrumentation overhead on the GPU. Profile
`wall_s=null`; `diagnostic_wall_s` is available for overhead inspection and is
excluded from speed ratios and normal summaries:

```bash
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$BENCH_DIR/routes.jsonl" --trace-meta "$BENCH_DIR/routes.jsonl.meta.json" \
  --profile --results "$BENCH_DIR/replay-profile"
```

Model SSE header/connect wait uses the total request timeout, without a hidden
30-second header cap. After headers, socket waits use the remaining deadline;
partial outputs and timeouts remain failure evidence. Review normal/profile
GPU overhead, real headers with long prefill, and effective child environment
before freezing the formal baseline. All GPU checks remain `not_run` locally.
