# 01 decode submission — ready for GPU testing, code review pending

Parent is reviewed 05 r1 `1d71e13964ef29db46fc103a76b13b7257e20d21`.
This patch depends sequentially on 06 r3 and 05. L2-off/packed-direct supplies
single-01 ablation, not a byte-independent 04+01 tree. Defaults remain expert
submission (`decode_submit_experts=1`), packed direct, L2 off. GPU deferred.

## Configuration and behavior

`decode_submit_experts` accepts 1 (default per-expert), 2 or another positive
expert count (bounded by manifest experts), or 0 (whole layer). It controls only
decode miss groups; prefill default staging granularity/heuristics are unchanged.
No automatic default promotion has been made.

Direct groups share pinned pages across experts. Each destination segment still
issues its own H2D DMA: Python/staging grouping is not DMA fusion. Full staging
pages submit immediately and final partial pages submit at the end of each group,
so a two-expert group never waits for future groups to fill a 64 MiB page. A single
miss keeps the existing single-expert fill. Whole-layer grouping is a candidate
that can delay first transfer, not a recommended value.

Each group waits only its own slots' prior readers; its submission happens before
waiting for later groups. In L2 mode, current promotion sources are protected
across the complete plan. Each group's demotions/promotions enqueue on the same
transfer stream before that group's host fill overwrites the slots. All demotion
sources have reader dependencies. Remaining L2 entries/alpha behavior stays as
reviewed in 06. No demotion is moved to an unprotected separate stream.

05 packed descriptors have immutable single-expert source offsets. Packed fills
remain one expert at a time even when decode_submit_experts is 2/0. There is NO
cross-expert packed/scatter payload concatenation or DMA merge. If packed threshold
falls back, the direct group path can share staging. This choice avoids incorrect
reuse of one-expert offsets; future cross-expert packed requires a separately
reviewed descriptor/offset design. The matrix labels both paths explicitly.

## Events and failures

Async `_copy_batch` gains a default-true `publish_ready` option. Only new direct
multi-expert decode groups pass false: their intermediate self.ready event is
unused by the threaded worker, which publishes final job.ready after all copy,
scatter/alpha work. Ring events remain recorded after every staging dispatch,
including partial failure. Existing default startup/prefill/async paths still
publish their required ready events. Single-expert fill retains its original ready
behavior. Group source views are released even if later view construction fails.

The slot reader, bounded host plan lifetime, reaper completion, pinned ring reuse,
worker failure poisoning and final job ready dependencies remain. No event needed
by another mode is removed. Profiling optionally records direct_h2d_submit host
intervals; normal runs add no timing events. GPU profiling must compare first/last
H2D enqueue, final readiness, actual MoE start and token wall time separately;
submission/group counts are diagnostic, not evidence of speed.

## Checks and commands

CPU tests cover 1/2/layer grouping, final partial group, submitting before later
reader waits, L2 promotion/demotion/source protection, packed single-expert dispatch,
view construction failure, real staging partial-page/ring wrap and failed DMA
completion event. Existing worker failure/final ready and prefill tests remain.
GPU tests are prepared for 1/2/layer payload+alpha checks with multiple streams,
ring reuse, packed+01 migration, L2+05+01 payloads. Kernel/async/MoE/KV/graph/long
context, performance and profiling overhead remain not_run.

```bash
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
docker run --rm --pull never --network none \
  -v /data/workspace/sglang:/workspace/sglang:ro -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 lmsysorg/sglang:v0.5.21 \
  tools/qwen38_cache_tests/compile_cuda.py
```

Use GPU_HANDOFF.md and PATCH05/PATCH06_HANDOFF.md to set local checkpoint/pack,
machine config, trace and frozen inputs. Current suite `oct13-01-r1` rejects old
results resume. Do smoke/correctness before measurements; keep all raw failures.

```bash
export PYTHONPATH="$PWD/python"
python3 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output "$BENCH_DIR/01-gpu-small.json"
python3 -m tools.qwen38_cache_tests.benchmark prepare-inputs \
  --suite tools/qwen38_cache_tests/benchmark/suite01.example.json \
  --checkpoint "$CHECKPOINT" --output "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --correctness --results "$BENCH_DIR/01-replay-correctness"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --results "$BENCH_DIR/01-replay-normal"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --profile --results "$BENCH_DIR/01-replay-profile"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/01-model"
```

suite01 includes expert/two/layer and packed-direct/packed-single-expert for
L2 off/fixed/dynamic, with 06's overlap restriction. Prefill packed combinations
remain in suite05 and will be merged in final matrix. Freeze one shared input/
trace/config experiment, randomize candidate order and compare with baseline
including its legal overlap-on deployment. Final tuning/profile tools are next
after 01 code review; no validated acceleration profile is generated now.
