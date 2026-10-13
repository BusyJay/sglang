# 05 packed expert transfer — ready for GPU testing, code review pending

Parent is reviewed 06 r3 `bb8a71dc35356f145cd3c6950d25da2e767e6a70`.
This is a sequential patch on 06, not directly applicable to 04/a2. Setting L2
`off` supplies the single-05 ablation; this is not a byte-independent 04+05 tree.
Default `packed_decode=direct`, `packed_prefill=direct` preserves the existing
transport. No 01 batching is introduced. GPU evidence remains not_run.

## Settings and selection

Loader options: `packed_decode` and `packed_prefill` independently select `direct`
or `packed`; both default direct. `packed_min_bytes=4096`,
`packed_max_bytes=67108864`, `packed_min_copies=2` are unvalidated experimental
thresholds. Selection requires total expert payload within those limits and a
page with at least min_copies fragments. Contiguous bulk prefill and a single
contiguous sparse run remain direct; only fragmented sparse prefill can choose
one-expert packed fills. Unsupported tensor destinations (e.g. fixed L2 pack
seed destination) use direct. Resident alpha decode packs four payloads; alpha
off packs all six; sparse prefill packs all six. L2 hit/demotion remains D2D,
05 only handles the H2D fallback.

Startup precomputes byte segment layouts and immutable device destination
address/offset/count tables for every enabled target slot. One table allocation
and descriptor upload per layer; subsequent fills lookup prebuilt views and issue
NO descriptor H2D. Adaptive migration drains outstanding jobs/readers, moves
payloads, assigns new cache views, then rebuilds descriptor address tables before
new jobs can use them. Same-shape layer layouts are checked by the base runtime.
Targets must have contiguous matching rows. Full arena is shared, so its table
is shared across layers. Disabled mode allocates no scratch or descriptors.
Startup descriptor memory and scratch bytes are exposed by memory_budget().

## Lifetimes and failures

One scratch page lives for the runtime. Every payload H2D and scatter uses the
runtime transfer stream, so scratch overwrite follows the previous scatter.
Descriptors are immutable during jobs; there is no mutable descriptor ring or
per-fill descriptor staging. Pinned host ring reuse waits its existing event;
completion event is recorded AFTER scatter, even when a launch fails. Failed
copy/scatter makes the transfer job fail through existing worker poisoning, so
partial destination bytes cannot become a later cache hit. mmap views are released
on success/failure. No source alias is kept after staging.

Decode waits the target slot's prior reader before entering either direct/packed
fill. Prefill uses the existing arena_last_use dependency. Job final ready is
published only after weight/scatter/alpha work is enqueued; compute waits it.
Host plans, descriptor tables, caches and scratch remain runtime-owned while
retired jobs await compute. Packed fill handles ONE expert; no cross-expert
batching or intermediate ready deletion. A future 01 may group experts via a
separate reviewed interface, not by bypassing these protections.

Stats expose descriptor_table_uploads, descriptor_h2d_calls_runtime (zero),
scatter calls, payload bytes, calls saved, decode/prefill experts, target refreshes
and threshold fallbacks. Submission savings alone are not a speedup claim.
Packed adds device scratch write/read and scatter target writes, which can exceed
direct DMA cost; GPU comparison must include end-to-end wall/TTFT/TPOT and
normal/profile overhead. Host packed staging/enqueue profiling is opt-in.

## Validation and commands

CPU regression checks segmentation/tail bytes, thresholds/fallback, scatter failure
ring protection, mmap release, target table relocation, disabled direct path and
fragmented versus contiguous prefill selection. Existing 06 production-Lock and
periodic no-device-read tests remain. Offline compile includes the new scatter
wrapper. GPU tests are PREPARED but not run: unaligned/tail payloads with guards,
alpha on/off, ring wrap with two streams, target migration, L2+packed payloads.
Full NVFP4 MoE, long context/graph/KV scheduling and performance remain not_run.

```bash
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
docker run --rm --pull never --network none \
  -v /data/workspace/sglang:/workspace/sglang:ro -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 lmsysorg/sglang:v0.5.21 \
  tools/qwen38_cache_tests/compile_cuda.py
```

On GPU, from the patched root with local dependencies/pack/checkpoint, follow
GPU_HANDOFF.md and PATCH06_HANDOFF.md for input/tokenizer/session/trace checks.
Current version `oct13-05-r1`; do not resume 06-only results as current evidence.

```bash
export PYTHONPATH="$PWD/python"
python3 tools/qwen38_cache_tests/compile_cuda.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output "$BENCH_DIR/05-gpu-small.json"
python3 -m tools.qwen38_cache_tests.benchmark prepare-inputs \
  --suite tools/qwen38_cache_tests/benchmark/suite05.example.json \
  --checkpoint "$CHECKPOINT" --output "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark validate \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --correctness --results "$BENCH_DIR/05-replay-correctness"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --results "$BENCH_DIR/05-replay-normal"
python3 -m tools.qwen38_cache_tests.benchmark replay \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" \
  --trace "$TRACE" --trace-meta "$TRACE_META" --profile --results "$BENCH_DIR/05-replay-profile"
python3 -m tools.qwen38_cache_tests.benchmark run \
  --suite "$SUITE" --machine "$MACHINE" --inputs "$BENCH_DIR/inputs" --results "$BENCH_DIR/05-model"
```

The matrix keeps slots/chunk frozen, includes L2-off single-05 decode/prefill/both
and fixed/dynamic L2+05 combinations with required overlap-off. Add the separately
prepared suite06 controls when freezing the final combined matrix. Dedicated
replay backing remains non-KV; real model uses scheduler KV lending. No validated
accelerated machine profile is created. 01 and final tuning/profile tools follow
code review; GPU will later measure baseline and candidates independently.
