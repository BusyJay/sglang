# 06 r3 response: boundary lock deadlock

Review object: `8fee09d118de4d3d56f0cc58409032ebfd69ea6b`.
Review source: reviewer-owned `PATCH06_R2_REVIEW.md`.
Only the lock/drain regression is changed; the accepted configuration/periodic
statistics fixes remain. Defaults off; no 05/01 implementation; GPU deferred.

The production runtime keeps its existing `threading.Lock()`. Device runtime
synchronization is split into a public `synchronize()` that acquires the lock
and `_synchronize_locked()` that drains metadata, transfer, slot-reader and
arena events without acquiring it again. Both the public method and detailed
boundary call this same helper. Boundary retains its lock across jobs/retired
queue joins, worker error checks, device drain, statistics and keys read. Keys
are not moved outside the protected interval. Periodic snapshots stay CPU-only;
replay boundary reads remain outside the timed interval.

The prior regression was insufficient because it used RLock and mocked the
synchronization method. The replacement uses ordinary Lock and the production
public synchronize/helper/boundary/error methods. Only CUDA primitives are
mocked; every primitive and keys read asserts the lock cannot be acquired by a
new submission and both queue unfinished counts are zero. Each call runs on a
daemon thread with a two-second join limit, so a nested-lock regression fails
boundedly. It covers off and attached modes, empty/completed queues, queue work
completed by a separate worker, and worker-error propagation without device reads
or a leaked lock. Periodic device-read rejection remains covered.

Reproduction commands:

```bash
tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
docker run --rm --pull never --network none \
  -v /data/workspace/sglang:/workspace/sglang:ro -w /workspace/sglang \
  -e PYTHONPATH=/workspace/sglang/python -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 lmsysorg/sglang:v0.5.21 \
  -m unittest tools.qwen38_cache_tests.test_cpu_l2_review_regressions -v
```

Suite version `oct13-06-r3` rejects resume of old deadlocking r2 results.
Prior validation records remain `PATCH06_R1_VALIDATION.json` and
`PATCH06_R2_VALIDATION.json`; current evidence is `PATCH06_VALIDATION.json`.
GPU kernel/payload/MoE, real scheduling/KV/graph/session/long-context and performance
checks remain not_run. Passing CPU mocks verify lock semantics, not GPU correctness.
