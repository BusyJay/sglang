# Mixed-precision MTP HiCache startup failure (2026-10-24)

## Failure and replay

The production startup failed in `prepare_host_pool_config`, before the service
became ready:

```text
ValueError: packed draft KV row (2, 256, 256, torch.bfloat16)
differs from target KV row (2, 256, 256, torch.uint8)
```

The target's logical FP8 cache is stored as bytes; the draft intentionally uses
BF16. The speculative worker selected packed host layers solely from the MTP
algorithm/depth, although packed layers must share the target row's dtype.
This is a HiCache host-layout selection failure, not an out-of-memory failure
or evidence that one of the three optimization switches crashed a CUDA kernel.

The production log and resolved arguments were saved locally as
`production-failure.log` and `production-args.json`. An isolated launch with
HiCache, target FP8, draft BF16, packed expert transport, the production expert
budget and memory fraction reproduced the same exception (`replay-before.log`,
exit 137 after child-process failure cleanup). Frontend ports/model paths were
adapted to the existing test container; this does not deploy the service.

## Fix

Commit `e96bc2d50a` selects the existing dtype-preserving KV/indexer sidecars for
a single MTP draft when target/draft KV dtypes differ and recurrent state is
shared (or absent). Shared Mamba state remains backed up with the target.
Same-dtype packing is unchanged. The packed-only external linker still rejects
this sidecar configuration. Multi-depth/separate-state packing is not silently
changed or truncated.

No cache precision, HiCache enablement, MTP setting or production launch script
is reduced to work around the failure.

## Validation

- Five new mixed-precision plan cases failed on the old implementation.
- Plan/assembler/host sizing tests after the fix: **74 passed, 35 subtests**.
- The isolated production-config replay starts successfully and passes all
  **21 output/retrieval checks**, including 32K/170K/219K documents and output
  checks after KV growth. See the local
  `../csf_defaults_20261023/hicache_startup_20261024/r3-000/result.json`.
- Commit `075d79de50` adds a real CUDA round trip built through the production
  draft-sidecar assembler: FP8 target KV, BF16 draft KV and both QSA indexers are
  filled with distinct byte patterns, copied to reordered CPU pages, cleared on
  GPU, then restored to different GPU pages. All bytes must match exactly.
  The complete QSA HiCache GPU file passes: **7 passed**.
- Applicable pre-commit hooks pass.

## Additional coverage observations

The extended matrix also requests a 170K continuation after the 219K workload.
The returned answer is correct, but the first probes report zero cached host
tokens. These probes **do not establish an end-to-end host cache hit** and must
not be described as successful cache-policy/load-back coverage. The explicit
GPU/CPU transfer test above proves the sidecar byte-transfer contract separately.

A diagnostic that doubled `hicache_ratio` to 12 hit CUDA OOM during 32K QSA
index expansion (66 MiB requested, 46.5 MiB free), with all three optimization
switches off. This diagnostic is excluded from the switch matrix. The original
ratio 6 configuration completes the 219K tests; the production ratio is not
changed.

Production-config Torch profiling also reported `CUPTI_ERROR_NOT_INITIALIZED`
and emitted CPU-only events. Those traces cannot establish GPU dispatch or
GPU-time shares. Earlier HiCache-off traces with actual GPU events are retained
separately; unavailable GPU evidence must not be inferred from absent events.

## Reproduction

`replay-command.json` retains the adapted production command and environment.
`source-manifest.json` records the frozen validation source before the later HC
default promotion. The simplified startup/default check can be replayed with:

```bash
docker exec -w /workspace/sglang qwen38-csf-defaults-20261023 \
  /opt/sglang/bin/python -u \
  tools/qwen38_cache_tests/all_optimizations_20261024/run_serving_ab.py defaults --quick \
  --reference /workspace/sglang/tools/qwen38_cache_tests/hicache_startup_20261024/replay-command.json \
  --output /results/hicache_startup_20261024/fresh-default-validation
```

Archive a previous output directory before a fresh replay; the runner does not
overwrite evidence. Test servers are stopped after each case. Production was
not started or deployed by these checks.
