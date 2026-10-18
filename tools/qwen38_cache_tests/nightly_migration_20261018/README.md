# FP4-Local nightly migration — 2026-10-18

The live `fp4-local` language worker and remote ViT now use
`swallow-sglang:nightly-dev-20261018-b234b5c3`, image ID
`sha256:f4f3b73ba30219458bcbe3fd824837a2552f686a345fde0cd102c80139f8f3e0`.
The configured source mount is `/home/jay/workspace/sglang` →
`/workspace/sglang`; the launcher sets `PYTHONPATH=/workspace/sglang/python`.
B12X is enabled for BF16 target decode/verify with 1–3 tokens; CUTLASS W4A4
prefill and MTP W4A4 remain active. Full breakable target and draft graphs are
working. No version or backend rollback was performed for the final deployment.

## Exact history

- Offload: `d0694c62ea0e93ee2c36813f70d459072d6c2516`.
- Upstream/image base: `b234b5c32e0d65e04a61912f85fc09e95e74a328`.
- Merge: `fb596abe56ad140d021347cac6d91921c8f59ca1` (offload first parent,
  upstream second parent).
- B12X patch: `65e6b84fa9fe3aabf624e47849e225d0d36081af`.
- These exact commits were imported from the supplied lightweight bundle;
  SHA256SUMS and Git bundle prerequisites passed. Runtime fixes below are
  subsequent commits, preserving the transferred history.

The primary working branch is `qwen3.8-flash-next-offload` in
`/home/jay/workspace/sglang`. It includes the exact merge, B12X patch, runtime
fixes and profile commits above. Legacy `nvfp4` is preserved at
`d0694c62ea0e93ee2c36813f70d459072d6c2516` on
`codex/nvfp4-backup-20261018`, checked out in
`/home/jay/workspace/sglang-nvfp4-backup-20261018`; its preset mounts that directory.

This branch/config correction did not restart either serving container.
The existing containers retain their original bind mount from
`/home/jay/workspace/sglang-nightly-b12x` at `f2466a3774`, which contains the same
runtime implementation. That worktree is retained only for the current live
containers; subsequent development uses the primary branch. The next normal
container recreation will apply the configured primary source directory.
Historical measurements and deployment snapshots retain their original paths.
Current preset snapshots are in `deployment/branch-layout-20261018/`.
The `fp4-local-a4` preset inherits the new image/source; its launcher explicitly
selects Triton when W4A16 is disabled, avoiding B12X's invalid A4 configuration.
The A4 preset passed configuration preflight; the serving measurements below
are for `fp4-local`.

## Measured service throughput

Same RTX 5090 D, model/revision/pack, 6390 expert slots, MTP=2, BF16 Mamba,
FP8 KV, 220000 context limit, 8192 prefill chunk, static fraction 0.90 and
single-request scheduling. Each version received a 192-token warmup followed
by two rounds of three fixed prompts, temperature=0, ignore_eos=true,
1024 output tokens each. No GPU microbench or profiler ran during these macro
measurements. Sustained decode uses the first and last streamed token timestamps.

| Deployment | Median sustained decode | Median end-to-end | Median MTP accept length |
|---|---:|---:|---:|
| Old image + offload d0694c62 | 90.62 token/s | 88.06 token/s | 2.429 |
| New image + merged source + Triton | 94.81 token/s | 92.06 token/s | 2.441 |
| New image + merged source + B12X (live) | **100.79 token/s** | **97.61 token/s** | 2.444 |

Final sustained decode improves **11.22%** over the old deployment and **6.30%**
over merged+Triton. End-to-end improves **10.84%** over old. These are six
short-prompt, single-request samples per version, not a claim about every
context length, concurrency or production workload. Outputs are not required
to be bit-identical across versions/backends. Raw results and timestamps are
in `old-round*.json`, `new-triton-round*.json`, `new-b12x-round*.json`; regenerate
`summary.json` with `python3 tools/qwen38_cache_tests/nightly_migration_20261018/summarize.py`.
`diagnostic-eager-triton-*` is an interrupted diagnostic with target graphs
bypassed; it is excluded from all final comparisons.

## Runtime faults fixed during migration

1. The merge retained an FFN call to `get_forward()` but removed its import.
   Restore the import; the unmodified bundle fails during target graph setup.
2. Main's prefill PLE/QSA eager bridges used `ForwardMode.is_extend()`, which
   also includes TARGET_VERIFY. Restrict those bridges to
   `is_extend_without_speculative()` so verify does not read a nonexistent
   TC-prefill context.
3. `_get_prefetch_buffer` routed all breakable captures to an expandable eager
   buffer. Target verify captures the PLE gather, and subsequent large eager
   prefill replaces/frees that buffer. Replay continued writing its old address;
   a new verify retrieval-index allocation reused the address and was overwritten
   with floating-point-looking bytes. GPU synchronization localized the first
   overwrite to target graph segment 0, before expert transport. Keep dedicated
   shape-keyed graph buffers for target verify (no TC-prefill context), retaining
   the live eager-buffer behavior for prefill bridges. The CPU regression checks
   verify buffer identity across an intervening 8192-row eager allocation.
   `corrupt-segment.dot` and startup/blocking logs retain the failure evidence.
4. B12X's scratch-use event could not wait on a prior eager recording during
   CUDA Graph capture. Use an external CUDA event so the wait/record dependencies
   survive capture/replay. All 24 numerical, graph, stream and pool-replacement
   cases then passed.

Target-eager, all-eager and strong-input-reference settings were diagnostic
experiments only. None is enabled in the final deployment. No CUDA blocking or
allocator-history tracing is left enabled.

## OOM and memory

No CUDA OOM reproduced with the merged source and final launch settings.
Loading target and draft weights consumes approximately 22.25 GiB + 1.48 GiB,
matching the old deployment at the log's precision. B12X reserves roughly
7 MiB of scratch across the target's 48 layers before KV sizing. Resolved KV
capacity is 221248 old, 221184 new+Triton and 220608 new+B12X; all accommodate
the configured 220000 context limit. Expert capacity remains 6390 slots.

The long-prefill regression passed: 23426 input + 1024 output tokens, with a
second 46825-input/64-output request queued during the first request's decode.
Actual 8192-token chunks executed with MTP enabled. GPU memory sampled every
0.5 seconds peaked at **32196 MiB / 32607 MiB**, leaving 411 MiB at that sample.
This is a tight observed margin, not a proof that every possible workload is
OOM-free. The user's original direct-migration OOM has no recovered trace here,
so it cannot be conclusively attributed to a particular upstream change.
The reproduced failures above were missing context/import and stale graph
addresses, not allocator OOM. See `long-prefill.json` and
`long-prefill-gpu-memory.json`.

## Validation and practical limits

- 334 CPU tests plus SGLang imports/config/runner checks: passed (`cpu.log`).
- Six B12X offline compile specializations: passed (`compile.log`).
- 24 isolated-GPU parity/graph/stream cases plus replacement capacities:
  passed; maximum absolute error 0.0001220703125
  (`gpu-parity-and-latency.json`, `gpu.log`).
- Text arithmetic, actual remote ViT red-image request, forced OpenAI tool
  call and OpenAI SSE: passed (`api-acceptance.json`).
- Long-prefill after sustained decode: passed. Both final service containers
  remained running without restart during acceptance.
- Image/runtime: PyTorch 2.14.1+cu130, FlashInfer 0.7.2, SM120 RTX 5090 D.

The isolated microbench measures a synchronous call envelope: B12X 1/2/3-token
calls were about 168/179/192 µs versus Triton fused 122/79/105 µs. B12X graph
replay was about 28/40/52 µs. Those diagnostic numbers exclude planner, PCIe
fills and other model work and did not predict the complete service result;
the measured serving result above is the basis for the deployment conclusion.

Configuration lives under `/home/jay/workspace/prod/homelab-llm` in
`models/fp4-local.yml`, `models/catalog.json` and
`home-jay/remote-mapping/llm/sglang-fp4-local.sh`. The pre-migration copies are
under `model-backups/nightly-b12x-20261018`. Continue using `./llmctl fp4-local`.

## Live profiling follow-up

2026-10-18: CPU/GPU traces from running production requests, overlap-aware
analysis, Python stacks/shapes and review directions are in [profile/README.md](profile/README.md).
This follow-up preserves the deployed runtime implementation and configuration.
