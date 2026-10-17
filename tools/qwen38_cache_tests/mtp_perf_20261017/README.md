# FP4-Local A16 + MTP diagnostic handoff

Start with [LIVE53.md](LIVE53.md): this is the **real production workload**, captured
while the user measured about 53 token/s. Earlier synthetic measurements have
higher acceptance lengths and must not be used as a direct production baseline.

## Runtime and findings

- Main routed MoE: prefill W4A4, decode/verify W4A16. Draft experts: resident W4A16.
- MTP steps=2, draft tokens=3; NEXTN is normalized to EAGLE in server_info.
- RTX 5090 D / SM120, single GPU, 5340 expert slots, pinned compressed PLE.
- Verify now uses the same cache as decode, fused GPU route planning, mapped-host
  expert fills, and shared-expert/fill overlap for up to 32 route positions.
- Live workload acceptance length is roughly 1.5–1.85, versus 2.5–2.6 in synthetic
  short prompts. Real verify cadence is ~33–35 ms/round.
- In the live trace, target A16 projections take ~11.5 ms/round and mapped expert
  fills ~11.0 ms/round. All 48 layers use the fixed path; explicit expert H2D
  memcpy and packed scatter are absent. GPU spans overlap: do not sum timings.
- Draft and target expert weights are distinct (`mtp.layers.0...` versus
  `model.layers.N...`). Verify-loaded target experts already persist in the
  per-layer decode cache until eviction; draft weights cannot replace them.

## Raw traces (included, not just summaries)

| File | Capture | Summary |
| --- | --- | --- |
| [a16-live53.trace.json.gz](traces/a16-live53.trace.json.gz) | Actual production request, 40 verify rounds | [live53-profile-summary.json](live53-profile-summary.json) |
| [mtp2-before.trace.json.gz](traces/mtp2-before.trace.json.gz) | MTP=2 before verify mailbox fix, synthetic | [mtp2-profile-summary.json](mtp2-profile-summary.json) |
| [mtp2-fixed-synthetic.trace.json.gz](traces/mtp2-fixed-synthetic.trace.json.gz) | MTP=2 with verify fix, synthetic | [fixed-profile-summary.json](fixed-profile-summary.json) |
| [mtp-off.trace.json.gz](traces/mtp-off.trace.json.gz) | Diagnostic MTP-off control, synthetic | [off-profile-summary.json](off-profile-summary.json) |

These are PyTorch/Kineto Chrome traces with CPU/GPU activities, not Nsight captures.
Open in Perfetto, or decompress to JSON if the viewer does not accept gzip.
`trace-manifest.json` records SHA256 hashes, byte counts, and original capture paths.

From the repository root:

```bash
python3 tools/qwen38_cache_tests/profile_service.py summary \
  tools/qwen38_cache_tests/mtp_perf_20261017/traces/a16-live53.trace.json.gz \
  --output /tmp/live53-summary.json
```

The summarizer matches GPU annotations by External id: CPU verify submission
returns before all queued GPU work finishes. CPU spans alone undercount device
work after the async fill fix. Normal throughput must be measured without a
profiler, and intervals containing trace export must be excluded.

## Code and deployment provenance

Code commit: `f3f4b80c868cf9927d94e907b71f81f8e26b6152`.

The accompanying code commit includes the previously uncommitted FP4-Local
checkpoint/kernel/PLE/MTP support required to reproduce this runtime, plus the
verify cache, sampling buffer, and PLE temporary-lifetime fixes.

`deployment/` contains review copies of the actual launch/config/controller files
from `/home/user/workspace/prod/homelab-llm`; these are provenance references, not
a standalone deployment directory. In particular, do not invoke its controller
without the original directory layout and model assets. No weights or pack are
included. The source helper is also available as `../profile_service.py`.

`deployed-source-manifest.json` hashes modified runtime files against the deployed
`fp4-local-verify-ple-ab0963e00024` snapshot. `deployed-vs-worktree.patch` records
the small remaining differences: parity test tooling and Triton load default
`0` versus `0.0`. Neither difference changes the deployed verify fix.

## Measurement history and limitations

- [VERIFY_FIX.md](VERIFY_FIX.md): mailbox fix, numerical checks, short end-to-end
  controls (39.5 before / 60.1 after; 53.0 MTP-off).
- [A16_STREAM.md](A16_STREAM.md): 1024-token stream measurements, short-context
  aggregate decode 70.9 token/s; synthetic 32K context 65.7 token/s. These are
  **not** the live production request above.
- [REPORT.md](REPORT.md): initial diagnosis; its temporary MTP-off setting is
  historical and superseded. Production remains A16 + MTP=2.
- `bench-off-overlapped-profile.json` is explicitly excluded: its third request
  overlapped profiling. It is retained to make the exclusion auditable.
- `combined-numerical-capture-failure.log` records one combined test run with a
  CUDA allocator capture error. The isolated verify numerical test passed;
  `verify-mailbox-isolated.log` and `verify-mailbox-numerical.json` retain it.
- `ple-prefill-memory.json`: exact output/state check; 8192×10240 BF16 short-conv
  temporary lifetime fix reduces isolated peak allocation by ~320 MiB. The
  original 32K OOM and successful retry are described in A16_STREAM.md.

No full quality evaluation or guarantee across all contexts/acceptance rates is
claimed. The next useful experiments are same-real-workload MTP steps=1 vs 2,
A16 target projection optimization, and expert-fill/prefetch effectiveness.

## Commit verification

The complete dependency-backed CPU suite was rerun in
`swallow-sglang:service-flash-next-f712e3118c75`: **276 tests passed**.
See `commit-container-cpu-tests.log`. The host Python lacks torch and cannot
run this complete suite; the container result is the relevant check.

## Shared launch log correction

Both FP4 presets invoke `sglang-fp4-local.sh`; A4 passes only `--decode-a4`.
The old A4 entry is an exec-only compatibility shim. The shared script now
also uses a single `sglang-fp4-local.log` for both precisions. The previous
precision-dependent log-name branch was removed. The currently running A4
log inode was renamed to the common path; the old A4 name is a symlink and
the previous common log is archived under the deployment model-backups.
No model restart or precision change was needed for this log migration.
