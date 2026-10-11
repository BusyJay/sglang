# Lazy PLE checkpoint loading

For the `qwen38_nvfp4_cache` loader, add to the existing launch command:

```bash
export SGLANG_QWEN4_PLE_LAZY_CACHE_MB=512
export SGLANG_QWEN4_PLE_LAZY_STATS_INTERVAL_S=30
# Existing launch flags, model and expert manifest remain necessary:
python -m sglang.launch_server ... --load-format qwen38_nvfp4_cache \
  --ple-offload-backend lazy
```

The canary wrapper also accepts `--ple-backend lazy`. No `--ple-dir` or PLE
conversion is required. `pinned` remains the default; switch back to compare.

Startup reads safetensors headers, validates PLE shard coverage/dtype/width, and
keeps a meta weight parameter. It skips PLE `get_tensor()` and never copies or
pins the full PLE table. Files remain read-only and must stay available and
unchanged throughout serving. BF16 and `F8_E4M3` checkpoints are supported, with
no lossy conversion. Shard layout must be `ngram_embedding.shard_N.weight`.

The cache budget is **total per process across PLE tables**, excluding metadata
and staging. At 512 MiB, 160-byte FP8 rows hold roughly 3.35 million rows. Staging
adds up to 10 MiB per such table at the default
`SGLANG_QWEN4_PLE_LAZY_BATCH_ROWS=65536` (16 heads × 4096 tokens).
CPU index/copy temporaries and bounded read buffers add transient RAM usage.
Actual storage is capped to the number of checkpoint rows.

Each gather transfers its IDs to CPU once. Each bounded batch deduplicates IDs,
copies cached bytes into staging, sorts misses by file offset, and merges nearby
reads within 4 KiB gaps and a 256 KiB maximum span. Up to four read workers execute
bounded partitions rather than creating one future per row. Lookups with at
most eight read spans execute inline to avoid decode thread scheduling. SLRU promotes
reused rows into an 80% protected segment and admits cold rows into probation.
A large request is processed in batches; it never requires the entire request's
rows to fit in cache. A CUDA event fences staging reuse, including between
streams/calls. The existing Triton host gather converts raw FP8/BF16 staging
rows to BF16 output; out-of-range IDs emit zeros.

This first version performs the lazy lookup on the main CUDA stream at an eager
graph boundary. It does not defer disk I/O into the existing pinned-table
prefetch stream. This keeps CUDA capture/replay correct but adds CPU ID transfer,
cache bookkeeping and cold-miss latency. Eager execution and the existing
**threaded expert runtime + breakable decode graph** are supported;
prefill graphs, full graphs and torch compile are not supported by this loader.
Capture only allocates the output bridge; warmup/replay perform actual lookups.
No prediction, whole-table prefetch or cache population from capture-time IDs.

`preadv` still uses the Linux page cache. `POSIX_FADV_RANDOM` on private read
handles suppresses unnecessary readahead; it does not cap page cache residency.
A long cold prompt can still read many pages and displace expert pages. There is
no broad `DONTNEED` on shared checkpoint files because it would evict expert
weights too. Lazy loading primarily eliminates the full-table **startup** scan
and pinning, and should not be assumed to guarantee faster cold prefill.

## Compare on GPU

Keep expert-cache layout, adaptive settings, graph settings, prompt, chunk size
and output length identical. Compare `pinned`, `lazy` with 512 MiB, then `lazy`
with 1024 MiB. Record startup elapsed time/RSS/MemAvailable, logged KV token
capacity, cold and repeated prefill, and decode median tokens/s.

`PLE lazy ready` reports actual cache/staging sizes and confirms that startup
loaded no PLE payload. `PLE lazy stats` reports cumulative **unique row** hits,
misses, storage bytes, read calls and host lookup wall time (includes ID D2H,
staging fences and I/O; excludes waiting for the final gather kernel to finish).
Compare counter deltas between prefill and decode; the cumulative hit rate can
otherwise be dominated by a cold prompt. Logging is disabled with interval 0.
Warmup requests intentionally populate only the rows they touch.

CPU checks:

```bash
bash tools/qwen38_cache_tests/docker_cpu.sh lmsysorg/sglang:v0.5.21
```

GPU parity/capture/staging checks (run inside the GPU image with repo Python):

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py
```

CPU fixtures cover metadata-only loading, avoiding PLE payload materialization,
cross-shard raw FP8/BF16 byte parity, hits without I/O, SLRU eviction, short reads,
checkpoint identity and invalid offsets. GPU tests additionally cover output
parity, invalid IDs, bounded staging reuse, eviction, and changed IDs on multiple
breakable graph replays. GPU performance and these GPU tests need a CUDA device.
