# Qwen3.8 verify experiments — validated defaults

2026-10-18, base `0d01fb2dc8e4c0d861fb16f5a77537936e598000`.
The original verify-experiments package was checksum-verified and tested on
RTX 5090 D / SM120 with the existing production image. See
[VALIDATION.md](VALIDATION.md) and [validation/](validation/) for evidence.

Defaults promote measured component or individual-sample benefits without
requiring stable end-to-end throughput gains. All recorded service samples
are retained; aggregate throughput is reported separately.

| Candidate | Default | Benefit | Rollback / opt-in |
| --- | --- | --- | --- |
| Capacity extensions | Up to 256 in analysis worker | Hot layers reach 256 at the same 6390-slot budget; positive individual service samples | `SGLANG_QWEN38_ADAPTIVE_CANDIDATE_MAX=` restores original grid |
| K=1 projection + SwiGLU | `fused` for split K=1 | About 30% less small-verify scratch; positive service samples | `SGLANG_QWEN38_W4A16_VERIFY_MODE=baseline` |
| Warp victim reduction | `2` | Miss-planner graph latency decreases 0.3–4.2% | `SGLANG_QWEN38_PARALLEL_MISS_PLAN=1`; `0` selects serial reference |
| Compact logs | `compact` | Historical service records shrink about 69%, losslessly | `SGLANG_QWEN38_CACHE_STATS_FORMAT=full` |
| Cross-token projection reuse | Off | Valid-route kernels were slower, including overlap | Explicit `SGLANG_QWEN38_W4A16_VERIFY_MODE=reuse` only |

Restart workers after changing modes. Target decode/verify remains W4A16;
MTP remains W4A4. Slots, memory fraction, KV budget and checkpoint are unchanged.
Fusion/reuse require split K=1. With no mode override, other split values keep
the baseline. Only 2/3-token verify with <=32 routes uses the new variants;
other shapes retain the existing path. No expanded resident expert layout is
allocated. Capacity defaults respect smaller expert sets and current layouts
already larger than 256. The direct allocator API keeps its explicit
`candidate_max` argument.

The compact encoder was repaired: Python equality previously merged `True`,
`1`, `1.0`, and signed zero, including nested values. Common-field comparison
now preserves JSON types and zero signs. `stats_snapshot()` keeps its original
structure. Canary and benchmark readers expand compact logs. Other consumers
can use `expand_stats()` or `cache_stats_decode.py` with a service log and
`--output /tmp/cache-stats-original.jsonl`.

## Reproduce

Use the existing GPU image with the checkout mounted and
`PYTHONPATH=/workspace/sglang/python`. The optional Dockerfile copies the
checkout into an image. No image archive or model weights are distributed.

```bash
python3 tools/qwen38_cache_tests/run_cpu.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output /tmp/gpu.json
python3 tools/qwen38_cache_tests/w4a16_verify_gpu_check.py --iterations 100 --rounds 3 --output /tmp/verify-kernels.json
python3 tools/qwen38_cache_tests/verify_experiments_20261018/graph_bench.py --output /tmp/graph-bench.json
```

Graph timing excludes Python enqueue gaps. Planner timing includes route D2D
copies; projections use resident synthetic weights. Both exclude expert fills
and full-service overhead. GPU entrypoints require the appropriate GPU.

`experiment_matrix.json` retains an explicit original baseline, individual
variants, and the promoted fused/capacity/warp combination. `run_service_ab.py`
reuses the previous MTP patch's validated production scripts/container command.
It launches isolated workers, flushes prefix cache before requests, retains all
outputs/meta-info, and refuses to overwrite existing rounds. The checked-in
27323-token code-review input fixes the workload. Use a fresh results directory
with the existing local production mounts, image and network:

```bash
python3 tools/qwen38_cache_tests/verify_experiments_20261018/run_service_ab.py --results /tmp/qwen38-default-check --variants defaults --tokens 2048 --near-limit
python3 tools/qwen38_cache_tests/verify_experiments_20261018/analyze_service_ab.py /tmp/qwen38-default-check
```

The comparison order was baseline/fused/planner/cache256/cache256/planner/
fused/baseline. Every worker used the same warmup, CPU-cache and Raft prompts,
two identical long code-context requests and four known-answer smoke checks.
The final default worker also runs a real approximately 218K prompt plus
128-token generation. Initial 1024-output diagnostics did not fill the 512-round
adaptive observation window; formal comparisons therefore use 2048 outputs.
Warmup and initial diagnostics are excluded from the comparison.

`service_bench.py` supports manually launched isolated workers and fixed prompt
files; `--flush` is for those isolated workers. Save acceptance length and round
time alongside tokens/s: generated paths and adaptive state varied across fresh
workers. Smoke checks are not a comprehensive model-quality evaluation. The
measured benefits do not establish a stable aggregate throughput improvement.
