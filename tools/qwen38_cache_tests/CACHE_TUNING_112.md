# Qwen3.8 expert cache tuning baseline (`112` slots/layer)

Date: 2026-10-11 (CST)

This note records the current working high-cache configuration for the
Qwen3.8-Flash-Next NVFP4 packed expert runtime on a single RTX 5090 D.

## Command

```bash
bash tools/qwen38_cache_tests/docker_model.sh \
  --image swallow-sglang:v0.5.21 \
  --device 0 \
  --checkpoint /home/user/workspace/prod/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --pack /home/user/workspace/sglang/results/packs/assembled \
  --results /home/user/workspace/sglang/.agent-workset/artifacts/cache-hit-ignore-eos-20261011-0112/slots-112 \
  -- canary \
  --runtime-mode threaded \
  --slots 112 \
  --contexts 8192 \
  --context-max-new-tokens 256 \
  --context-ignore-eos \
  --cache-stats-log-interval 1024 \
  --mem-fraction-static 0.90 \
  --mamba-full-memory-ratio 0.22 \
  --max-mamba-cache-size 15
```

## Result

- Request status: passed
- Context: `8192` prompt tokens, `256` completion tokens, `ignore_eos=true`
- Request wall time: `51.903s`
- Cache hit rate: `0.8361` (`103307` hits / `20258` misses)
- Cache path invocations: `12240`
- Arena path invocations: `48`
- Decode throughput summary from `server.log`:
  - average: `14.03 tok/s`
  - max: `16.80 tok/s`
  - last: `16.80 tok/s`

## Decision

- Under the current memory knobs, treat `112` as the practical high-cache
  baseline.
- `128` slots/layer does not start: after loading weights there is no
  remaining KV-cache headroom, so the server exits during startup.
- Raising slots improves cache hit rate and decode throughput, but request
  latency is still mixed by prefill behavior. If we want the next gain, the
  next experiment should separate prefill policy from decode cache capacity.

## Evidence

- Report: `../../.agent-workset/artifacts/cache-hit-ignore-eos-20261011-0112/slots-112/report.json`
- Server log: `../../.agent-workset/artifacts/cache-hit-ignore-eos-20261011-0112/slots-112/server.log`
- Comparison summary: `../../.agent-workset/artifacts/cache-hit-ignore-eos-20261011-0112/summary.json`
