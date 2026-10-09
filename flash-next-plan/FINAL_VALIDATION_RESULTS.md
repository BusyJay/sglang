# Qwen3.8-Flash-Next NVFP4 dynamic offload - SM120 final validation results

Date: 2026-10-10 (CST) · Run dir: `.agent-workset/artifacts/gpu-switch/20261010-025326`
Source commit: `53b8daf3c2` (branch `qwen3.8-flash-next-offload`, base v0.5.21 e00930c548)
Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4` revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` (local offline cache only)
Image: `swallow-sglang:v0.5.21` · GPU: RTX 5090 D (SM120), switched via gpu-switch workflow with automatic prod stop/restore around every batch.

All four stages passed (`stages/final-validation-summary.json`, status=passed):

| Stage | Result | Wall time | Notes |
|---|---|---|---|
| layer0 sync parity | rc=0 | 27s | cached vs resident CUTLASS per-case max_abs_error = 0.0 across decode_miss/hit/fill*/prefill_arena; hits=10 misses=70 arena_loads=1 |
| layer0 threaded parity | rc=0 | 27s | same case set, identical 0-error parity in the threaded transfer runtime |
| assemble all layers | rc=0 | 194s | 48 layers exported + assembled into `results/packs/assembled/experts.bin` (~64.5 GiB) + manifest |
| canary threaded 8K/32K | rc=0 | 210s | real server on packed cache, thread mode, slots=56/layer arena enabled; weight load 33.8s; GPU weights 18.07G / kvcache 7.80G / startup avail 2.38G; token_capacity 641088 |

Canary cases (`stages/canary-threaded/report.json`):
- 8192-token prompt -> 32 output tokens, e2e 13.4s, finish_reason length (no download: weights_downloaded=false)
- 32768-token prompt -> e2e 23.6s, cached_tokens=8192 device-side prefix reuse confirmed

Memory profile used by stage D (proven production values): `--mem-fraction-static 0.90 --mamba-full-memory-ratio 0.22 --max-mamba-cache-size 15`.

Raw evidence paths (excluded from git via .git/info/exclude):
- run logs: `.agent-workset/artifacts/gpu-switch/20261010-025326/stages/*.log`
- parity reports: `results/packs/layer0-{sync,threaded}/layer-000/parity.json`
- canary report/server log: `.../stages/canary-threaded/{report.json,server.log}`
- assembled pack: `results/packs/assembled/`

Production verification (independent of supervisor claims), performed before/after every dispatch: `/health_generate` HTTP 200, GPU0 ~31.4G resident for prod llm, GPU1 vit untouched, fresh traffic lines in `prod/homelab-llm/home-jay/remote-mapping/llm/sglang.log`.
