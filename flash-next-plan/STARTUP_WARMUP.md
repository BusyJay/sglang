# Representative startup warmup for Flash-Next offload

Enable on the Python HTTP server with:

```text
--warmups qwen38_offload
```

For the existing canary/Docker canary, add `--offload-startup-warmup` after the
`-- canary` separator. The canary passes `--warmups qwen38_offload` to the server.
Do not confuse this with `--warm-decode-cache`, which seeds decode slots during
prefill and is an independent optional feature.

The custom warmup uses the existing pre-listener FastAPI startup registry and
TokenizerManager directly. It completes requests with 64/512/2048 prompt tokens
and 16 generated tokens each, `ignore_eos=true`, temperature zero. Inputs use IDs
from the actual tokenizer. Small configured contexts cap and deduplicate shapes.
An aborted or empty response raises rather than claiming successful warmup.

This exercises routing/planner, the full arena path, sustained decode cache
miss/hit activity, and representative kernel/runner initialization before normal
traffic. It deliberately does not enumerate 220K shapes or all 512 experts in
decode. Warmup extends startup time; it moves cold costs rather than removing them.

Limits:

- Only non-disaggregated Python HTTP serving is supported. Rust TreeCore is
  independent and can be enabled; the separate Rust HTTP server path does not
  execute this FastAPI custom-warmup registry.
- Does not warm OpenAI chat template/reasoning parsing, individual tool schemas,
  xgrammar constraints or every batch/shape-dependent compilation path.
- Runtime expert caches retain sampled experts, but a different real request can
  miss. Real long prompts still require arena loads, regardless of warmup.
- Startup readiness/load-balancer timeout must allow the added warmup duration.
- Persistent JIT caches should be mounted separately for repeated restarts.

Local CPU suite: 97 tests, 84 passed, 13 skipped (PyTorch unavailable), including
actual async warmup control flow through tokenizer/request mocks, shape limits,
forced decode, failure propagation and canary flag forwarding. AST/diff checks
pass. No weights downloaded and no GPU warmup timing claimed.

On the target GPU compare the first and second identical conversations after
startup with warmup on/off, recording startup duration, first-token latency and
kernel compile messages. Keep real request parameters identical and separate
prefix-cache benefits from one-time compilation effects.
