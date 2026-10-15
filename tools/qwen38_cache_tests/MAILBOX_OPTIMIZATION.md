# Mailbox experiment 1: Direct mmap reads

Mailbox fill reads immutable registered expert-pack tensor addresses directly. The CPU worker publishes counters/readiness only; no Python payload staging allocation or copy remains.

This patch is independent and targets the mailbox baseline from review head
`44ed74ebd519621b821fbaec34f6faf0890c44f1`. Apply only one experiment per checkout;
experiment 3 already includes experiment 1.

The mailbox retains a zero-copy torch.frombuffer view of the read-only pack mmap
and registers the entire mapping once with cudaHostRegisterMapped |
cudaHostRegisterReadOnly. Registration pins file pages and can be expensive at
startup or fail if host memory/driver support is insufficient. No staging
fallback silently substitutes for registration failure. Memory accounting includes
the registered mapping. Shutdown drains GPU readers, unregisters, and drops the
view before the caller closes the pack. Only one live mailbox may register a
particular mapping at a time. Migration refreshes tensor destinations after drain.

Scope is single-token eager MoE mailbox decode outside adaptive sampling windows.
Ordinary prefill, warm-up, and sampling transport continue to use their existing
staging buffers. Experiments 1/2/3 remove Python staging from the mailbox path.

Run from the repository root on a GPU machine with the existing SGLang environment:

```bash
PYTHONPATH="$PWD/python" python3 -m unittest tools.qwen38_cache_tests.test_gpu_mailbox_bytes -v
PYTHONPATH="$PWD/python" python3 -m unittest tools.qwen38_cache_tests.test_gpu_cache -v
PYTHONPATH="$PWD/python" python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output /tmp/mailbox-opt1-gpu.json
```

The new byte test covers 1/2 misses, distant expert/slot IDs, large aligned tensors
with multiple tiles and partial tiles, and unaligned byte tails. Runtime tests
cover every required tensor, resident and nonresident alphas, duplicates, all hits,
and cross-stream slot reuse. Existing tests cover sampling/migration and failure.
CUDA runtime execution and performance must be validated on the GPU machine.

For throughput, retain identical prompts, routes, cache budget, output length,
and observation phase across baseline and experiment. Run alternate A/B orders
without profiling. Record client TPOT p50/p95, tok/s, host locked memory and
startup time. Nsight should show mailbox_fill_kernel reading the mmap (plus staged multi-miss payloads for experiment 4),
and host payload staging disappears for mailbox calls.


## RTX 5090 D compatibility (2026-10-15)

The delivered read-only host registration returns CUDA 801 on this machine.
The mailbox now owns an `ACCESS_COPY` private writable mapping and registers it
without `cudaHostRegisterReadOnly`. The checkpoint and the pack reader's original
read-only mapping remain unchanged. CUDA pinning can materialize private COW
pages: budget another pack-sized host allocation, in addition to existing cache
and HiCache memory. Closing drains readers, unregisters, releases the tensor
view, and closes the private mapping, including initialization failure cleanup.
Control allocations are initialized with an explicit CPU default device so
model loading under a CUDA default device also works.

GPU regression and CPU regression passed in the local CUDA 13.0 production image.
Full-model performance measurements and raw logs are recorded in the main
checkout's `results/mailbox-optimization-20261015/`. Use the final deployment
report for the chosen implementation and measured tradeoffs. GPU regression
success alone does not imply a throughput improvement.
