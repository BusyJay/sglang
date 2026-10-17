Validation on 2026-10-17, RTX 5090 D, swallow-sglang:v0.5.21-qwen38-capacity
(image sha256:f712e3118c757894f77ebab49081badab75e47fb16523a088435ec1910a37f59):

* Fixed duplicate memory_budget() definitions: borrowed-arena accounting now runs
  in the existing method, retaining adaptive, migration, alpha and transport fields.
* Updated GPU regression expectations for overlapping arena/cache storage.
  Separate cases cover non-adaptive, adaptive with borrowing off, and borrowing on.
  Borrowed tests assert storage aliasing, exact byte accounting, actual expert
  restoration, and output parity across compute streams.
* 66 CPU regression tests passed.
* All 38 test_gpu_cache tests passed on GPU (47.065 seconds).
* Full-size BF16 embedding (248320 x 2560): byte parity and CUDA Graph replay passed
  for 1, 3, 128 and 4096 tokens. GPU weight allocation saved 1,271,398,400 bytes
  (1212.5 MiB); mapped weight GPU allocation was zero.
* Embedding latency assessment still FAILS its original <=5% prefill overhead
  gate. Isolated 4096-token gather: 629.14 us mapped vs 10.00 us GPU;
  128-token gather: 19.20 us vs 6.74 us. No assessment thresholds were relaxed. See results/fixed for measured
  latency and the isolated rerun; functional success is not a production PASS.
* Regenerated patches pass clean-base forward application and reverse rollback
  using a temporary Git index. The working index was not modified.

Evidence: bundle results/fixed/cpu-regression.log, gpu-regression.log,
embedding-isolated.json, result-isolated.json. No service was restarted.
Real-model TTFT/TPOT/throughput and production arena savings remain unverified.

2026-10-17 整模型复测：两项优化均已部署。embedding 节省 1212.5 MiB，arena 借用 1350 MiB，独立 arena 为 0；token 容量 235456→425088。原服务挂载旧快照，现已改为始终挂载 /home/user/workspace/sglang。两次主模型加载为 59.56/124.75 秒（原 196.59 秒）；缓存冷热影响明显。完整记录见 tools/qwen38_cache_tests/startup_memory_20261017/README.md。

2026-10-17 OOM 修订：增加 05-bounded-mtp-prefill.patch。MTP draft extend 的 MoE 临时量按 128 tokens 分块，8192-token GPU 峰值从 1020 MiB 降到 55.94 MiB，输出逐字节一致；外部 chunk=8192 和 KV 容量不变。GPU manifest 增加长 MTP 分块/graph 回归。详情见 tools/qwen38_cache_tests/decode_prefill_oom_20261017。
