# LFU / LRU 固定路由 transport profile

这是独立进程中的解释性采样，不属于正式的 5 次重复计时。当前 FP4-Local pack、RTX 5090 D、6390 slots，先从空 cache 回放相同 128 sweep，再采集同一 coding tape 的后续 8 sweep。每个 sweep 为 48 层、3×10 原始 route IDs；每种策略各 384 次层调用。没有 attention、GDN、draft、B12X 或 shared expert 计算。

| 实际 CUDA kernel | LFU 累计 / 平均 | LRU 累计 / 平均 |
|---|---:|---:|
| decode_plan | 10.32 ms / 26.87 μs | 50.83 ms / 132.37 μs |
| mailbox_fill | 59.38 ms / 154.63 μs | 73.15 ms / 190.49 μs |

该段 LFU miss 946，搬运 2,615,500,800 bytes；LRU miss 1167，搬运 3,226,521,600 bytes。两者的 fill kernel 有效搬运速率都约 **44.1 GB/s**。LRU 约多搬运 23.4% 数据，其 fill CUDA 时间约多 23.2%。

更大的额外开销出现在 planner：LRU 的平均 decode_plan 时间约为 LFU 的 **4.93 倍**。在这段短 trace 的额外累计 kernel 时间中，约 75% 来自 planner；这个占比不外推为完整 896 sweep 或线上请求的收益分解。

源码解释与采样一致：`python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu` 的 `decode_plan_kernel` 仅对 LFU 启用 all-hit 快路径及并行 miss planner；LRU 在工具中用 SLRU(protected_fraction=0) 实现，和其他 SLRU 一样进入通用 `apply_plan`。该路径串行更新 active experts，并在需要时扫描 victim 或 protected slots。因此本轮比较同时包含**策略命中效果**和**现有 GPU 实现成本**，不能把 LRU/SLRU 的较大耗时只解释为 miss 数较多。

以下为 `llm-torch-profiler-analysis` 的单 trace 三表输出。源位置由 trace 关联得到；没有提出完整模型的 overlap/fusion 结论，因为该回放省略了模型计算。原始 trace 压缩存于 `live_fp4_results/profiles/`，并保留原始 kernel 汇总及 profile-only 计数。


## LFU

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /home/jay/workspace/prod/homelab-llm/models/cache-policy-live-20261018-r2/profiles/lfu-decode.trace.json

### Kernel Table

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void (anonymous namespace)::mailbox_fill_kernel<true> | memory | 59.38 ms | 84.6% | 384 | python/sglang/srt/layers/moe/qwen38_cache_mailbox.py:203 submit_fill | cudaLaunchKernel |
| (anonymous namespace)::decode_plan_kernel(int const*, int*, int*, int*, int*, float*, long*, long*, long*, int*, int, int, int, float, int, int, bool, long*, int*, int) | other | 10.32 ms | 14.7% | 384 | python/sglang/srt/layers/moe/qwen38_cache_mailbox.py:131 run | cudaLaunchKernel |

### Overlap Opportunity Table

| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

### Fuse Opportunity Table

| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |


## LRU

Triage View
Mode: single-trace
Framework: SGLang
Input traces: /home/jay/workspace/prod/homelab-llm/models/cache-policy-live-20261018-r2/profiles/lru-decode.trace.json

### Kernel Table

##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| void (anonymous namespace)::mailbox_fill_kernel<true> | memory | 73.15 ms | 58.7% | 384 | python/sglang/srt/layers/moe/qwen38_cache_mailbox.py:203 submit_fill | cudaLaunchKernel |
| (anonymous namespace)::decode_plan_kernel(int const*, int*, int*, int*, int*, float*, long*, long*, long*, int*, int, int, int, float, int, int, bool, long*, int*, int) | other | 50.83 ms | 40.8% | 384 | python/sglang/srt/layers/moe/qwen38_cache_mailbox.py:131 run | cudaLaunchKernel |

### Overlap Opportunity Table

| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

### Fuse Opportunity Table

| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |
