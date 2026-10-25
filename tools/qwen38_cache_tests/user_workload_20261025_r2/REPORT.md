# Hermes 第二轮：开启 cached-FC1 overlap 的性能诊断

运行代码为 `9e1daf70469a6091de8ca9a508ab63cc5ca7c619`。复用 Hermes → homelab-llm（5090 D）/ homelab-vit（4080）拓扑，只在 loader 配置中新增 `decode_early_cache=true`，保留 MTP 2 steps / 3 draft tokens、8192 chunk、220k context、FP8 KV、自动 expert capacity 与分段 decode graph。[配置](effective-config.json) · [部署补丁](deployment.patch) · [验证](launcher-validation.json)。

**结论：feature 已实际运行，计算与 fill 的并发明显增加，decode 观测性能改善；同时暴露了 rebalance 后约 1.84 秒的首次新容量冷路径，值得优先评审。** 不能把观测的 +9.6% 吞吐全部归因于 feature，因为两轮输入、输出、路由、随机种子和采集条件不完全一致。

## 观测对比

主请求的日志比较排除了第一条 decode 区间，因为它包含 prefill / 前序空闲时间；第一轮后续 title/tool 请求不混入日志吞吐比较。[逐区间数据与方法](log-comparison.json) 可核查取值。完整 trace 的指标则涵盖各自捕获的全部请求。

| 指标 | 第一轮，关闭 | 第二轮，开启 | 解读 |
| --- | ---: | ---: | --- |
| 主请求输入 tokens | 25,776 | 25,560 | 不完全相同 |
| 主请求稳态日志区间数 | 6 | 10 | 生成长度不同 |
| 主请求稳态吞吐中位数 | 80.45 token/s | 88.16 token/s | 观测 +9.58%，不是隔离后的加速比 |
| 同组日志 accept length 中位数 | 1.645 | 1.725 | 也影响输出吞吐 |
| 完整 capture 平均 accept length | 1.73475 | 1.7275 | 第一轮包含追加请求，与上一行口径不同 |
| VERIFY CPU forward 中位数 | 12.509 ms | 9.668 ms | 观测下降；包括不同路由与采集影响 |
| 相邻 VERIFY 开始间隔中位数 | 19.277 ms | 17.640 ms | 是完整迭代节奏的代理，不等于单 kernel 时间 |
| 主请求 target prefill CPU span 总和 | 6.170 s | 6.149 s | 变化很小，最后一段输入也更短 |
| 完整 capture VERIFY 次数 | 4,105 | 4,673 | 绝对 GPU 总时间不能直接当作加速比比较 |
| fill 与别流 kernel 同时运行的比例 | 9.34% | 31.20% | 两轮均用完整 trace、相同定义计算 |
| 每 VERIFY 的未与别流 kernel 并发的 fill 时间 | 5.921 ms | 4.616 ms | 观测下降约 22%；不是可直接兑现的延迟节省 |

第二轮没有运行第一轮未正常导出的 py-spy 采样器。两轮都记录 CPU/GPU profiler 活动，关闭 shapes / Python stacks。主请求也不是相同的冻结 payload，重启后的随机种子和 adaptive cache layout 不同。因此适合确认路径与瓶颈，不足以宣称 feature 净加速 9.58%。

## 路径实际生效与重叠

最后周期快照中 `decode_early_cache_invocations=210863`，和 `decode_device_gate_invocations` 一致；第一轮前者为 0。第二轮 trace 出现 `Qwen38SmallMPhase1` / `Qwen38SmallMPhase2` / `prepare_fc1_routes_kernel<true>`，进一步确认 native 路径真实运行，而不只是配置变更。

全量 fill 区间并集为 31.354 s，其中 9.781 s 与其他 stream 的 kernel 同时运行，占 31.20%。第一轮对应为 26.812 s、2.505 s、9.34%。别流 kernel 并发是物理时间线证据；不表示这部分依赖都可移除，也不是同等大小的端到端加速。shared-fill overlap 在第一轮已经存在，第二轮加入 cached-FC1 工作。

源码门控见 [qwen38_nvfp4_cache.py](../../../python/sglang/srt/layers/quantization/qwen38_nvfp4_cache.py#L607)，并发提交次序见 [qwen38_cache_mailbox.py](../../../python/sglang/srt/layers/moe/qwen38_cache_mailbox.py#L352)。

## Kernel 表：第二轮完整 trace

分母包含本阶段 kernel / memcpy / memset 的累计 duration：prefill 6.438 s、decode 103.836 s。并发导致 duration 相加超过墙钟时间；百分比不是可节省时间。

| 阶段 | 活动 / kernel 家族 | 累计时间 | 阶段占比 |
| --- | --- | ---: | ---: |
| Prefill | pinned H2D memcpy | 4.027 s | 62.55% |
| Prefill | CUTLASS grouped FP4 MoE | 0.695 s | 10.80% |
| Prefill | MXFP8 GEMM | 0.444 s | 6.90% |
| Prefill | sparse GQA chunk prefill | 0.243 s | 3.77% |
| Decode | mailbox_fill_kernel | 31.354 s | 30.20% |
| Decode | MXFP8 GEMM | 18.623 s | 17.93% |
| Decode | native FC1 Phase1（cached + late） | 10.351 s | 9.97% |
| Decode | cuBLAS GEMV 最大单一变体 | 7.701 s | 7.42% |
| Decode | BF16 WMMA GEMM | 5.468 s | 5.27% |
| Decode | native FC2 Phase2 | 4.180 s | 4.03% |
| Decode | HC mix up-gate + down-partial | 5.545 s | 5.34% |
| Decode | decode_plan_kernel | 2.425 s | 2.34% |
| Decode | fused GDN update | 1.312 s | 1.26% |
| Decode | prepare_fc1_routes | 0.873 s | 0.84% |

fill 仍是第一大单项。第二轮的 FC1/FC2/route-prep 原始 GPU 工作合计约 3.297 ms / VERIFY，第一轮原生 fused routed MoE 为约 3.006 ms / VERIFY：拆分/准备增加了工作，靠并发隐藏部分成本。不能把“Phase1 占比低于原先整个 fused MoE”解释为计算本身直接变快。

Prefill 记录到 pinned H2D 199.760 GB，按 copy 活动累计时间约 49.61 GB/s。继续优化应先减少重复专家传输或改进流水，而不是只增加 CTA。Decode 的普通 pinned H2D 只有 1.102 GB，但 GPU fill 从 mapped host 地址读取缺失专家，最后周期进程计数约 1.095 TB。该周期计数含启动 warmup 且未覆盖最后全部 forward，不是精确的 workload 字节增量。

## 重点尾延迟：rebalance 后的新容量冷路径

约在请求 GPU 活动开始后 30.61 秒，`step[VERIFY bs=1 forward_id=1264]` 的 CPU forward 持续 **1892.013 ms**。其中两段全设备无 GPU 活动间隙分别为 **453.393 ms**、**1390.381 ms**，合计 **1843.774 ms**。不能用稳态中位数掩盖这个长停顿。

[过渡 trace](rebalance-transition.trace.json.gz)、[裁剪统计](transition-summary.json)、[加载证据](cold-load-evidence.json) 显示：

- 两段空隙分别结束于 `cudaLibraryLoadData`，随后约 0.25 ms 首次出现相应的 `Qwen38SmallMPhase1` / `Phase2` kernel。
- 两次库加载 API 本身合计仅约 0.115 ms；该窗口明确标记的模块加载 profiler overhead 也只有亚毫秒级，不能解释 1.84 秒。
- 本次 trace 共发生 96 次 graph instantiate / begin / end；Phase1 比正常每层两次多 96 个 warmup launch，Phase2 比每层一次多 48 个，和 48 个新 stage、每个两个 graph 的构造形态吻合。整个 capture 的 graph instantiate API 累计仅约 2.8 ms，因此不能把停顿归因于该 API 自身。
- rebalance 前每层容量为 142 / 150 / 151，对应 power-of-two compile bucket 256；rebalance 后包括 80 / 96 / 112 / 128，对应新的 bucket 128。

源码中 [fc1_stage 缓存键](../../../python/sglang/srt/layers/quantization/qwen38_cached_b12x.py#L302) 包含容量和张量地址；未命中会构造新 stage。[prepared launch](../../../python/sglang/srt/layers/quantization/qwen38_cached_b12x.py#L52) 按向上取整到二次幂的 capacity 调用编译入口；[NativeFc1Stage](../../../python/sglang/srt/layers/quantization/qwen38_b12x_fc1_overlap.py#L174) 在构造时 warmup 并捕获两个 graph。

**高度疑似新 128-capacity bucket 的 Phase1 / Phase2 首次编译或编译绑定冷路径。** 证据支持这个方向，但没有 Python 栈，未声称精确证明全部停顿发生于某个 Python 函数。建议先在无 profiler 下重现 rebalance，然后验证启动时预热可达 bucket，或把新 binding / graph 的准备移出请求关键路径。仅调整 fill CTA 表不能修复这个 host 冷路径。

## Overlap 评审表

| 优先级 | 对象 | 第二轮证据 | 建议 |
| --- | --- | --- | --- |
| P0 | rebalance 后冷编译 / 新 binding | 1.892 s VERIFY，两个空隙后分别加载 Phase1 / Phase2 | 无 profiler 复现；优先评估 128/256 bucket 预热及新 stage 准备时机 |
| P1 | 剩余专家 fill | fill 占 decode 活动 30.20%，仅 31.20% 与别流 kernel 并发 | 保持 cached-FC1；进一步减少 miss / 字节，评估真实 workload 下 CTA 表 |
| P1 | Prefill DMA | 占 prefill 活动 62.55% | 减少跨 chunk 重复搬运，再验证双缓冲/预取；增大 chunk 前验证显存 |
| P2 | dense / shared GEMM 与 GEMV | MXFP8 + 最大 GEMV 变体约 25.35% | 用 shapes / mapping 明确具体 projection，避免泛化融合建议 |
| P2 | CPU 等待 / graph launch | decode cudaEventSynchronize 45.724 s、cudaGraphLaunch 9.304 s | 等待与 GPU 工作重叠，不能按累计 wait 时间估算 CPU 可节省量 |

## Fuse 评审表

| 路径 | 状态 | 本次判断 |
| --- | --- | --- |
| cached-FC1 / native Phase1+Phase2 / route prep | 已启用且 kernel 与计数均确认 | 保持用于进一步评估，需解决首次新容量冷路径 |
| shared-fill overlap / fused planner | 既有路径仍运行 | 不作为新缺失优化重复提出；planner 约 2.34% |
| HC mix up-gate / down-partial | 合计约 5.34% decode 活动 | 次于传输与冷路径，可作为后续小矩阵/融合候选 |
| fused GDN update / norm | 已有融合，单项较小 | 不是第一优先级 |
| 通用工具中的 shared-expert append / activation+quant 标签 | 只有宽泛名称匹配且 Python site unresolved | GDN sigmoid 不是 shared-expert append 的充分证据；整个 MXFP8 GEMM 时间不是 activation+quant 的可消除成本 |

原始工具三表见 [prefill-triage.md](prefill-triage.md)、[decode-triage.md](decode-triage.md)。保留工具输出以便复核，上表是源码核对后的取舍。当前为 **single-trace**，没有 graph-off mapping / graph-on formal 配对。

## 硬件、状态与后续评估约束

第二轮有效 GPU 活动范围 **98.383 s**，活动区间并集 **92.009 s（93.52%）**，全设备空隙合计 6.374 s。input 为 8192 × 3 + 984；此轮没有第一轮的两次追加请求。不要用两轮总耗时比较加速比。

KV 保持 80k 档位、专家容量 7161 slots；没有 KV 扩容，日志队列为 0。GPU1 全程空闲。活跃采样中 PCIe 均为 Gen5 ×16、SM 时钟 2.79–2.985 GHz、最高 53°C。采集下显存峰值 **32192 / 32607 MiB**，余量约 415 MiB；早期 sparse-prefill kernel 仍有首次 device-load 日志。没有 OOM，但不应直接扩大 chunk 或额外挤占空间。

下一步 CTA 评估应冻结一份实际 Hermes 请求和采样参数，用相同 serving 配置、同一输出预算、统一缓存起点做旧表 / 30-miss 候选表 ABBA 交错复测，并关闭 profiler 测吞吐。记录输出 hash / tokens、TTFT、稳态 decode、MTP 接受长度、rebalance 尾延迟；fill-only 微基准不替代实际端到端结果。本次第二轮没有切换 CTA 候选开关。

## 复核和提交边界

启动脚本语法、W4A16 / A4 分支、补丁精确复现均通过；READY 生成验证通过；完整 trace 流式解析、代表窗口和冷路径提取完成。采集及 telemetry 已停止，服务 `/ready` 返回 200，early-cache 保持开启。

本目录只提交第二轮报告、配置补丁、统计、分析脚本和三个受限 trace。第一轮只保留比较所需的小摘要；完整捕获位置和校验和见 [trace-manifest.json](trace-manifest.json)。请求正文未被这两轮 profiler 保存，不能声称当前两轮已经是完全相同请求的受控 A/B。
