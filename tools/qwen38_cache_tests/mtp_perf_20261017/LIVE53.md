# 线上实测 53 token/s 的 profile（2026-10-17 16:51）

用户授权后直接采集正在运行的真实请求；没有发送合成请求，也未修改或重启服务。
配置保持主 MoE prefill A4 / decode、verify A16，MTP steps=2、3 draft tokens。
采集 CPU+GPU、record_shapes=true、with_stack=false，共 40 个 verify step，自动停止。
trace：`/home/jay/workspace/prod/homelab-llm/models/profiles/live53/`
`a16-live53-1792227087.3071592-TP-0.trace.json.gz`。

## 结论

线上 53 与日志相符，不能用此前合成短请求的 70.9 作为这条真实负载的性能预期。
采集前真实请求约 26K–30K 上下文，MTP 接受长度约 1.66–1.85，
吞吐 48.20–55.57 token/s。此前 1024-token 合成短请求接受长度约 2.53–2.60。

| 未开启 profiler 的日志窗口 | 接受长度 | token/s | 推算每轮毫秒 |
| --- | ---: | ---: | ---: |
| 2026-10-17 16:50:23 | 1.84 | 55.57 | 33.11 |
| 2026-10-17 16:50:36 | 1.85 | 55.39 | 33.40 |
| 2026-10-17 16:50:49 | 1.68 | 50.99 | 32.95 |
| 2026-10-17 16:51:03 | 1.81 | 53.83 | 33.62 |
| 2026-10-17 16:51:17 | 1.66 | 48.20 | 34.44 |

这些窗口的每轮成本平均 33.50 ms，profile 中连续 verify 启动间隔平均
33.39 ms，二者一致。粗略换算：
`1.75 接受 token / 0.0335 秒 ≈ 52.2 token/s`。
因此吞吐差异主要由真实负载的低接受长度解释；每轮没有比合成测试明显变慢。

## 热点

| GPU verify 窗口内事件 | 每轮 GPU 时间 | 40 轮调用数 |
| --- | ---: | ---: |
| target W4A16 `_cache_project` | 11.475 ms | 3840（48层×2投影×40） |
| mapped-host `mailbox_fill_kernel` | 10.985 ms | 1920（48层×40） |
| 主要 MXFP8 projection kernel | 3.738 ms | 8160 |
| fused `decode_plan_kernel` | 1.395 ms | 1920 |

verify GPU 窗口平均 31.268 ms，GPU busy 区间并集占窗口约
98.26%。GPU 工作基本连续。
所有 48 层均走 mailbox/fused planner；verify 中没有专家 H2D memcpy 或 packed
scatter，旧的逐专家 CPU staging 没有重新出现。Mailbox 仍会经过 PCIe 读取主机权重。

全段 draft `_project` GPU 总时间约 14.31 ms（约 0.358 ms/轮）。
完整 draft GPU span 约 1.185 ms/轮，draft_extend 约 1.391 ms/轮。
CPU 的 `cudaEventSynchronize` 累计约 10.364 ms/verify CPU span，
`scheduler.process_batch_result` 平均约 9.5 ms；这些等待与 GPU 工作重叠，
不能当作额外 Python 计算开销加到 GPU 时间上。

profile 的 CPU verify span 仅 20.229 ms，GPU 尚在异步执行；汇总使用关联的
GPU annotation 保留尾部工作。不同 kernel/span 的时间存在嵌套和并行，不能直接相加。
采集及 trace 导出本身扰动性能；16:51:35 打印的 33.54 token/s 跨过 profile 和
导出窗口，因此没有把它作为正常吞吐。

## 后续优化依据

保持 A16 和 MTP 开启。进一步提速应优先评估：
1. 在同一真实负载下比较 steps=1 和 steps=2 的接受长度/轮成本，避免低接受率时
   为第三个验证 token 支付额外 A16 计算和专家并集搬运。
2. 优化多 token A16 专家计算与 mapped-host fill；它们是当前实际热点。
   单独削减 draft 网络的收益上限很小。

本次只做了线上 profile 与分析，没有改动运行参数。
