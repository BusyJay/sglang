# 正在运行的真实推理：MTP A4 profile（2026-10-18）

瓶颈集中在 target verify 的专家读取和 W4A16 专家投影。当前 6390 slots、MTP=2、draft experts A4、target decode/verify A16、KV=221248。没有重启、改参数、flush cache 或发送合成请求；直接采集用户正在生成的请求。40 个 verify 轮次，CPU+GPU、record_shapes=true、with_stack=false，约 1 秒采集后自动停止。trace 导出扰动了短时吞吐，下面正常 token/s 使用采集前日志。

## 正常运行与整轮时间

02:39:25–02:40:55 的稳态日志，约 26.5K–32.5K 上下文：60.50–69.61 token/s，接受长度 1.45–1.81。按接受长度/吞吐换算，整轮约 24–26 ms。不能用此前合成 prompt 的约 90 token/s 预期这条真实负载。

trace 中 39 个完整 verify 开始到下一次 verify 开始的区间，平均 23.80 ms；GPU kernel/copy 活动并集为 23.19 ms，覆盖 97.43%，空隙约 0.61 ms/轮。这里的“GPU busy”是时间覆盖率，不是 SM 占用率或带宽饱和率。

| 范围/热点 | GPU 时间/轮 | 备注 |
|---|---:|---|
| target verify 全窗口 | 20.935 ms | 关联 GPU annotation，含异步尾部 |
| 专家主机映射读取 mailbox_fill | 6.543 ms | 48 次/轮，实际仍经 PCIe |
| target W4A16 _cache_project_splitk | 5.803 ms | 96 次/轮，两次专家投影/层 |
| 主要 MXFP8 dense projection kernel | 4.363 ms | 两类主 kernel 的累计时间 |
| HC mix | 1.319 ms | verify 窗口内 |
| decode planner | 1.280 ms | 48 次/轮 |
| draft 全窗口 | 1.137 ms | GPU interval envelope |
| draft_extend 全窗口 | 1.218 ms | GPU interval envelope |
| draft A4 专家 GEMM，两个阶段合计 | 0.084 ms | 仅 GroupProblemShape GEMM，未含路由/quant/finalize |

表中包含嵌套、并行和 PDL 重叠，不能将各行相加作为整轮时间。CPU verify annotation 仅 13.70 ms，GPU verify 为 20.93 ms；仅按 CPU return 裁 GPU 会漏算异步专家 fill。CPU cudaEventSynchronize 累计 4.68 ms/轮、process_batch_result 约 5.95 ms/轮，大量是等待/重叠，不能作为额外 Python 开销直接叠加。

## 读取与缓存

shared expert 与 fill 的并发路径仍然生效。按每层 fill 到第一段 W4A16 投影之间的 shared kernel 区间并集计算，shared 分支约 0.952 ms/轮，其中 0.773 ms 与 fill 重叠。fill 仍有约 5.770 ms/轮未被这段分支遮住，不能靠现有 shared overlap 全部隐藏。

相邻稳态计数器窗口中，cache hit rate 约 86.7%–90.1%，平均每层每轮 miss 约 2.25–2.97 个专家，总逻辑专家传输约 285–375 MiB/轮。这些来自相邻 400 轮窗口，不是精确的 profile 40 轮计数；不应用邻近流量除以本段 fill 时间来宣称精确带宽。当前链路读数 PCIe Gen5 x16。GPU trace 无专家 H2D memcpy 是因为 mapped-host kernel 直接读取主机内存，不意味着没有 PCIe 流量。

| 热层（0 起始） | fill 均值/轮 | slots | 邻近 400 轮命中率 |
|---|---:|---:|---:|
| 2 | 429.8 µs | 144 | 73.9% |
| 0 | 420.2 µs | 144 | 77.3% |
| 1 | 404.5 µs | 144 | 77.5% |
| 7 | 337.3 µs | 144 | 77.2% |
| 6 | 230.1 µs | 144 | 82.3% |
| 5 | 200.3 µs | 144 | 86.5% |

最热的 0/1/2/7 层合计约占全部 fill 时间 24.3%。32/48 层当前分配为 144 slots，包括这些热层。源码默认候选为 16 的倍数，upper=max(initial_slots)+16：初始最大 134 使候选上限落到 144。它不是 GPU 或总显存的硬上限，而是当前 shadow-policy 搜索范围的上限。

## 优先验证的方向

1. **固定总预算 6390，扩大热点层的分配搜索范围。** 先离线回放更宽的容量候选，检查是否能从冷层重分配到 0/1/2/7 等热点层，不增加总 slots、不挤占 KV。当前 32 层触到候选上限，这是值得验证的具体限制；仅凭命中率不能保证扩大后有收益。候选增多也增加 shadow CPU 分析成本，应采用有界/分阶段搜索，而不是盲目把所有层的每个候选扩到 512。
2. **继续优化实际宽度 3 的 target W4A16 split-K 投影。** 这部分每轮 5.80 ms，优先微基准评估加载/解包、跨 route 权重复用或投影融合；不能把 target 直接改为 A4 当作同精度优化。
3. **之后再比较同一真实负载的 MTP steps=1/2。** 当前接受长度只有约 1.5–1.8，每轮成本和可接受 token 数共同决定吞吐；该实验需要受控测试。单独继续压缩 draft A4 专家 GEMM 的收益上限已经很小。

日志中的 adaptive analysis=9.00s 是后台分析累计耗时；该次迁移 pause=0.226s，不能把 9 秒解读为每轮或迁移暂停。它与稳态热点分开处理。当前没有对正在运行的服务应用上述实验。

## 原始证据

- mtp-a4-live-1792262458.0863557-TP-0.trace.json.gz：完整压缩 Chrome/Kineto trace，约 9.4 MiB，可解压后在 Perfetto 查看。
- server.json：采样时服务配置。
- summary.json：profile_service.py 的 verify 摘要。
- detailed-summary.json：GPU 阶段区间、每层 fill、shared overlap、完整轮次和邻近缓存统计。
- service-window.log：采样前后服务日志；02:40:58–导出完成附近的吞吐不作为正常速度。
- capture.log / SHA256SUMS：采集命令输出及 trace 校验。

通用摘要复现：`python3 tools/qwen38_cache_tests/profile_service.py summary <trace.json.gz>`。此次是 PyTorch/Kineto trace，不是 Nsight；没有收集具体每轮 plan tensor 的 miss 值，也没有硬件计数器，不能据此断言 PCIe 带宽已经饱和。
