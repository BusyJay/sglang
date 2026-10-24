# 最终 Nsight trace 分析

结论：持续 decode 主要耗时仍在**权重填充和 LFU planner**，而不是 CPU 结果合并或 routed MoE 矩阵计算。Prefill 则以大批量 H2D 与填充开销为主。另外发现频繁统计刷新会强制 GPU/CPU 同步，值得优先做小范围关闭日志对照；本轮只分析，不改运行时。

对象：`default.nsys-rep`（SHA-256 见 `capture.json`），SGLang / Qwen3.8-Flash-Next-FP4-Local，RTX 5090 D，CPU 2,3,4,5 共算开启，校准与配置见上级目录。源码基于 `3803f03a70`，采集提交 `6b1b485395`。没有额外临时 arena。

## 时间窗口和口径

全段 GPU 活动跨度 9.269 秒，kernel 或 memcpy 覆盖 84.9%；kernel 时间并集 5.471 秒、memcpy 时间并集 2.399 秒。kernel 累计时间为 6.082 秒，含不同 stream 的重叠，不能与并集混用。

持续 decode 切为 trace 时间 **5.871282051–9.301972306 秒**：取最后一次 `packed_scatter` 后首个 hybrid planner 作为起点，排除 1088-token 请求自己的 prefill。窗口 3.431 秒，共 5232 次 target MoE planner / fill / merge（109×48 层）。这是 kernel 标记推定的阶段边界，不是 HTTP 请求 NVTX 边界。长 prefill 近似窗口 0.033445232–4.619132390 秒；短请求 prefill 近似窗口 4.777286578–5.871282051 秒，后者起点仍包含前一请求少量尾部工作。

## Kernel 表：持续 decode

占比使用该窗口所有 kernel 累计时间 3.316 秒为分母。不同 stream 可重叠，以下不可直接解释为端到端加速空间。

| Kernel / 家族 | 次数 | 累计 ms | kernel 占比 | 平均 μs |
| --- | ---: | ---: | ---: | ---: |
| mailbox_fill | 5232 | 1000.42 | 30.2% | 191.21 |
| hybrid_plan indexed | 5232 | 719.38 | 21.7% | 137.50 |
| Dense MXFP8 GEMM | 26157 | 415.15 | 12.5% | 15.87 |
| B12X FP32 routed MoE | 5232 | 322.61 | 9.7% | 61.66 |
| GEMV | 865 | 183.98 | 5.5% | 212.69 |
| hc_mix | 11222 | 150.31 | 4.5% | 13.39 |
| BF16 WMMA GEMM | 6538 | 127.61 | 3.8% | 19.52 |
| hybrid_merge | 5232 | 95.29 | 2.9% | 18.21 |

fill + planner 累计 1.720 秒，约为持续 decode 窗口的 **50.1%**；两者与其他 kernel 的时间交集均为 0。相比之下 routed MoE 计算只有 0.323 秒。GPU 活动覆盖 91.9%，空隙共约 277ms；说明“再增加 CPU 计算量”本身不能直接消除主要等待，必须减少换入或调度成本。

`mailbox_fill` 源码直接执行 `ld.global.cv.v4.u32` 从注册的主机映射读权重，写入显存缓存。因此这部分 PCIe 传输记在 **kernel** 上，不在 CUPTI memcpy 表里。持续 decode 的 H2D memcpy 仅 1728 字节，不能据此推断没有 H2D 权重流量，也不能仅靠这份 trace 算出 mapped-load 的实际 PCIe 带宽。

Prefill 方面：长 prefill 近似窗口 H2D 86.72GB / 86658 次，设备 memcpy 累计 1.752 秒，kernel 并集 1.851 秒，二者交集仅 3.22ms；另有 mailbox_fill 540ms。短请求 prefill 近似窗口 H2D 31.09GB / 36485 次、累计 631ms，另有 fill 218ms。可见大量专家搬运仍在；本 trace 只有 16k 长输入，**不能单独证明或量化 170k 回退原因**。

## 重叠机会表

单 trace 判断，未采额外 mapping/formal 对照；重叠表示实际时间交集，不代表资源利用率。

| 路径 | 证据 | 判断与下一步 |
| --- | --- | --- |
| Prefill DMA 与计算 | 长窗口仅 3.22ms 交集，H2D 1.752s | 传输与计算基本串行。先核查现有 staging / async_pack_prefetch 的依赖，确定是否存在下一组可预取数据；不能直接假定改 stream 即可重叠。 |
| Decode fill 与 shared 分支 | fill 1.000s，与其他 kernel 交集 0；`DeviceHybridPlanner.run` 先 fill 再调用 shared closure | 属于已有 shared/routed 双分支重叠家族。可评估把独立 shared 工作放到 alt stream，但要验证共享资源、图与缓存生命周期；同层 routed MoE 必须等 fill。 |
| CPU 原生计算与 GPU | 未开启函数时长 NVTX，仅周期采样 | 无法可靠给出 CPU 原生计算的重叠百分比。merge 总计 95ms，包含等待/读取/相加/转换，也包含 GPU-only 路由，不能全算 CPU 等待。 |
| 统计刷新同步 | 41 次 1536B DtoH，host API 1537.63ms，设备拷贝 0.030ms | 小拷贝触发等待前面 GPU 队列。优先对照 stats_log_interval=0；不能把 1.538s 全当作可回收端到端时间。 |

统计调用的尺寸与源码吻合：`miss_mailbox.flush_stats()` 回读 `(48,4)` int64 hit_stats（1536B），`stats_snapshot()` 接着调用 `planner.sync_host()` 回读 lookup、frequency、owners、keys 等。后几项对应 trace 中各 41 次的 98304B、196608B、28024B 拷贝。日志周期为 128 层调用，109×48 / 128 约为 41 次。此路径会等待事件、导出完整 Python 状态。即便不会额外计算 1.538s，也会周期性切断异步提交；持续 decode 最长几个空隙约 2.8–3.6ms，但未把每个空隙逐一因果归给日志。

## 融合模式表

已对照技能中的 fusion/overlap catalog，再以本分支源码为准。

| 模式 | 本地状态与 trace | 结论 |
| --- | --- | --- |
| CPU 结果等待 + 映射读取 + FP32 加法 + BF16 转换 | `hybrid_merge_kernel` 已融合，95ms / 2.9% | 已启用；当前优先级低。回到每个 FC2 writer 轮询的旧 epilogue 缺少收益证据。 |
| LFU planner → fill → routed MoE | 同 stream 且存在真实数据依赖；planner 719ms、fill 1000ms | 不是通用 elementwise fusion 缺失。应先减少 LFU 热路径工作、换入次数；将 planner/fill 合成一个 kernel 的收益不能直接等于二者耗时之和。 |
| shared / routed 双分支 | 本地 shared closure 当前在 fill 后提交；catalog 有既有 alt_stream/SBO 模式 | 可检查已有分支并行设计的适配，属于调度重叠机会。不要为此新增 arena。 |

## 建议顺序与限制

1. 小范围对照关闭周期统计：验证同步边界和 Python 状态导出的成本，避免先改复杂 kernel。
2. 减少无效权重换入与缓存抖动；同时检查 planner 的串行 heap/候选处理。维持既有 arena，不引入额外 scratch。
3. 在依赖允许的范围内评估现有 prefill staging、shared 分支的重叠；先验证正确性再看性能。

此 trace 开启 profiler、CUDA graph node 跟踪和 Python sampling，不能用请求时间替代无 profiler 的性能验收。没有 SM 利用率、PCIe 硬件计数器，也没有 CPU native duration 标注，因此不能判断 fill 是链路饱和、访存延迟还是 CTA 配置达到最优，亦不能据此重选 CPU 1/2/3 专家拐点。上面的建议是下一轮验证顺序，并非已完成修复。

复现分析：使用采集所用镜像的 `nsys export --type sqlite` 将报告导出为 `analysis.sqlite`，然后运行 `python3 tools/qwen38_cache_tests/migration_20261021/analyze_nsight.py tools/qwen38_cache_tests/migration_20261021/nsight-final/analysis.sqlite --output tools/qwen38_cache_tests/migration_20261021/nsight-final/analysis.json`。SQLite 是本地派生大文件，不提交。
