# 最终 Nsight 分析

对照用户给定的 `migration_20261021/nsight-final/default.nsys-rep`，最终报告为本目录 `default.nsys-rep`。使用容器内 Nsight Systems 2026.4 导出 SQLite，以 `migration_20261021/analyze_nsight.py` 分析，源码和校准 SHA256 见上层 `source-manifest.json`。

| 持续 decode 指标 | 原 trace | 最终 trace |
|---|---:|---:|
| planner 调用数 | 5232（109×48） | 5184（108×48） |
| planner 平均，µs | 137.50 | 48.99 |
| fill 平均，µs | 191.21 | 207.18 |
| fill 与其他 kernel 重叠，ms | 0 | 99.92 |
| merge 中位数，µs | 5.28 | 1.44 |
| merge 平均，µs | 18.21 | 30.32 |
| merge P95，µs | 5.95 | 247.23 |
| GPU busy | 91.93% | 96.85% |
| GPU idle，ms | 277.02 | 89.59 |

planner 平均耗时降低 64.4%；fill 并没有变快，但与 shared expert 的计算发生约 100 ms 重叠。merge 中位数下降，均值/P95 上升，说明当前 CPU 结果等待尾部仍需优化，不能把转换 kernel 加速等同于 CPU 共算的整体改善。两次 trace 的 CPU 校准表、周期统计、MTP 接受率与路由不完全一致，kernel 时间不直接等价为端到端加速，也不能直接相减累加。

16k prefill 代理窗口从 4585.69 ms 降到 3868.23 ms，GPU idle 从 978.49 ms 降到 627.71 ms。这个窗口以首个 decode planner 为界，包含请求衔接，不等于完整 170k 请求延迟。

prefill DtoH 从 37.18 MB 降到 5.89 MB；主路径 sparse HtoD 仍约 86.86 GB / 86816 次，DMA 时间约 1.754 s。后台缓存准入的 host 权重读取已换成 GPU D2D，前台 arena 填充仍是主要传输瓶颈。

原 trace decode 含 41 次周期性全状态导出，最终关闭周期导出，仅保留每步输出长度/结果等必要小拷贝；统计关闭的独立端到端对照收益约 2%，不能把原 trace 的主机 API 等待时间直接当成可节省总时长。

后续优先级：CPU 工作提交/唤醒和结果就绪尾延迟；mapped fill 有效 PCIe 吞吐与 shared 分支重叠收益；前台 prefill 的稀疏 H2D 调用与 CPU 分派间隙。无堆并行扫描目前未优于索引堆。
