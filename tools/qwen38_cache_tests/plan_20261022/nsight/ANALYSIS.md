# 包含 LFU 的完整 plan，最终 Nsight

对照 `overhead_20261022/nsight-final/default.nsys-rep`（检查点 0d9dcf3a2c），本目录为新代码和重新测量的成本表。双方无周期全状态统计，均使用 MTP 2，预热后捕获 16k prefill 和 1k/256 请求。

| 持续 decode 指标 | 检查点 | 本轮 |
|---|---:|---:|
| 调用数 | 5184（108×48） | 5136（107×48） |
| 完整 plan 平均，µs | 48.99 | 35.74 |
| 完整 plan 中位数，µs | 44.38 | 31.17 |
| 完整 plan P95，µs | 85.31 | 70.88 |
| fill 平均，µs | 207.18 | 203.83 |
| merge 平均，µs | 30.32 | 61.92 |
| merge 中位数，µs | 1.44 | 1.44 |
| merge P95，µs | 247.23 | 356.29 |
| 整个持续 decode 窗口，ms | 2842.02 | 2891.29 |

完整 plan 包括热度更新、LFU 准入/淘汰、冷回收、成本表匹配和 CPU 请求 staging，平均降低 27.1%。空成本表 kernel 微基准的收益更大，不拿其替代这项真实服务测量。

CPU 等待尾部在这轮 trace 中变长，窗口整体没有变快。双方各有 10 个重新测量通过的 CPU workload key，其中只有 6 个相同；MTP 接受率/路由也不同。表内 kernel 总和或窗口差不能用来给源代码单独归因，端到端看上层 REPORT.md 中未开启 profiler 的服务对照。

本轮没有改变 cache 容量、跨层 LFU 语义和权重传输方式。尚存的 plan 开销包括串行跨层 victim 堆维护、准入/强制安置，以及 CPU profile 匹配；CPU 唤醒和结果尾延迟是 plan 之外的独立问题。

分析原始数据见 `analysis.json`，精简对照见上层 `nsight-comparison.json`；原始 trace 为 `default.nsys-rep`。SQLite 为本地导出中间文件，不纳入提交。
