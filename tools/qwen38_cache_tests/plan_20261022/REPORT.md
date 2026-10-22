# 包含 LFU 的 plan 优化，2026-10-22

先前优化、成本表和最终 trace 已提交：`0d9dcf3a2c`。本轮在该检查点上继续优化完整 plan，包括热度更新、LFU 准入/淘汰、冷回收、CPU 分流、路由输出。

## 改动和正确性

- warp ballot 并行构造活跃/冷专家 bitmap；用按 expert ID 排列的前缀压缩替代单线程路由插入排序及逐位冷集合构建。
- 热度统一以原 FP64 的独立乘法和加法更新；当前层堆从底层到根按层并行 heapify，替代逐个命中专家的串行堆修复。同层节点拥有互不相交的子树，各层之间有 CTA barrier。最低热度、layer/expert 并列规则保持不变。
- 增加 128 字节 CTA shared scratch，没有新增持久 GPU 缓冲或改变缓存预算。跨层 victim 选择与准入仍使用原索引堆。
- host replay 显式串行重建 warp ballot 的相同谓词，仍只用于策略与传输敏感性分析，不预测 CUDA 时间。

GPU suite 12/12 通过，扩展为 146/511 个驻留专家、M=1/2/3、top-k=10 的重复替换，以及 512 专家的独立 Python LFU 对照，显式覆盖 0/31/32/63/64/127/255/256/510/511 的位图边界。固定路由重新回放的全部计数与检查点逐项一致。原始证据 `gpu-expanded.log`、`route-replay.json`。

## 包含 LFU 的 kernel 微基准

5090 D，M=3/top-k=10（30 条路由、12 个不同专家），同样的路由构型与 reset，CUDA events 中位数，reset 排除在计时之外。旧单层 146 槽位、FP32 heat；新全局 48 层、512 专家/层、7006 槽位、FP64 heat。旧路径使用真实 mailbox 调用的 `decode_plan_parallel` / parallel_miss=2。全局微基准成本表为空；完整服务与新 trace 则加载本机重新测得的成本表。

| 场景，µs ↓ | 共算前单层并行 LFU | 上个检查点的完整全局 plan | 本轮完整全局 plan |
|---|---:|---:|---:|
| 全命中 | 12.45 | 34.27 | 16.14 |
| 满缓存 miss | 32.67 | 52.96 | 44.59 |
| 空闲槽位 miss | 32.32 | 30.72 | 22.18 |

同一批路由下，旧 LFU 确实更轻；全局 plan 还需维护跨层索引与 CPU 分流。旧/新缓存语义和 heat 精度不同，不把这个微基准当作等价策略对照或端到端吞吐证明。完整模型结果另列。

基准复现：`PYTHONPATH=python python test/manual/kernels/moe/bench_qwen38_hybrid_planner.py --tokens 3 --top-k 10 --slots 7006`，旧实现：`PYTHONPATH=python python tools/qwen38_cache_tests/plan_20261022/bench_legacy_lfu.py`。实际执行使用本机已有开发容器及 GPU 0；日志保存在本目录。

252 个重新生成的配对成本测量 case，5% margin，最终接受 10 个 workload key；`calibration.json` 绑定新 runtime fingerprint。旧成本表在源码变动后不适用。

## 完整服务对照

保持相同 GPU/CPU、6388 名义槽位、B12X、MTP 2、8192 chunk 和 salted prompts。双方重启服务，关闭周期全状态统计，每组三个短请求取中位数。

| 指标 | 上个检查点共算 | 本轮共算 | 本轮纯 GPU |
|---|---:|---:|---:|
| 1088 输入 / 512 输出 decode，tok/s ↑ | 100.72 | 103.51 | 100.08 |
| 跳过前 64 token，tok/s ↑ | 105.91 | 110.50 | 104.48 |
| 自然聊天 decode，tok/s ↑ | 96.10 | 98.24 | 101.71 |
| 170064 输入 / 8 输出 TTFT，s ↓ | 30.33 | 30.49 | 29.94 |
| 长输入后短请求 decode，tok/s ↑ | 94.45 | 96.39 | 96.60 |

本轮短 decode 较检查点提升 2.8%，在这次纯 GPU 对照中高 3.4%；自然聊天仍低 3.4%，长 TTFT 高 1.8%。旧纯 GPU 的多轮短 decode 中位数在约 100–106 tok/s 间波动，不能把本轮单次超过当成稳定全面超过。MTP 接受率与输出轨迹不同也会影响端到端结果。8-token 长请求的 decode 不作为持续吞吐指标。

本轮只改 LFU/plan kernel 和对应回放工具、测试；长 prefill 不预期有系统性收益。完整记录 `performance/performance-summary.json`，精简值 `comparison.json`。

复现：`bash tools/qwen38_cache_tests/plan_20261022/run_performance.sh --modes default legacy_gpu`；trace：`bash tools/qwen38_cache_tests/plan_20261022/run_profile.sh`。脚本使用 GPU 0 和 CPU 核 2–5，默认写新的 replay 目录。

## 最终真实服务 trace

`nsight/default.nsys-rep`、`nsight/ANALYSIS.md`：包含 LFU、成本表匹配与 CPU staging 的完整 plan，平均 **48.99 → 35.74 µs（-27.1%）**，中位数 **44.38 → 31.17 µs**，P95 **85.31 → 70.88 µs**。

这轮 trace 的 merge/CPU 等待均值从 30.32 增至 61.92 µs，持续 decode 窗口从 2842.02 增至 2891.29 ms；不能声称 profiler 中整段 decode 加速。新旧成本表各接受 10 个 key，仅 6 个相同，MTP 路由也有变化。上面的无 profiler 完整服务结果和 kernel 微基准分别提供各自范围的证据。

后续 plan 重点是串行跨层 victim 堆维护、准入/强制安置和 CPU profile 匹配；CPU 结果尾延迟需独立处理。
