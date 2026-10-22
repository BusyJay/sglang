# Qwen3.8 共算开销优化，2026-10-22

本轮已明显缩小差距，尚未稳定超过开启共算前的纯 GPU 水平。代码未提交。

## 完整服务对照

同一 RTX 5090 D、Ryzen 9950X，CPU 核 2–5，6388 名义专家槽位（初始借用后 7006），B12X、MTP 2、8192 prefill chunk。每个服务重新启动，固定 salted prompts，短请求三次取中位数。最终共算服务加载 10 个本机配对测量通过的 CPU workload key，未校准 workload 回退 GPU。最终两组均关闭周期全状态统计。

| 指标 | 优化前共算 | 最终共算 | 同轮纯 GPU |
|---|---:|---:|---:|
| 1088 输入 / 512 输出 decode，tok/s ↑ | 82.36 | 100.72 | 105.18 |
| 以上请求跳过前 64 token，tok/s ↑ | 86.88 | 105.91 | 111.89 |
| 自然聊天 / 512 输出 decode，tok/s ↑ | 78.12 | 96.10 | 106.67 |
| 170064 输入 / 8 输出 TTFT，s ↓ | 40.20 | 30.33 | 29.87 |
| 长输入后短请求 decode，tok/s ↑ | 75.40 | 94.45 | 96.80 |

短请求 decode 较原共算提升 22.3%，长输入 TTFT 降低 24.5%；相对同轮纯 GPU，短 decode 仍低 4.2%、自然聊天低 9.9%、长 TTFT 高 1.5%。原始共算关闭统计的单独对照短 decode 为 84.03 tok/s，说明收益主要来自实现改动。MTP 接受率和输出轨迹会波动；这些是单轮服务测量，不是统计显著性证明。8-token 长请求的 decode 不作为持续吞吐依据。

原始记录：`before/performance-summary.json`、`no-stats/performance-summary.json`、`performance-final/performance-summary.json`，精简比较见 `comparison.json`。

## 保留的实现

- LFU 索引：当前层和顶层堆缓存进 shared memory；只处理最多 32 条实际路由对应的有序专家集合。冷回收先并行构造 bitmap，再按原顺序回收。
- 命中热度按专家逐个更新和修复堆；只有当前最小 victim 碰到活跃读者时才临时移除，保持原 LFU 决策、并列顺序及活跃保护。测试跨 64 层缓存边界与 32 行成本表边界。
- prefill：GPU 统计路由直方图，只回读 513 个 int32，替代每层将最高 8192×10 个 ID 转成 Python 整数。保留 int64/CPU 输入验证路径。
- prefill 后台准入直接从借用 GPU arena 拷入缓存，避免再读 host 权重；源 arena 复用前等待准入完成。
- fill 与 shared expert 分支重叠；复用 M=1/2/3 路由元数据和 fill events，合并 GPU/merge 完成事件。新增常驻路由元数据 1008 字节，没有增加专家权重 arena 或缓存预算。
- merge 并行检查 CPU 路由，多 CTA 转换和合并；保留 FP32 累加到最终唯一一次 BF16 转换。
- prefill 阶段暂停无效 mapped mailbox 的空轮询。

## 无堆多线程实验

已实现 GPU 多线程全局候选扫描/归并，并做 shared-memory 缓存、warp reduction、整数排序键、寄存器缓存等迭代。最后版本用 256 线程、寄存器保存候选、8 个 warp 有序流和虚拟状态选择。完整实验补丁为 `experiment-parallel-vs-heap.patch`，未接入默认路径。

48 层 × 512 专家，7006 槽位，M=3、top-k=10 的相同状态微基准（CUDA event，微秒）：

| 场景 | 优化后的索引堆 | 无堆并行最终实验 |
|---|---:|---:|
| 全命中 | 34.27 | 34.16 |
| 满缓存 miss | 52.96 | 118.05 |
| 空闲槽位 miss | 30.72 | 110.05 |

无堆满缓存 miss 从首版约 633 微秒降到了 118 微秒，但仍慢于堆；全命中近似持平，不能据此切换默认算法。原始版本扫描开销更大，`planner-parallel-*-m3k10.log` 保存逐轮结果。实验和生产使用相同 LFU 语义，并非通过放宽策略换速度。

另试过 CUDA 13 batch DMA，稀疏字节与 stream 测试通过，但完整模型长 prefill 未改善，已从生产代码移除。

## 验证与复现

- 生产 GPU kernel suite：12 tests passed，包含重复 indexed/scan 状态一致性、冷回收与活跃保护、路由直方图、非连续准入拷贝、merge 边界和 mapped completion。
- CPU policy/dispatch/heap suites：84 tests passed；namespace suites：70 passed。ruff fatal checks 和 git diff whitespace check 通过。
- 重新回放固定路由并生成 252 个配对成本测量 case，按 5% margin 接受 10 个 workload key；未命中 key 走 GPU。M=1/2 fixtures 是原 M=3 路由的截断分支测量，不是独立模型轨迹。
- 最终校准：`calibration-final.json`；代码 SHA256 与 runtime fingerprint：`source-manifest.json`。修改版本绑定的源码后需重新测量，不应复用旧成本表。
- 完整服务复现：`bash tools/qwen38_cache_tests/overhead_20261022/run_performance.sh --modes default legacy_gpu`，默认写入 `performance-replay/`。运行会占用 GPU 0 与 CPU 核 2–5。

生命周期回放 `lifecycle-final/validation-summary.json` 通过：1024 → 170000 → 1024 → 170000 → 1024，8 次 KV/专家池切换，125 个状态快照，实际 1439 次 CPU 共算 / 3648 个 CPU 专家，探针返回 42。累计 614.66 GB prefill 后台准入使用 GPU D2D 拷贝，未再读取对应 host 权重。该累计值覆盖整个回放，不是单次请求传输量。

最终生产源码重新执行 GPU suite：12/12 通过（`gpu-tests-final-source.log`）。无堆最终实验从补丁重建到隔离目录，GPU suite 12/12 通过（`gpu-tests-parallel-isolated.log`），包括重复状态与 scan 参照逐项比较。

最终 Nsight：`nsight-final/default.nsys-rep`、`nsight-final/ANALYSIS.md`、`nsight-comparison.json`。planner 平均 137.50 → 48.99 µs；GPU busy 91.93% → 96.85%；fill 与其他 kernel 的重叠从 0 → 99.92 ms。merge 的中位数下降，但 CPU 结果等待尾延迟仍在；前台 prefill sparse H2D 仍约 86.86 GB。这些是当前尚未全面超过纯 GPU 的后续优化重点。
