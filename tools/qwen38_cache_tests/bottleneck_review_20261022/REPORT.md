# 最新共算瓶颈复核（2026-10-22）

分析对象是提交 `9321a3f4504534811676c4d1f39d62341089e610`。工作区在分析前干净；没有修改生产代码。本机无 GPU、Nsight CLI，未重新采集或独立重新导出 Nsight：使用已提交 `plan_20261022/nsight/analysis.json`，复核统计脚本、源码、校准原始样本和无 profiler 服务结果。`source-manifest.json` 四个源码哈希及原始 `default.nsys-rep` SHA256 全部匹配。trace 为 Nsight 格式，未将其冒充 torch-profiler mapping/formal pair。

配置：SGLang，Qwen3.8-Flash-Next-FP4-Local，RTX 5090 D / Ryzen 9950X，CPU workers 核 2–5，B12X，MTP 2 / 最多 3 个验证 token，48 层，top-k 10，6388 名义槽位，8192 prefill chunk，单请求。性能测试长输入为 170064 token；trace 的长输入约 16k，不能混用。

## 判断

最大绝对瓶颈仍是 GPU 从映射主机权重填充专家缓存；当前最值得首先处理的共算特有问题是 CPU 结果尾延迟，以及过粗成本键丢弃了可盈利分流。全局 LFU 的固定开销仍在，且切换 legacy_gpu 同时切换缓存组织，已有端到端对照不能单独归因于 CPU 共算。

## Kernel 表：持续 decode

窗口 4.915964427–7.807250699 s，共 2891.286 ms；5136 次层调用，即 107 个 48 层 sweep。每 sweep 不等于每个输出 token。kernel 累加 3081.103 ms，跨流重叠使其超过墙钟。

| Kernel 家族 | 累计 ms | kernel 累计占比 | 平均 µs | 每 sweep ms |
|---|---:|---:|---:|---:|
| mailbox_fill_kernel | 1046.893 | 33.98% | 203.83 | 9.784 |
| Dense MXFP8 GEMM | 442.365 | 14.36% | 17.23 | 4.134 |
| hybrid_merge_kernel | 318.026 | 10.32% | 61.92 | 2.972 |
| B12X FP32 routed MoE | 314.060 | 10.19% | 61.15 | 2.935 |
| hybrid_plan_kernel | 183.551 | 5.96% | 35.74 | 1.715 |
| GEMV | 180.606 | 5.86% | 212.73 | 1.688 |
| _hc_mix_persistent_kernel | 147.312 | 4.78% | 13.37 | 1.377 |
| BF16 WMMA GEMM | 125.166 | 4.06% | 19.50 | 1.170 |
| kernel_cutlass_kernel_flashinf_3... | 32.801 | 1.06% | 2.13 | 0.307 |

来源：`../plan_20261022/nsight/analysis.json`，仅列累计 GPU kernel 时间占比 >=1% 的家族。其 family 分类见 `../migration_20261021/analyze_nsight.py`；截断名字不能用于猜测具体融合算子。

## Overlap 表

| 链路 | 实际证据 | 优先方向与限制 |
|---|---|---|
| fill 与 shared expert | 1046.893 ms fill 只有 93.277 ms 与其他 kernel 重叠，约 8.91%；953.616 ms 未与其他 kernel 重叠 | 已启用独立流；当前所有 routed GPU 专家仍需等待 fill 完成。考虑驻留 hit / miss 分组流水时必须测量第二次 MoE launch、mask 和 FP32 合并的代价，不能直接宣称额外重叠即收益 |
| CPU 分支与 GPU 分支 | merge 总计 318.026 ms，p50 1.44 µs、p95 356.285 µs，未与其他 GPU kernel 重叠 | 结果同步已暴露在关键路径上。CPU 计算可能已被部分覆盖；缺少 native CPU duration，不能量出覆盖比例，也不能把全部 merge 时间都记为 CPU 算术 |
| CPU 调度与结果发布 | poll 结束重新获取 GIL；Python 调用 CPU native，返回后 Python 更新统计并调用 complete | 优先补 head-observed、native-start/end、completion-store 时间戳；检验将整条交接移入原生 worker 的收益。GIL 和调度是代码支持的假设，当前 trace 没有证明其各自耗时 |
| Prefill copy 与计算 | long_prefill_proxy：H2D 86.727 GB、累计 DMA 1755.049 ms，copy/kernel overlap 0.937 ms | 单独优化 prefill 流水与搬运量；decode 共算改动不会自动降低 170k TTFT。该窗口是 proxy，不能将其全部解释成精确 prefill GPU critical path |

`mailbox_fill_kernel` 直接执行 `ld.global.cv.v4.u32` 读取已注册 host 权重。decode MEMCPY 表只有少量 H2D 并不代表没有权重跨 PCIe 传输。96.93% gpu_busy 是 kernel/copy 活动区间并集，包含 merge 自旋等待，不是有效算术占用率。

## Fusion 表

| 精确源码模式 | 当前状态 | 判断 |
|---|---|---|
| plan + CPU request staging | `hybrid_plan_stage_indexed` 已使用 | 已融合，不能重复列为待做收益 |
| wait + mapped result + FP32 add + BF16 cast | `hybrid_merge_kernel` 已使用 | 常规开销很小，长尾主要应从等待链路定位；不能把 318 ms 当成可通过简单 fusion 全部消除的开销 |
| B12X FC2 + CPU merge epilogue | 代码保留，默认服务使用 FP32 B12X 后独立 merge | 历史逐输出轮询曾造成 3.7–5.5 ms 交接，切回必须重新设计等待/读取与实测；见 migration_20261021/REPORT.md |

同时检查了 profiler skill 的 fuse/overlap catalog。shared/routed branch overlap 已是现有模式；没有仅凭 kernel 名提出新的融合匹配。

## 最重要的新发现：成本键的分辨率限制收益

`hybrid_plan_kernel` 的匹配包含 token 数、GPU/CPU route 数、CPU expert 数、CPU/rejected route histogram，未包含 admitted expert 数、剩余真实 GPU fill 字节和层标识。GPU route 数不能区分已经驻留与刚准入需搬运。

校准并非没有考虑这个问题：`bench_qwen38_hybrid_gate.py:523` 明确要求同键所有已测代表都获胜，选择最差配对比例的原始时间。因此它对已测分布偏保守，但不保证未测场景成立；`prepare_cases.py` 每种 key/准入数/剩余 rejected 数只选一个最常见代表，未覆盖所有层与专家。

重新统计原始文件：

| 样本组 | 测量 case | 单独通过收益门槛 | 通过但被同键整体拒绝 | 最终成本 key |
|---|---:|---:|---:|---:|
| M=3 | 180 | 135 | 78 | 7 |
| M=1/2 | 72 | 58 | 41 | 3 |
| 合计 | 252 | 193 | 119 | 10 |

这是代表样本数，不是线上调用占比或可兑现加速比例。具体例子，同键 `[3,29,1,1,[1],[1]]`：层 15、admitted=0，CPU/GPU 为 275.627/170.778 µs；层 32、admitted=2，239.098/279.647 µs。前者导致整个 key 被拒绝，后者约 14.5% 的局部分支收益也无法使用。两例层和专家不同，不能把差异唯一归因为 fill 数，但已足以证明现有 key 混合了盈亏不同的工作负载。

建议先将 admitted/GPU-fill count 或 bytes 纳入成本分桶，保留收益与稳定性门槛，不直接降低 margin。候选选择使用相同 workload 的分支时间。还需覆盖层、CPU cache 冷热、连续服务 GIL 竞争及 P95。表变大后要改为稀疏索引/桶查找，避免直接扩大当前单线程 `k × samples × histogram` 扫描；超过 32 行时还会失去当前整表 shared-memory 缓存路径。

校准已计入 full plan、fill、CPU worker、GPU、merge，不是裸 CPU GEMM 微基准；不过 invoke 每次新建 buffers/stats 并同步，GPU 工作不含服务 shared-expert 并发，CPU worker 不经历真实服务的 Python 排队负载。单层反复 reset 的微基准还不能代表连续 48 层和 CPU 权重 cache 冷热。

## CPU 长尾与 plan 收益

按相同 48 层 sweep 归一化，上个检查点到本轮：plan 2.352→1.715 ms，节省 0.636 ms；merge 1.455→2.972 ms，增加 1.517 ms；fill 9.945→9.784 ms。窗口每 sweep 26.315→27.021 ms。算术上，merge 增量超过 plan 节省；两个 trace 的成本 key 与路由不同，不能据此断言 plan patch 导致 CPU 变慢。

CPU 路径的可疑点是 `qwen38_mailbox.cc` poll 返回时 reacquire GIL，`qwen38_hybrid_device.py:297` 在 Python 中调用 native 后再发布完成，及 `cpu_nvfp4/kernel.cpp` 对每个 active expert 串行发一次四线程 condition-variable dispatch。四核并行发生在单专家内部，不是三个专家同时各自并行。建议先定位这些间隔，再试持久原生任务循环和批量 dispatch；不先盲目加 CPU 核数。

merge 中位数 1.44 µs × 5136 次只有 7.396 ms，实际总计 318.026 ms，差额约 310.630 ms。这只是重尾量级参照，不是精确 CPU wait 计数。假设 merge 时间减半且其他阶段不变，该 trace 理想速度提升约 5.82%；这不是实测承诺。

全局 plan 尚有成本：最新同构型微基准 all-hit 16.14 vs 旧逐层 12.45 µs；full miss 44.59 vs 32.67 µs。微基准的缓存语义、热度精度不同，不能作为等价策略速度证明。重点是 hit 快路径、跨层 victim 更新和精确成本查找；不应拿空成本表测量替代真实服务 35.74 µs。

## 为什么还不能稳定超过 legacy_gpu

近似分支关系是：

`T_mixed = P_global + max(T_CPU_ready, max(T_remaining_fill, T_shared) + T_GPU_routed) + T_merge_base`

`T_legacy = P_local + max(T_all_fill, T_shared) + T_GPU_all + T_output`

忽略的 enqueue 间隙、资源争用和未来缓存状态需在实验中补齐。CPU 需要省下关键路径上的权重搬运，并弥补全局 plan、交接、输出开销，增加执行设备本身并不保证变快。当前 CPU 只候选 LFU rejected：已准入专家总会搬到 GPU；单次 branch 最优也不一定等于未来缓存轨迹最优。

无 profiler 最新结果：短 decode 103.51 vs 100.08 tok/s；自然聊天 98.24 vs 101.71；长输入后短 decode 96.39 vs 96.60；170k TTFT 30.49 vs 29.94 s。短测基线三次本身跨度 98.04–108.39 tok/s，无法声明稳定全面胜出。自然聊天 accept length 1.9542 vs 1.9617 接近，以 accept_length / tok/s 粗略归一化得每验证步 19.89 vs 19.29 ms，约 0.60 ms 差距；这是近似推导，不是直接 step trace 测量。

全局 LFU 不保证较更好的逐层分配省搬运：历史固定路由对动态逐层已测最优配置包络的回放，7006 槽全局纯 GPU H2D +1.95%。这是带事后按请求择优、立即迁移假设的回放包络，不是当前默认基线实测，也不是最新 trace 的传输差值。

## 下一次实验顺序

1. 固定相同全局实现，比较当前成本表与空 samples 成本表，CPU hybrid 开关保持开启；另保留 legacy_gpu。空表还绕过 mapped merge，因此如需进一步拆解固定开销，加一个保留管线但拒绝 CPU 的诊断变体。三组同输入、交错运行、分别重启且统一预热，记录 MTP accept、ms/sweep、cache fills/bytes。
2. 添加原生交接时间戳，不用每层 Python 打印。拆分 publish→observe、observe→native-start、native compute、native-end→complete、GPU merge start→complete，按实际 CPU key 和 fill 数分桶。
3. 用更细分 key 重做候选 gate，回收现有 119 个被粗粒度拒绝的盈利代表所对应场景；不能用样本数推断收益。新增时间桶控制 P95 回退。
4. 再优化 resident-hit / miss overlap 和全局 plan 的固定成本。每次都在相同全局实现上隔离 CPU 净收益，最后与 legacy_gpu 比较。

原始产物：`../plan_20261022/nsight/default.nsys-rep`、`analysis.json`、`requests.json`，`../plan_20261022/gate-final-m3.json`、`gate-final-m12.json`、`calibration.json` 和 `performance/performance-summary.json`。
