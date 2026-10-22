# Strata 对照：分层共算为什么收益小

2026-10-22 开始，2026-10-23 完成。我们的实现提交 `8b02621849`；只读检查 Strata `011c9e600a93bbc9fd451c563501b5831c32bbca`。本轮没有改正式计算路径或 Strata 源码，没有启动完整 Strata engine。

**结论：当前主要限制是路由提供的 CPU 工作太少，以及执行流水线的固定成本；CPU 内核也有优化空间，但它不是一个数量级的差距。当前测量不支持“共算已稳定提速”，也不能据此否定共算。**

## 1. 当前算法的收益上限很低

之前完整组的 57600 次 layer-runtime 调用：154926 个原始 miss，CPU 只算了 1095 个，即 **0.707% 的 miss**，其余 153831 个仍搬 GPU。一个专家 payload 2764800 字节，CPU 直接绕过约 **2.82 GiB**。不能把总 decode-copy 快照从 431.73 GiB 降到 396.10 GiB 的全部差额归因于 CPU：大部分来自 LFU 驻留、实际路由与调用组成的变化。

这符合用户确认的规则：miss 必须严格冷于本层 eligible GPU victim，且冷候选数量 C≤n，才留 CPU；C>n 全 GPU。热度包含 CPU 路由，并可触发迁回 GPU，正确性已验证。规则是保守的 admission 判断，却没有回答“当前 CPU 与 GPU 两条支路各有多少余量”。多个候选时全 GPU 还会使 CPU 工作量出现不连续的跳变。

当前同源对照：n=0 无 CPU 为 104.475 tok/s，n=1 为 102.355，n=2 为 101.562（短组）；自然聊天的 MTP 接受长度不同，不能拿它单独证明 CPU 收益。原 Git→n=1 的 +5.78% 短请求提升包含原补丁、热度策略等变化。详见 [上一轮报告](../layered_lfu_20261022/REPORT.md)。

代码：[候选选择](/home/jay/workspace/sglang/python/sglang/kernels/jit/csrc/moe/qwen38_layered_policy.cuh:28)。本轮没有擅自放宽已确认的候选规则。

## 2. Strata 的结构差别

| 项目 | 当前分层实现 | Strata 本地源码/文档 |
|---|---|---|
| 执行位置 | 低于 victim 热度且 C≤n 的 miss 才在 CPU | 在非驻留 miss 集合内按 PCIe 比例或成本模型分 CPU/GPU，不以 victim 热度为执行位置门槛 |
| GPU miss | 本层 cache fill 后计算 | 临时 staging 或映射地址参与本次 GPU 计算，长期 cache adaptation 单独处理 |
| 热度 | 每次 decode/verify，原始 unique route union 加 1，包括 CPU | 分流前按每个原始 route 计 usage，adapt 窗口后衰减；包括 CPU。计数单位和衰减周期不同 |
| CPU/GPU 流水 | 同 stream 的 fill 在 routed GPU 计算前；CPU 结果最终 merge | 先发布 GPU plan、提交 fetch，再算 CPU；resident GPU 工作可与 CPU、搬运并行 |
| 搬运 | GPU SM 执行 mapped-host load 并写入 cache | 默认 fetch 可用 copy engine；也有 device-alias 路径。不能把“有映射地址”都当 DMA |
| CPU pool | 最多 4 物理核；dispatch 条件变量；一批专家按专家顺序执行，内部阶段屏障 | 默认按物理核选 worker，host 可参与；短时自旋后睡眠；跨专家合并行任务，gate/up 与 down 分阶段 |
| CPU 算术 | NVFP4 权重，BF16 activation，AVX512-BF16 dot | NVFP4 权重，Q8 activation，VNNI dot，额外 activation 量化和软件预取 |

主要源码：Strata [分流前 usage](/home/jay/workspace/Strata/src/core/expert_source.cpp:4824)、[miss 分流](/home/jay/workspace/Strata/src/core/expert_source.cpp:4878)、[staging / publish / fetch](/home/jay/workspace/Strata/src/core/expert_source.cpp:4940)、[窗口衰减](/home/jay/workspace/Strata/src/program/generate.cpp:10315)、[pool](/home/jay/workspace/Strata/src/kernels/cpu/pool.cpp:976)、[NVFP4 CPU 算子](/home/jay/workspace/Strata/src/kernels/cpu/nvfp4_avx512.cpp:1)。我们 [fill](/home/jay/workspace/sglang/python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu:15)、[batch 内逐专家执行](/home/jay/workspace/sglang/python/sglang/kernels/ops/moe/cpu_nvfp4/kernel.cpp:443)、[service 每次空轮询 yield](/home/jay/workspace/sglang/python/sglang/kernels/ops/moe/cpu_nvfp4/kernel.cpp:547)。最后一项是源码发现，本轮没有单独测出其端到端损失。

Strata 的自动比例和可选 per-layer 成本拟合也不是无条件最优。应借鉴的是执行/驻留分离、并行发布、冷态测量与本机校准，而不是照搬某个固定比例。

## 3. 同权重 CPU 算子实测

本机 Ryzen 9950X，16 物理核、32 线程；RTX 5090 D。选真实 pack 的 layer/expert (0,7)、(22,111)，各测 gate/up 和 down。共同的持久自旋行任务池；同一组 BF16 输入；输入量化、格式转换、clflushopt 在计时外。Strata 行算子直接链接其未修改的源文件。两个实现各自使用其权重布局；不是整个 MoE 或整个 engine 对比。

每个条件 31 个计时样本，随机交替执行；表中时间为两个专家各自 gate/up+down 中位数之和的平均，**不包括 SwiGLU、全局 scale、路由加权、Q8 量化及实际生产线程池**。

| 核数 | tokens | 权重状态 | 我们 BF16 行算子 µs | Strata Q8 行算子 µs |
|---:|---:|---|---:|---:|
| 4 | 1 | 热 | 35.6 | 31.2 |
| 4 | 1 | 冷 | 80.5 | 65.9 |
| 4 | 3 | 热 | 39.8 | 47.9 |
| 4 | 3 | 冷 | 78.0 | 67.2 |
| 8 | 1 | 冷 | 75.6 | 67.7 |
| 8 | 3 | 冷 | 76.8 | 68.6 |
| 15 | 3 | 冷 | 76.4 | 70.1 |

4 核冷态 Strata 延迟约低 14%–18%，但 3 token 热态我们的 BF16 更快。加核对冷态纯行算子的收益趋于饱和，不能按核心数线性外推。尤其本机 9950X 与文档中的 9950X3D 缓存不同。

对同一 BF16 输入的 FP32 解码参考，BF16 行算子最大 relative-L2 约 9.4e-8，Strata Q8 路径约 0.00603。因此 Q8 不是数学等价的替换；本轮仅作诊断，未做全模型质量评估，也未集成进正式代码。详细数据 [rows.json](rows.json)，复现 [bench_rows.py](bench_rows.py)、[row_bridge.cpp](row_bridge.cpp)。

## 4. 线程唤醒和实际 CPU expert 实测

复制当前 native library 到诊断目录，保留相同计算，仅把 pool dispatch 改为最多自旋 20 ms 后再睡眠；两份诊断库都放宽测试核心数上限至 16。实际 production 文件未改。两种 pool 不同时占用相同核心；创建 pool 不计时。每条件 31 样本，输出逐元素相等。

| 条件 | 当前 CV pool µs | 诊断 bounded-spin pool µs |
|---|---:|---:|
| 4 核空任务，连续 dispatch | 3.7 | 2.3 |
| 4 核空任务，间隔 200 µs | 5.1 | 1.6 |
| 4 核空任务，间隔 2 ms | 19.1 | 1.1 |
| 4 核空任务，间隔 25 ms | 20.9 | 16.1 |
| 4 核，1 expert、1 token，冷 | 129.1 | 125.9 |
| 4 核，1 expert、3 tokens，冷 | 135.4 | 133.3 |
| 8 核，1 expert、3 tokens，冷 | 110.3 | 100.0 |
| 8 核，3 experts、3 tokens，冷 | 300.5 | 251.4 |
| 15 核，3 experts、3 tokens，冷 | 301.5 | 294.2 |

唤醒问题存在，但不等于 CPU 总时间可按空任务比例缩短。当前少量 CPU miss 的冷态完整 expert 计算仍约 130 µs。8 核在部分条件更好，15 核未继续改善；多核屏障、跨 CCD、带宽和低负载调度都可能影响，不能直接把上限改 15 当作优化完成。部分 p95 有毫秒级离群，本轮是微基准诊断，未声称尾延迟达标。

数据 [pool.json](pool.json)，脚本 [build_pool_variants.py](build_pool_variants.py)、[bench_pool.py](bench_pool.py)。完整 native 计时包含校验、路由处理、分配、SwiGLU、gate/up、down、加权求和；与上一节纯行算子不能直接等同。

## 5. 去除 prefill 的 decode trace

使用 `llm-torch-profiler-analysis` 技能，分别运行同源 GPU-only 和 layered_lfu n=1。服务配置保存在 profile-gpu/command.json、profile-lfu/command.json。1088 输入、512 输出、MTP2，先预热。原 trace 混有初始 prefill，直接看原始 H2D 占比会误诊。

按第 49 个 decode_plan_kernel 的时间戳截取，即丢弃 prefill 和第一整个 decode/verify 轮；各保留 **1104 个 planner = 23×48 层**。Chrome trace 留在本地 profile-*/decode-only.json.gz，详细统计 [decode-summary.json](decode-summary.json)。这些是带 Python stack 的 profiler 时间，不是无 profiler 吞吐，也不是严格加和的 critical-path 时间。

### Kernel table（本次诊断重点）

| kernel | GPU-only 总 ms | LFU n=1 总 ms | 说明 |
|---|---:|---:|---|
| mailbox_fill | 89.47 | 94.71 | GPU 发起的 mapped host 权重读取，消耗 SM；普通 H2D memcpy 表不会体现这些字节 |
| routed B12X | 66.28 | 66.37 | 前者 BF16 输出，后者 FP32 输出；这次差距很小 |
| decode_plan | 22.32 | 34.85 | LFU 增加 12.53 ms，包含 CPU-inclusive 热度和选择逻辑，不能全归为 CPU dispatch |
| hybrid_merge | 0 | 1.65 | LFU 1104 次均执行，median 1.47 µs；没有看到长时间等待 CPU 的 merge 尾部 |
| 常规 pinned H2D memcpy | 0.019 | 0.013 | prefill 去除后，不是主要 decode 搬运指标 |

累计 GPU kernel/memop 时间 436.28 / 454.03 ms，可重叠，不可直接换算吞吐损失。保留区间跨度 569.81 / 584.92 ms，仅供诊断；CPU Python profiler 会扰动 wall time。planner 的额外代价确实遍布每层，CPU 直省搬运却极少，这比“FP32 B12X 很慢”更符合这次 trace。

### Overlap-opportunity table

| 项目 | 当前证据 | 判定与验证方式 |
|---|---|---|
| resident GPU 计算与 miss 搬运重叠 | 当前 fill→routed compute 同 stream；Strata 先 publish hits / fetch | 值得实验的结构差别；需要拆 resident/miss 与 staging 后重测，尚无端到端收益数字 |
| CPU 与 PCIe 平衡 | 本轮 CPU miss 少，merge 最大约 1.95 µs | 没有证据表明 CPU 等待支配这次 workload；不能仅凭 CPU 内核快慢选择固定比例 |
| 线程唤醒 | 独立微基准测得 2 ms 间隔 CV 19.1→spin 1.1 µs | 已测局部机会；未集成、未宣称端到端同幅提升 |

### Fuse-pattern table（人工复核）

| 项目 | 源码/trace 证据 | 结论 |
|---|---|---|
| planner 热度更新与选择 | 同个 planner 里已有并行 heat 和 histogram，仍有选择/扫描/同步 | 应先优化或剥离开销；没有现成可直接套用的标准 fusion |
| FP32 输出与 merge | B12X 时间几乎相同，merge 总 1.65 ms | merge 可压缩但不是本次最大开销；C=0 快路径需避免额外 host 同步 |
| 技能自动匹配的 Qwen2 shared-expert / 通用 activation fusion | 见原始三表，其中将 GDN activation 匹配到 Qwen2 家族 | 属于规则自动提示，本模型/后端未确认适用，不能把提示的整组 GPU 时间当作可节省时间 |

保留技能的原始三表：[GPU decode-triage.txt](profile-gpu/decode-triage.txt)、[LFU decode-triage.txt](profile-lfu/decode-triage.txt)。这是两个独立 single-trace triage，不是 graph-off mapping / graph-on formal 双 trace 的 overlap 归因实验。

## 6. Strata 宣传数字的比较条件

[Strata NVFP4 文档](/home/jay/workspace/Strata/docs/NVFP4.md:44) 给出的 5090 + 9950X3D、NVFP4 400-token answer 是 **114–116 tok/s，约 8350 cache slots**；我们当前组约 100–104 tok/s，stage0 约 7006 slots。Strata 的 dense MXFP8 转 Q8_0、部分 BF16、MTP Q2 draft 与我们原精度/draft 不同，context、输入、接受率也不完全相同。这不是同条件的实测差距。

同文档 NVFP4 PCIe share sweep（文档测量，非本轮复现）0.55→0.25 时短/长由 103/105 到 117/124 tok/s，说明负载平衡可有用，但不是证明我们的同配置可获得相同收益。

[200+ tok/s 的 callback-affinity 报告](/home/jay/workspace/Strata/bench/results/2026-10-11-rtx-5090-callback-affinity/README.md:9) 用的是 IQ3_S，且 4K 142.82→207.75 的 +45.5% 对应 CUDA callback 线程继承 CPU 单核 affinity 的修复（回调 median 984.51→4.01 µs）。我们当前 CPU native service 不通过 CUDA host callback 完成核心计算，不能直接认领这个修复机会。

[tail skip](/home/jay/workspace/Strata/docs/NVFP4.md:90) **在 NVFP4 默认关闭**，因为改变结果明显；其他量化路径的跳过低权专家结果不能当作完全计算的免费加速。也不能反过来把 Strata 的 NVFP4 默认速度归因于 tail skip。

## 7. 后续优先级

1. **先分离归因**：固定 n=0 / n=1 使用同样 CPU-inclusive LFU，配合原 GPU-only；统一 route replay、MTP 接受长度、cache 容量与冷权重，分别记录 candidate 数分布、实际 CPU 数、分支完成时间和字节量。避免继续用原 Git→新实现的总增益代表共算增益。
2. **减少每层固定支出**：优化 planner 扫描/排序/同步；保持 C=0 尽量接近 n=0 的执行成本。trace 中 planner 比 merge 更值得优先处理。保留现规则作为可比较模式。
3. **新增执行位置独立的实验模式**：LFU 决定长期驻留/晋升；另按 CPU、PCIe、GPU 实测完成时间选择本轮位置，用 staging 承接 GPU miss。成本目标是 `max(T_CPU(k), T_GPU_and_transfer(M-k)) + T_merge`，但必须计入共享内存带宽争用及测得的实际重叠，而不是只按 bytes/bandwidth。冷于 victim 的规则可以保留为候选约束做第一组对照；若它仍只有 0.7% 覆盖，再用独立新模式放宽，不能悄悄更改当前用户已确认语义。
4. **配合流水线和本机核心数校准**：resident GPU 先算、搬运并行、CPU 跨专家行任务；4/8/15 核都测，不能默认越多越快。短时自旋作为单独消融，计入空闲 CPU 占用。
5. **最后再评估 Q8/VNNI 路径**：目前冷态纯行算子约 14%–18% 延迟差，属于次一级机会，需全模型质量与端到端验证。先利用分流和流水线，才可能把 CPU 局部优化放大。

没有执行完整 Strata E2E：当前没有同精度、同 cache 容量、同 draft 的可直接等价部署；转换它的完整模型又会引入 dense/draft 精度和布局变化。本报告将源码机制、Strata 自述数字、同机行算子测量、我们自己的 profiler 和既有 E2E 明确分开。当前结论是有证据的根因诊断，不是新算法已达标的性能报告。
