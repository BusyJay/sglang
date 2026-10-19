# 路由开销与 prefill workspace 复验

这是前一轮 `investigation/INVESTIGATION.md` 之后的源码优化，仍基于 d3eb24c40031999b3e62e4e9351c5b6118dfaa24、同一 5090 D / FlashInfer 0.7.2 镜像和 FP4-Local checkpoint。三个选项仍默认关闭，未更改生产部署。

## 不依赖 allocator 参数的 prefill 峰值处理

开启 prefill_direct_cache 的普通 BF16 输入 CUTLASS 路径，每个内部 MoE 调用最多处理 2048 tokens；外层 chunk 仍为 8192，专家规划、权重填充各执行一次，不按子块重复搬运。子块输出立即写回独立输出缓冲，包含尾块。attention/GDN 的外层 batch 不变。

相同 6388 基础 slots、MTP=2、static=0.90、原 KV stages / 容量、三个开关均开启，**没有设置 PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF**：

- 无 profiler：1K、8K、74K、170K 请求通过，算术探针返回 42。见 `model-chunked-moe/report.json`，入口 `run_model_chunked_moe.sh`。
- 独立 allocator history：1K、8K、74K 与探针通过，记录没有 OOM 事件。`memory-summary.json` 中设置为空、max_split_size=-1。
- 原先单次 CUTLASS 最大申请约 839.76 MiB，现在最大 217.66 MiB，下降约 74%。这是单次 workspace 申请的变化，不能称作整个模型显存峰值降低 74%。
- 4101-token 合成对照覆盖两个完整子块和尾块，和未分块 resident CUTLASS 的最大绝对误差 0.00048828125，原 rtol/atol=0.02 未放宽。见 `large-prefill.json`。

该方案增加内部调用和输出复制，以降低瞬时 workspace。预量化输入带有特殊 scale 布局，保留原路径，不适用此上界；W4A16 路径也没有套用 CUTLASS 分块。不构成任意 GPU、并发或长时间运行无 OOM 的保证。之前的 allocator 参数仍是另一种经验证的缓解方案，此普通输入路径已不依赖它。

## 两个计算开关的实际优化

Prefill 直接使用运行时 GPU lookup，以受保护 slot 前缀排除借用 arena 覆盖的尾部；一个 routing kernel 输出 cache/arena IDs。移除每次 CPU 字典转 CUDA expert/slot 列表，也不再生成两份零掩码权重；无效 ID=-1，保留原路由权重。

Decode planner 在修改 lookup 前保存命中并直接输出 early/late IDs。只有混合命中才提前计算有效 hits；全命中/全缺失的 early IDs 全是 -1，Triton projection 直接跳过 dot product，late 后端处理有效路由一次。去掉重复的 ATen 命中快照、权重掩码和无必要的输出 clone。没有增加逐 token 的 CPU 命中判断或 D2H 同步。

**全命中仍会提交空的 early kernels、归约和合并，并未做到零额外 launch。** 混合命中时 early 仍是 Triton；B12X 的无效路由仍有冗余工作，尚未实现有效路由压紧或原生双权重源后端。不能把这次改进描述为“两个开关已有净收益”。

同一 5 次热调用的 profiler 方法，计数只包含 cat=kernel，不混入 GPU annotation：

| 每层每调用 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 全命中 decode kernels | 23 | 12 |
| decode kernel 累计时间 | 144.4 us | 39.6 us |
| early 两次 projection 合计 | 97.1 us | 2.7 us（空分支） |
| 3-token prefill kernels | 41 | 21 |
| prefill kernel 累计时间 | 136.7 us | 114.4 us |
| 新增路由元数据 H2D | 2 | 0 |
| 对应 cudaStreamSynchronize | 2 | 0 |

Prefill 原有 histogram/planner 的 D2H 和必要事件等待仍存在；没有把其计作新引入或宣称消除。原始新 traces 在 `profiles/`，汇总见 `component-summary.json`。对照原 traces 在 `../investigation/profiles/`。

随后在同一空闲 GPU 上顺序重跑优化前源码快照和优化后源码，固定合成权重/路由，每例 30 次热重复调用，CUDA event 中位数（包括主机提交间隙）：

| 小批量热场景 | 原开关实现 | 本次实现 | 本次关闭开关 |
| --- | ---: | ---: | ---: |
| 1-token decode 全命中 | 0.551 ms | 0.462 ms | 0.272 ms |
| 3-token prefill | 0.621 ms | 0.455 ms | 0.298 ms |

相对原实现分别降低约 16% / 27%，但仍慢于关闭开关。关闭开关的 prefill 对照在前后两轮为 0.339 / 0.298 ms，表明还有主机计时波动；这不是交错多轮置信区间，也不是完整模型吞吐收益。原始数据在 `controlled-before-b12x.json`、`ab-b12x.json` 和 `performance-summary.json`。优化前源码快照在 `before-source.tar.gz`，旧交付补丁另存为 `before-optimization.patch`。

剩余减速有明确原因：热 prefill 的旧 arena 已驻留，省不到权重复制，却仍需两套 CUTLASS dispatch、两次结果舍入和合并；decode 虽把额外 GPU 投影算时大幅降下来了，仍多提交空 kernel、事件和 Python 回调。在缺少高效原生混合路由后端时，不建议默认打开两个计算开关。实际 miss/PCIe 重叠收益必须另测，不能从热重复测试推断。

## 正确性与验证边界

- 348 项 CPU 回归通过；7 项运行时边界测试通过。
- 8 组 B12X / W4A4 prefill 的 eager / breakable 开关矩阵通过，容差未放宽。
- 新 GPU routing/planner 测试覆盖借用尾部、非法/填充 ID、全命中、全缺失、混合命中、重复路由及 LFU/SLRU；有效路由在 early/late 中只出现一次。
- 最终边界/kernel/GPU cache 联合回归 48 项通过（含 7 项 CPU 边界测试），16 项子测试通过；Triton W4A16、W4A4 与共享分支合成校验通过。
- GPU cache 回归原有迁移测试调用已删除的 `_maybe_rebalance()`，并假设 run 自动迁移。已更新测试以模拟 snapshot -> result completion -> publish；借用 arena 的 tiny verify fixture 在快照前显式归还所有权。生产维护流程未增加等待；所有原有权重、布局、指针和跨 stream 断言保留。
- 新 operator 具备 lazy registry 元数据、注册 GPU 数值测试和 marker microbenchmark。namespace 测试 70 项通过。
- 本次涉及的注册项 AST 检查通过。整树注册审计仍被既有 `test_qwen38_preallocated_gdn.py` 缺注册阻挡；既有 expert-cache CUDA 源也已有整文件 clang-format 违规，本次未混入整文件格式重排。Python lint、diff 检查及新增 CUDA header 格式检查通过。

完整模型请求使用合成输入并有前缀复用，生成数量有限；证明该请求序列可跑通及探针正确，不是长时间压力、全模型精度或生产 p99 测试。`preserve_cache_on_resize` 的生产实现沿用上轮已验证版本，这次没有增加另一套搬迁机制。
