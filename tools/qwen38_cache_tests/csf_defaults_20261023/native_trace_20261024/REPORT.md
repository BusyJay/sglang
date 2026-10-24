# Native 自然聊天与长推理：trace 和优化复验（2026-10-24）

本轮以单用户自然中文聊天、长 thinking 输出为主。使用 RTX 5090 D、Qwen3.8-Flash-Next FP4-Local、native scales、W4A16 B12X decode/verify、W4A4 prefill、两步 NEXTN / 3-token verify、FP8 target KV / BF16 draft KV、7489 初始专家 slots、并发 1。未降低精度、拆分 MTP verify 或部署服务。

## 采样与配置边界

- `reasoning_workload.py` 保存两条中文自然请求：MoE 推理服务分析和带依赖约束的排程推理。通过模型 chat template 开启 thinking，没有随机 token 前缀。每个场景独立生成 2048 token，上限保留 EOS，三次交替重复。所有实测请求均达到 2048 上限，属于持续 thinking 片段，不是已经完整回答任务的质量评测。
- 吞吐来自未开启 profiler 的完整请求，主指标排除最初 128 个输出 token。完整响应、流式到达时间、MTP 接受率及 verify 次数均保存。
- 最后另发一个相同长推理请求，在生成至少 512 token 后，抓取 25 个 scheduler forward steps 的 CPU/GPU trace；分析去掉首尾不完整轮，保留 23 个完整轮 / 1104 个 target layer planner 调用。该请求不进入吞吐统计。
- 普通自然英文聊天（91→256）、短合成输入（1088→256）与 8192→1 prefill 仅作背景，保存于 `formal/`，不作为优化验收主场景。
- `SGLANG_QWEN38_OBSERVE=0`，与生产脚本匹配。最初观测开启的探索性 trace 留在顶层 `short/`、`natural/`，不作为最终基线。
- 当前同步的生产命令在本工作树上开启 HiCache 时，因 packed draft BF16 KV 和 target FP8 KV row 不匹配而启动失败，见 `startup-hicache-failed.log`。本轮隔离实例关闭 HiCache，并使用纯文本入口；其余主要计算配置保持一致。生产脚本没有修改。结论不能覆盖开启 HiCache 后的端到端行为。

## Kernel table：中文长 thinking 基线

`reasoning-baseline/summary.json`，单位 µs / target layer planner 调用。dense/MTP 等 family 时间同样除以 target 层数，不代表只属于该层；不同 streams 有重叠，累计 GPU share 不是 wall-time share。

| GPU 工作 | 累计 GPU 时间占比 | 平均 µs / target 层 |
|---|---:|---:|
| mapped-host 专家权重 fill | 24.2% | 102.9 |
| MXFP8 GEMM family | 19.2% | 81.8 |
| routed B12X | 14.8% | 63.1 |
| BF16 GEMV family | 8.3% | 35.4 |
| BF16 GEMM family | 5.8% | 24.5 |
| HC mix persistent | 6.8% | 28.7 |
| cache planner | 5.6% | 23.7 |
| GDN update family | 1.4% | 5.9 |
| alpha refresh | 0.3% | 1.3 |
| 其余 GPU 工作 | 13.6% | 57.8 |

采样区间 435.39 ms；全部 GPU 活动区间取并集后的空隙占 3.63%。这不是硬件 SM 利用率，亦不能证明非 profiler 环境下没有主机瓶颈。CPU `cudaEventSynchronize` 时间与 GPU 执行重叠，不应直接作为额外延迟加到 GPU 时间上。

## Overlap-opportunity table

| 项目 | 当前证据 | 改善方向 |
|---|---|---|
| fill 仍是第一单项 | 长 thinking 中 fill 占累计 GPU 活动 24.2%；只有 10.8% 的 fill 时间与其他 GPU 活动有交叠 | 缓存 miss 数和 fill 的暴露时间仍值得优先优化；仅凭 trace 不能断言 PCIe 已饱和 |
| shared / fill overlap 已执行 | 本地 `fill_with_overlap` / mailbox 源码及服务计数；fill 在独立 stream | 不重复增加已经存在的 shared overlap |
| resident hits 与 misses 的计算仍有依赖 | mailbox 在 fill join 之后提交 routed B12X；旧 early-cache 分支仅对 decode，verify 不适用 | 真正支持 native MTP verify 的紧凑路由拆分需要新实现和 A/B；旧 Triton hit + B12X late 方案已有负收益记录，不直接打开 |
| native planner 存在串行依赖 | 单线程对最多 32 条路由插入排序、去重并逐项查 miss | 本轮实现 warp 排序、压紧和 miss 归约，保持原先升序和所有缓存策略状态 |
| 主机提交不是采样中的第一问题 | steady trace GPU 并集空隙约 3.6% | 不把 Python 回调数量或同步 API 的累计时间直接当作可回收的 wall time |

fill 与其他 GPU 活动不重叠的区间共 101.31 ms，占采样 wall span 的 23.27%。这只量化该窗口中可见的 fill 独占活动，不是承诺可获得的加速；消除 fill 会改变资源竞争与依赖。

## Fuse-pattern table

| 路径 | 判定 |
|---|---|
| Shared expert / fill overlap | 已存在且启用，保持 |
| HC mix 多阶段融合 | 当前已经是 persistent fused kernel，包含软件 grid barrier；不能把 6.8% 全部视为尚未融合的开销 |
| HC mix 参数调优 | 八组独立权重轮换，M=1/3 的最好候选约快 4%；尚无完整模型稳定收益，不改默认。`hc-tuning.json` 保留所有结果 |
| Native 专家 64/128-bit 访存 | 上轮已测，收益约噪声量级且有 stack traffic 代价；本轮不重复启用 |
| 通用 skill 自动 fusion 提示 | 原始三表保存在 `*-triage.txt`；例如 Qwen shared append 等匹配需人工检查，不能将自动 Confirmed 标签当作本模型已验证的可用优化 |

本轮 overlap 结论来自单 trace 的区间统计和本地源码，不是 graph-off / graph-on 两 trace 的精确 Python attribution；BF16 / MXFP8 family 没有全部精确拆分到模块。没有把全部 BF16 GEMV 武断地归为 lm_head。

## Planner 改动与验证

将一个 warp 的最多 32 条路由先在寄存器中 bitonic sort，然后用 ballot/popcount 压紧唯一有效 expert IDs，并并行统计原始 misses。保持原来的 expert 升序，所以 LFU 更新次序、age、淘汰 tie break、早晚路由和 CPU hybrid 的输入语义一致。没有增加 kernel、D2H、CPU 同步、配置开关或量化变化。

- `qwen38_expert_cache.before.cu` 是本次修改前的工作树原样快照，包含用户既有修改；未用 git HEAD 覆盖已有工作。
- `check_warp.py` 完成 432 组旧/新模块逐位比较，检查 remap、counts、lookup、keys、frequency、ages、generation、clock、plan；覆盖 LFU/SLRU、32/147 slots、全 hit、mixed、miss、重复、masked、32 条路由和随机序列。
- `test_qwen38_cache_partition.py` 新增独立 histogram → 串行 policy → remap 的精确参考，检查排序去重对策略状态的影响。
- 三组相关 GPU 回归：24 passed，47 subtests passed；未放宽容差。
- `warp-check.json` 的 event 时延包含相同的状态恢复成本，不能当作 planner 纯 kernel 时延。每次重放重置真实 miss 状态，避免将 miss 预热成 hit。

完整长推理的对照、后续 trace 和 sanitizer 结果见下方最终结果及相应 JSON / log。

## 最终结果

每条自然中文提示生成 2048 token，EOS 保留。指标是去掉前 128 输出 token 后的持续生成速率，三次的中位数，单位 token/s。

| 场景 | 修改前 | Warp planner | 观测变化 |
|---|---:|---:|---:|
| 推理服务方案分析 | 99.50 | 104.27 | +4.80% |
| 带约束排程推理 | 110.11 | 111.14 | +0.93% |

这些是本机这轮前后结果，不是统计显著性的声明。只有两条提示、每条三次，所有 baseline 先于所有 candidate；不是多会话随机交错 A/B。文本、MTP 接受率及自适应层容量迁移会变化，不保证生产场景普遍获得同一比例。完整样本及接受率见 `comparison.json`。Planner 的逐位状态对照是给定相同路由的保证，不能据此要求整模型具有逐 token 确定性。

长 thinking steady trace 中，planner 从 **23.69 降到 9.87 µs / target 层调用，约降低 58.3%**；其累计 GPU share 从 5.57% 降到 2.77%。这是两个真实窗口的观测，独立的固定状态重放也确认该算子方向的收益。

不能把 trace span 的 435.39→364.90 ms 全部归因于 planner：两个窗口的 fill 从 102.91→50.52 µs / target 层，路由与驻留轨迹不同。后者不是本次代码直接减少权重字节的证据。

数值/安全检查：

- 相关 GPU 回归：24 项、47 个子测试通过。
- Compute Sanitizer memcheck：聚焦精确策略状态测试通过，4 个子测试；**0 errors**。沿用本机运行方式 `--report-api-errors no`，仅关闭 CUDA import 探测的 API 错误报告，内存检查开启。
- Black 格式检查、改动区间 clang-format 和 `git diff --check` 通过。
- 保留所有权重、KV、state 精度与 MTP 配置；没有将调优效果建立在更低精度上。隔离测试实例已退出，未重启或部署生产服务。

## 后续优化优先级

1. **针对自然长 thinking 的 cache miss / fill 暴露时间。** 首要量化每个输出阶段的原始路由、实际 miss 字节与填充时间；再比较保持同一路由的缓存策略、分层容量和传输实现。当前 fill 独占时段仍是主要可见成本，但不能由一个窗口预估固定加速。
2. **原生 B12X verify 中的 hit/miss 解耦。** 需要紧凑有效路由、与 fill 同时计算 resident hits，并控制额外 reduction/launch；保持 3-token batched verify。旧 early-cache 的负收益方案不能直接启用。
3. **HC mix 的 SM120 小 M 路径。** 仍约 29 µs / target 层，既有 SM100 的 tcgen05 split-K 路径不适用于 5090 的 SM120。单改 launch 参数本轮只有约 4% 算子收益，未改默认；更深入的替代应先通过数值与 graph 重放检查。
4. **BF16/MXFP8 常规线性层与 draft 工作。** 稳定存在且占比不低；要精确命名具体模块，仍应补 graph-off mapping。改变 lm_head 或 draft 量化会改变精度，不作为本轮改进。
5. **HiCache 兼容性单独处理。** 今天同步命令与当前 worktree 的 draft/target KV 混合 dtype 不兼容；应修复 host pool 的分配/传输契约，而不是为通过启动擅自改变 draft 精度。

本次源码变更仅为 planner 的 warp 路由 union；`planner-warp.patch` 对 CUDA 源使用本轮开始的工作树快照作基准，只包含本次改动，保留已有 CSF 修复。
