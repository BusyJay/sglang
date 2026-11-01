# 上游机制审查：GLM MLX 可利用的改进

2026-11-02 在 BlazesMacStudio remote 仓库重新 `git fetch origin main` 后，main 为 `e2982985110fb432fd8ddba5f0147a5004fd03ab`（2026-11-01 22:21:05 +0800）。审查原基准 `6310a056f6` 之后的 298 个提交，按 kernel / model / scheduler 调用路径筛选，再读取以下实际 diff 与当前 MLX 实现。merge 已完成，但不是优化成果；下列“候选”尚未在设备或完整模型测试，不计入本轮收益。

## 优先候选

| 上游提交 | 实际机制与减少的成本 | 当前 MLX 对照 | 采用判断与验证条件 |
|---|---|---|---|
| `302b78ccf0` #46024 | 小 batch 时每行拆成 4/2 个 CTA；每个 CTA 用相同完整 reduction 计算统计量，只写互斥列，避免跨 CTA reduction。将 MXFP8 padding scales 清零并入 kernel。 | `mlx_hc.py:147` 当前每 token 一个 threadgroup；`mlx_hc_mix.py:130` 已有另一种 projection tile 并行。GLM `_branch_inputs` 的 fused collapse 分支有资格限制，实际是否启用必须看 runner 设置；通用路径仍有逐 row compile。 | **值得做独立候选**：借鉴“重复相同统计、分割输出”的小批量并行思路，不能直接复制 MXFP8 方程。保留 GLM 的 collapse BF16 舍入以及 normalization 再 BF16 后乘 weight 的两处边界；与本轮失败的 deferred residual packet 分开。重复读取/Sinkhorn 可能抵消并行收益。先覆盖真实 AR/MTP2，再测每 token 首次编译与完整模型。 |
| `974831643f` #46184 | width=4 线性 verify，把重叠卷积快照的 `3T` 写入改成同一底层窗口的 `T+2` tile 写入；逻辑窗口通过相同步长的 view 读取。 | `mlx_kda.py:88` 为每个 token 写三个窗口值；输出形状 `(T,3,dim)`，`glm5_next.py` 用 token 切片发布消费前缀。KDA 已将 conv/recurrent 算术融合，仍存在重复 conv snapshot 数据。 | **可复用布局思路**。MTP2 的 T=3 时 9→5 元素/通道，减少约44%卷积快照写入；T=1 无收益。需改 kernel 输出及 snapshot view，保留每个接受深度的状态与请求 ownership。FP32 recurrence snapshot 不随之缩小，而且通常更大；不能按44%估计全模型速度。树状验证不能未经推导沿用线性窗口。 |
| `2ccbdab65f` #47387 | 根据每请求实际长度计算 tile 数，累积并把所有行的 tile 均衡分给 workers，避免固定每行 CTA 导致长请求尾部等待。保留 BF16 rounding 点。 | `Indexer.__call__` 当前通过 MLX matmul、ReLU、head reduction 与 argpartition 计算 pooled scores（`glm5_next.py:1326`）；没有上游那个固定每行 persistent worker 循环。 | **长短混合请求 indexer 的设计候选**，不是现存瓶颈证据。先确认实际多请求调用是否合批以及时间分布；需要专用 ragged scorer 才能利用。长度/offset 作为运行时参数，不能每步重新编译或 CPU 同步计算 schedule。不能把 NVIDIA SM 数和 FP4 格式直接搬到 Metal。 |

## 已有机制与暂不适用项

| 上游提交 | 核查结论 |
|---|---|
| `f2d3188925` #46090：路由缩放并入 pre-dispatch | MLX `MoE.__call__:784` 已把 `routed_scaling_factor` 交给 fused router；通用 selection 也接收该参数。不存在这条 CUDA 路径的独立 tensor multiply，**无需重复实现**。 |
| `bd1eb0bb89` #44465：分片接受采样避免 gather 全词表 | 单机 MLX 无 TP draft-vocab all-gather。当前 `target_tokens` / `draft_token(s)` 对 dense head 已用直接 argmax kernel，greedy token-only 合同明确；要求 logprob/processor 或 sampling 时仍需保持数学语义。分片采样是未来 TP 思路，**当前无可消除的对应通信**。 |
| `f171859219` #47684 / `da4ffac806` #47328：按完整 SM wave 分 tile、仅能少一轮时选更高 CTA residency | 有价值的 geometry 选择原则：增加并行度要能减少执行轮数，并权衡单 CTA 成本。当前 MLX argpartition/普通 matmul 无同一 launch plan，**不能直接套 SM/CTA 常数**。需要实际 Metal kernel profile 后才判断是否收益。 |
| `9767ce3a5a` #47550：metadata 用 kernel 参数写入以避开 copy-engine 队列 | 当前 MLX 不走该 CUDA pinned DMA/HiCache metadata 通路。运行时参数避免值特化的原则已用于本轮 F 的 split/visible ends；**没有证据支持再增加 metadata fill dispatch**。 |
| `20036b7add` #47115：大而窄的 padding buffer 用 strided grid 填充 | 当前 GLM MLX expert dispatch 没有 mega-MoE 那种相同 capacity buffer 清零调用；**无直接成本可省**。 |
| `2f6e7f3475` #44718：host-only Mamba checkpoint、请求 ring 与 GPU slot 解耦 | 当前统一内存下 host/GPU 存储域不同的 CUDA 容量收益不能直接照搬。请求生命周期与 ring ownership 解耦值得参考，但不是 expert 权重 LFU，也不省量化解包。现有 MLX snapshots 按请求管理；需独立容量实验后再改变状态缓存。 |
| `006ad25169` #45174：top-k ties 按低 index 决定 | 是将来替换 MLX indexer/top-k 时的正确性要求。当前 argpartition 的已有选择行为应先固定，不应无验证更改 tie 集合。 |
| `3a01d4cc8c` #46840：模拟接受率也保留真实验证算术 | 本轮全模型使用真实 token/acceptance，不模拟跳过 verify cost。**用作实验审查规则，无额外优化要移植**。 |

## 直接集成的服务改动

`b8d86350d2` owner-list 接口要求 MLX overlap launch 调用 `_join_launched_reqs`；`e3718813ec` 增加 scheduler iteration 计数；`c105a91b43` 修复无法重建 KV 的 retracted request abort。这些通过 merge 接入，并以合并后 scheduler/runtime 16 项与 MTP2 双槽 HTTP 验证。它们有生命周期/可观测性价值，未测独立吞吐提升，不应包装为 B 调度性能改进。

## 决策

本轮保留已验证的 E 和 F 可见范围合同；上游没有可以无验证直接替换 B/C/D/F 的 MLX kernel。下一轮建议先做 **KDA 去重卷积快照**（收益对象与语义清晰），再做 **mHC 小 batch 分割输出**（需要完整模型对照）；长短混合 indexer 先收集真实批次与 tile 工作分布。本轮收尾不再扩展新 kernel 实验。审查依据的提交 diff 保存在 `results/upstream_review/`，以及仓库中对应提交的 git blob；未宣称 Metal 平台不支持这些机制。
