# 分层 cache 借鉴全局成本分流：实现与实测

2026-10-22，RTX 5090 D / Ryzen 9950X，CPU 物理核 2–5。

## 结论

分桶分流已经实现，正确性通过，但**不能签署分层共算性能提升**。扩大覆盖确实减少了权重搬运，却放大了 CPU 分支等待；重复重放的热权重校准偏乐观。冷权重校准后，33 个候选桶全部拒绝。因此不启用热表作为生产优化，默认配置仍关闭分层共算，未部署或重启正式服务。

本轮只修改分层分流策略及对应测试；全局分流、CPU 原生计算库和 LFU 归属规则未修改。全局最初的严格不回退目标仍以此前 [补丁验收报告](../patch_validation_20261022/REPORT.md) 为准，不能用本轮全局冒烟代替两轮性能验收。

## 实现

- 新增 schema 4 工作量桶，保留 schema 3 精确查表。key 保留 M、top-k、有效 route 总数、active 专家数、overlap 和全部 miss 的 `(distinct tokens, route count)` 分布。前缀选择后，CPU/GPU routes、专家数及 GPU fills 都由 key 和前缀确定。
- 层号、槽位容量、专家 ID 和逐 token 掩码不再切碎成本键，同工作量可跨层、跨容量复用测量；保留硬件、源码、CPU 库、backend、overlap 编排等配置指纹绑定。
- 候选排序为 `(distinct tokens, route count, expert ID)`，取测量通过且最优的前缀。区别于全局模式的 LFU-rejected 集合，候选仍是本层原始 misses，不引入全局 LFU 或跨层 victim。
- CUDA 二分 hash 后比较完整规范化 key；未知桶选 C=0，保留容量不足和 B12X 无安全 GPU route 的保护。旧 schema 3 排序/查表兼容。schema 4 至少要求三个代表。
- 静态支持检查允许同桶跨层/容量复用；无获准桶时直接回到原始 GPU 路径，不启动 CPU 服务。

实现增量见 [workload-buckets.patch](workload-buckets.patch)，相对本轮开始时已应用的补丁与修复，不是相对 Git 原基线。五个原文件快照在 `pre-change/`，当前文件校验和在 [source-manifest.json](source-manifest.json)。设计说明更新在 [LAYERED_CPU_HYBRID_DESIGN.md](../LAYERED_CPU_HYBRID_DESIGN.md)。

## 真实服务校准

都使用同一实际权重 pack 和原始路由/缓存状态，保留 shared-expert overlap、GPU fill、原生 CPU 批处理与最终合并。每个前缀 4 次预热、15 对随机交错样本，逐对检查输出与实际 CPU/GPU-fill 专家集合。候选前缀为 1–3，至少三个不同层代表全部满足：中位数节省 >5us 且 >5%、P95 改善、至少 12/15 配对获胜，才授权该桶。

基线使用原始 GPU planner/fill/B12X BF16 输出，包含新路径 FP32/merge 的额外成本；并非与同框架 C=0 比较。强制前缀的计时没有包含成本表二分查找，独立端到端测试包含。采样钩子会同步、保存/恢复缓存和统计，校准运行的服务吞吐不纳入性能对比。

| 条件 | 桶 / 代表 | 配对数 | 获准桶 | 最大输出绝对误差 |
|---|---:|---:|---:|---:|
| 重复重放，权重可留在 CPU cache | 80 / 237 | 9000 | 33 | 0.00048828125 |
| 对上述 33 桶重新测冷权重 | 33 / 99 | 4455 | 0 | 0.000244140625 |

冷校准在每个计时分支开始前，对全部候选 host 专家权重和 scale 执行 clflushopt + sfence；清除操作在计时区间之外，对纯 GPU 和共算两条分支一致。只改变 CPU cache 状态，不改变任何权重内容，也不加入生产代码。具体实现 [cache_flush.cpp](cache_flush.cpp)。冷表仅验证原 33 个热表获准桶，不声称穷尽其他工作量或其他硬件上可能的收益。

冷条件下，每个代表各先算 GPU/CPU 分支中位数，再对全部代表取中位数：

| CPU 前缀 | GPU，µs | 共算，µs | 通过代表数 |
| 1 | 382.2 | 450.7 | 0/99 |
| 2 | 380.5 | 449.9 | 0/99 |
| 3 | 379.2 | 529.0 | 0/99 |

这证实校准的 host cache 状态会改变授权结果。它与独立服务的回退方向一致，但不能把端到端变化全部归因于 CPU cache；空分流的 FP32/merge 开销、真实路由、LFU 后续驻留和 MTP 接受率也会变化。

热表 [costs.json](costs.json) **只保留为已拒绝的实验对照，不推荐启用**。冷表 [costs-cold.json](costs-cold.json) 为空表，意味着本次无获准 CPU 分支，不能拿它冒充共算提速。完整随机配对及三个层代表见 [layered-calibration.json](layered-calibration.json)、[cold-calibration.json](cold-calibration.json)；汇总见 [calibration-summary.json](calibration-summary.json)。原始诊断快照的 status=running 不作为完成标记，两个父级 `calibrate*/performance.json` 的 passed 记录实际完成全部请求。

## 独立无采样服务 A/B

同本地镜像、模型 revision、MTP 2、breakable decode graph、adaptive 分层 LFU 和低频 stats_interval=19200。3 个 1088 输入/512 输出短请求取中位数，再执行自然聊天/512 输出、170064 输入/8 输出、长输入后短请求/512 输出。每个进程启动后预热；顺序为新分桶 → 原 GPU。GPU 对照为原 Git 提交 `9321a3f4504534811676c4d1f39d62341089e610` 的独立 worktree。

| 指标 | 原分层 GPU | 热表分桶共算 | 变化 |
|---|---:|---:|---:|
| 短 decode 中位数，tok/s | 103.916 | 101.266 | -2.55% |
| 自然聊天 decode，tok/s | 100.992 | 94.448 | -6.48% |
| 170k TTFT，s | 30.074 | 30.158 | +0.28% |
| 长输入后短 decode，tok/s | 93.971 | 83.983 | -10.63% |

同为 57600 次 runtime 调用的最近周期统计快照：

- 分桶共算实际 CPU 调用 4701 次，CPU 专家 12616 个；原始 miss=163529，GPU fills=150913，满足原始 miss = CPU 专家 + GPU fills。
- GPU decode copy bytes：原 GPU 463527014400（431.69GiB），分桶共算 417244262400（388.59GiB），约 -9.98%。**这是同调用计数的周期快照，不是整个请求组最后一次调用后的总量**。CPU 直接绕过的 payload 为 12616×2764800 字节；两者差别还包含后续 cache 驻留和路由变化，不能全部归为直接绕过。
- 自然聊天 MTP accept length：GPU 1.9104、共算 2.0562；长输入后短请求：GPU 2.4038、共算 2.2358。固定输入与 seed 不保证两个数值路径生成相同文本或相同推测接受率。

全量事件、输出文本、命令与日志：[bucket-r1](bucket-r1/performance.json)、[gpu-r1](gpu-r1/performance.json)。分析 [analysis.json](analysis.json)。单组 A/B 已显示这张热表的明显风险，冷条件也拒绝了所有桶，因而未继续反复测量这张已判定不适用的热表来挑最好结果。未锁 GPU 时钟，也不声称约 1% 的差异有统计显著性。

## 验证与复现

- [tests.log](tests.log)：128 passed、328 subtests passed；包含全局和分层 policy/dispatch、resize、loader、close，以及真实 CUDA planner。新用例覆盖工作量等价、跨层/容量、token 掩码变换、不同 CPU token 工作量隔离、overlap 隔离、旧表兼容、重复 route、B12X 屏蔽、统计与未知桶回退。
- [ruff.log](ruff.log)：本轮 4 个 Python 源/测试文件检查通过；git diff --check 通过。
- 真实共算和 GPU 对照均完成 graph serving、短→170k→短、KV stages 80k/160k/220k 及返回。没有执行全量模型 accuracy 基准或多小时压力测试。
- 同一张 5090 串行测试，未使用另一张 GPU。测试结束清理临时容器，保留原始证据；未部署正式服务。

用新输出名复现，勿覆盖已保存测量。表与当前源码、CPU 库和硬件绑定：

```bash
cd tools/qwen38_cache_tests/layered_bucket_20261022
bash replay.sh fresh-gpu --baseline --mode off --stats-interval 19200
# 下行仅复现已拒绝热表的对照，不是推荐启动配置。
bash replay.sh fresh-warm-diagnostic --mode layered --layered-table /results/costs.json --stats-interval 19200
bash replay.sh fresh-cold-fallback --mode layered --layered-table /results/costs-cold.json --stats-interval 19200 --short
```

重新标定可用同目录的 run_variant.py / run_variant_cold.py、benchmark_calibrate.py 和两套 sitecustomize.py；冷标定需先编译 cache_flush.cpp。使用新证据目录更改硬编码 /results 路径，保留本次配对记录。当前的下一处性能工作应针对冷权重下 CPU 服务/解量化耗时和零分流固定开销，随后重新标定；继续增加热表的分流比例不足以解决本次回退。

## 最终回退与全局检查

冷空表的真实 serving 检查 [cold-fallback](cold-fallback/performance.json) 通过：短请求中位数 102.617 tok/s，自然聊天 103.763 tok/s；未创建有效 CPU 分流，直接回到原分层 GPU 路径。这仅证明无获准桶的回退有效，不计作共算性能收益。

本轮全局服务 [global-smoke](global-smoke/performance.json) 通过，短请求中位数 103.009 tok/s，自然聊天 103.349 tok/s。最近 38400 runtime 调用统计中有 6620 次真实 CPU 调用、15802 个 CPU 专家；miss/admitted/rejected/CPU/GPU 计数守恒。它是全局路径的短服务回归验证，不代替此前严格无回退的性能验收。
