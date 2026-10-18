# FP4-Local cache 策略实测：固定路由与真实 GPU 搬运

结论：**在本次两份固定路由上，当前实现的衰减 LFU 最快。** SLRU(0.8) 是最接近的候选，但仍多耗时约 32%–33%；LRU 多耗时约 60%–63%。差异明显超过这批测量的重复噪声。没有据此切换线上淘汰算法。

新包原始提交 `3baf2f484d61c9bdfecfbca87ff81cd3e88e5ce5` 已从 `6b7af74a22` 快进合入 `qwen3.8-flash-next-offload`。本提交另保存录制与实验集成修正、原始结果和诊断 trace。模型运行代码与 CUDA kernel 没有修改。语言服务已恢复、HTTP health=200，ViT 未重启；原生产启动脚本 SHA256 与实验前相同，prepared=1、fill CTA=adaptive 仍生效。

## 比较方法

- 当前 `swallow-sglang:nightly-dev-20261018-b234b5c3`、RTX 5090 D、driver 615.78.08；pack revision `b9ce18a6887e18c089fdea8509a33e63429a3300`。完整镜像 ID、manifest、源文件和 tape 校验和见 `live_fp4_results/`。
- 两个独立录制 worker，各发送一个合成请求：英文代码设计、中文 Raft 设计；输入分别为 94 / 104 tokens，输出上限均 3072。实际输入 IDs、响应 metadata 均存档。
- 正常 MTP=2 下，各录 1024 个完整 target verify sweep；每个 sweep 48 层，全部为原始 `3×10` expert IDs。**不是每个算法重新自由生成**，而是逐层回放同一份完整路由，保留顺序及重复 ID。
- 5 个配置：LFU(decay=0.95)、LRU、SLRU(protected fraction=0.5/0.7/0.8)。LRU 由 SLRU(protected=0) 实现，符合包内及当前运行时的定义。
- 每个配置每次使用新进程、空 cache、相同 128 sweep 预热，随后计时相同 896 sweep。容量固定为前 6 层 134、其余 42 层 133，合计 **6390 slots**；关闭在线 rebalance。不同算法形成不同的 cache 状态和 miss 计划，这是待测结果。
- 每个 workload 做 5 轮，轮内随机排列算法，种子 20261018，共 **50 个独立 GPU 计时进程**。计时包含真实 planner、native fill、事件依赖和末尾 GPU 完成；编译、注册、输入 tensor 分配与预热在区间外。
- GPU 源权重使用与当前服务一致的 shared THP cache；运行参数在独立 shell 脚本中。没有修改生产 YAML 或启动脚本。

这是 **planner / 权重搬运回放**。attention、GDN、shared expert、B12X 和 draft 计算均未执行；不把结果换算为端到端 token/s。录制读回与 payload 校验另起进程，均未进入正式计时。

## GPU 结果

时间为 896 sweep 的中位墙钟秒数；增幅为同轮相对 LFU 的成对耗时比的几何均值，正值表示更慢。

| 策略 | 代码：秒 | 相对 LFU | Raft：秒 | 相对 LFU |
|---|---:|---:|---:|---:|
| LFU | **9.1713** | 基准 | **7.9226** | 基准 |
| SLRU(0.8) | 12.1830 | +32.80% | 10.4757 | +32.27% |
| SLRU(0.7) | 12.7698 | +39.19% | 10.9807 | +38.65% |
| SLRU(0.5) | 13.3454 | +45.47% | 11.5068 | +45.25% |
| LRU | 14.6806 | +60.04% | 12.8792 | +62.62% |

同配置 5 次重复的最大变异系数为 **0.073%**。最接近 LFU 的 SLRU(0.8)，成对 log-ratio 的描述性 95% t 区间分别为耗时 **+32.68%..+32.91%**、**+32.12%..+32.43%**。这些区间只描述固定 tape 条件下的进程重复噪声，不代表跨业务请求的置信区间。

原始重复值（秒，按 repeat 0..4）：

| 策略 | 代码 | Raft |
|---|---|---|
| LFU | 9.1860, 9.1771, 9.1710, 9.1713, 9.1711 | 7.9253, 7.9143, 7.9266, 7.9226, 7.9149 |
| SLRU(0.8) | 12.1868, 12.1830, 12.1901, 12.1814, 12.1810 | 10.4755, 10.4835, 10.4774, 10.4736, 10.4757 |
| SLRU(0.7) | 12.7749, 12.7679, 12.7764, 12.7685, 12.7698 | 10.9867, 10.9825, 10.9785, 10.9803, 10.9807 |
| SLRU(0.5) | 13.3440, 13.3519, 13.3453, 13.3508, 13.3454 | 11.4987, 11.5003, 11.5068, 11.5122, 11.5073 |
| LRU | 14.6933, 14.6806, 14.6913, 14.6770, 14.6804 | 12.8792, 12.8806, 12.8767, 12.8779, 12.8887 |

每个配置的 hit/miss、native copy bytes、all-hit calls 在五次重复中**完全相同**。10 个 workload×策略组合的 GPU 完整区间 hit/miss 均与 CPU 参考实现一致。每次区间均为 43,008 次 native 层调用，路由内容校验和及预算一致，没有以离群值为由删除任何正式计时样本。

## 为什么耗时差比 miss 差更大

在代码 tape 的完整计时区间，LFU miss 124,787，SLRU(0.8) miss 130,013（+4.19%），LRU miss 137,484（+10.17%）。不能直接按这些搬运量比例归因 +32.8% 和 +60.0% 的总耗时差，因此另采 planner/fill trace 检查。

独立的 LFU/LRU 短 trace 使用同一 128 sweep 预热和随后 8 sweep（各 384 层调用）：

| Kernel | LFU 平均 | LRU 平均 |
|---|---:|---:|
| decode_plan | **26.87 μs** | **132.37 μs** |
| mailbox_fill | 154.63 μs | 190.49 μs |

这段 trace 的 fill 有效搬运速率两者均约 44.1 GB/s，而 LRU planner 约慢 **4.93 倍**。源码也确认：LFU 专享 all-hit 快路径及并行 miss planner；LRU/SLRU 走通用策略路径，包含串行 active expert 更新以及 victim/protected 区扫描。故本轮差异同时来自**命中效果**和**现有 GPU 策略实现成本**。该短 trace 的分解不外推为完整区间或线上请求的精确占比。

完整 trace 说明及 `llm-torch-profiler-analysis` 的 kernel、overlap、fuse 三表见 [LIVE_PROFILE.md](LIVE_PROFILE.md)。两份原始 Chrome trace 均压缩存档，profile-only 数据未进入任何正式计时汇总。

## CPU 分层容量结果与边界

按前 512 sweep 训练、后 512 留出验证；128 sweep 预热。LFU 的结果：

| Workload | 近均匀容量 miss | 按训练 miss 选择容量后的留出 miss | 降幅 |
|---|---:|---:|---:|
| 代码 | 70,322 | 62,973 | 10.45% |
| Raft | 59,877 | 52,481 | 12.35% |

所有布局都使用 6390 slots。历史 CTA8 fill 成本目标与 miss 目标在这两个 LFU 样本上选出了相同布局；另存 Raft 的 adaptive CTA 成本敏感性结果。成本代理只计 fill，省略了本次实测明确存在的 planner 成本，也未计迁移、模型计算重叠或在线分布变化。

这些容量结果属于 CPU 留出估计，没有做该布局的 GPU A/B；**线上已经启用自适应分层容量，近均匀基准不等于当前线上状态**。不能将 10%–12% miss 降幅当成新增线上吞吐收益。两份合成短输入 tape 也不足以建立跨业务策略优势。

## 校验与复现

- 正确环境下既有 **334 项 CPU 回归及导入检查通过**，包内新增 **4 项 CPU 用例通过**。
- 两个 workload 的 5 种策略均通过实际 pack payload 逐字节检查，共 **78,336 次逐张量检查**。首个已开始的校验为 128 个计分 sweep，其余为 16；调整原因与时点记在 [LIVE_PROTOCOL.md](LIVE_PROTOCOL.md)，正式计时范围始终是 896 sweep×5 次。
- 录制加入显式 arm 文件门控，排除服务器建图时的合成 verify。所有已保存 sweep 均为完整的 48 层 `3×10` 路由。录制前修正了 tokenizer 返回类型和 Docker 自动删除容器的命名冲突；这些准备失败不属于正式计时样本。
- 恢复后 health=200、原启动脚本校验和一致；实际 scheduler 仍为 prepared=1 / CTA=adaptive / backend=b12x，无录制路径。ViT 启动时间一致。

`live_fp4_results/summary.json` 含完整重复值、每配置区间、CPU/GPU 计数核对；`SHA256.json` 同时记录压缩归档与原始数据的校验和。`coding-timing/`、`raft-timing/`、`coding-parity/`、`raft-parity/` 保存每个进程的原始报告与日志。路由及 profile 仅压缩，没有重采样。

在仓库根目录重新生成统计（不需要 torch/GPU）：

```bash
python -m tools.qwen38_cache_tests.cache_policy_experiments_20261018.summarize_live \
  tools/qwen38_cache_tests/cache_policy_experiments_20261018/live_fp4_results \
  --output /tmp/qwen38-cache-policy-recheck.json
```

GPU 重放命令、录制参数和明确的 scope 限制见 [README.md](README.md)、[LIVE_PROTOCOL.md](LIVE_PROTOCOL.md) 及归档的 `gpu.sh` / `record-*.sh`。本轮保留线上 LFU；若继续研究 LRU/SLRU，现有 native planner 的路径差异值得先处理，再用相同 tape 重测。
