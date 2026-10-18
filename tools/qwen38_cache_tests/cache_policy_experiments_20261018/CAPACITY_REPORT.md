# 容量分配算法：真实 GPU 迁移与留出段实验

已补齐上一轮遗漏的容量 GPU 对照。**当前现网分配器在两份路由上均最快；继续使用现网 LFU + adaptive allocation。** 计入一次真实迁移暂停后，相对均匀分配，coding / Raft 的搬运回放耗时分别下降 **7.23% / 7.71%**。两个 patch 默认布局方案均未超过现网分配器。

本次补充的是各层容量分配，淘汰策略统一为 LFU。测量范围是实际 planner、权重搬运、事件依赖和迁移；attention、MoE、draft 计算未执行，结果不能换算为线上 tokens/s。

## 实验设计

- 使用上一轮相同的 coding、Raft 路由，各 1024 个完整 verify sweep；每轮 48 层，原始 3×10 路由及重复 ID 保留，MTP 与路由序列固定。
- 每个配置始终使用 6390 个 expert slots。所有配置启用相同的 adaptive pool/metadata，先从空 cache 用均匀布局（前 6 层 134、其余 133）回放相同前 512 轮。校验了前缀 hit/miss/copy bytes 完全相同。
- 仅用前 512 轮选择布局，训练评分排除最初 128 轮。随后调用真实 `_maybe_rebalance`，保留按 frequency/age 选中的 resident，实际搬移 pool、恢复 metadata、刷新 mailbox。最后计时相同的后 512 轮，保留迁移后的补 cache 和首次调用成本。
- 主比较使用“迁移墙钟 + 后 512 轮完成墙钟”。布局分析提前在 CPU 执行，所有组关闭异步采样和再次分配；分析 CPU 成本及其在线完成时间不在这项 GPU 比较内。
- 每份路由每算法 5 次独立进程，轮内随机顺序，种子 20261019；共 40 次正式计时。所有样本保留，无离群剔除。另以 8 个独立进程在迁移后校验 16 轮全部 active expert 的全部 6 个 tensor，共 36,864 个层/tensor 检查，全部通过。
- 当前 swallow 镜像、RTX 5090 D、真实 FP4 pack，fill CTA=adaptive、prepared launch=1、shared THP 权重源。全部环境变量在独立 `gpu.sh` 中；生产启动脚本未编辑。完整预定协议见 [CAPACITY_PROTOCOL.md](CAPACITY_PROTOCOL.md)。

## GPU 结果

秒数是五次的中位数；相对耗时变化是同轮成对比值的几何均值，负值表示省时。各分量分别取中位数，总时间中位数可能与两列之和有细微差异。

### coding

| 分配算法 | 后 512 轮 / s | 迁移 / s | 含迁移 / s | 相对均匀 | 相对现网 | 实际 miss |
|---|---:|---:|---:|---:|---:|---:|
| 均匀分配 | 5.1844 | 0.0000 | 5.1844 | +0.00% | +7.80% | 70,322 |
| 现网分配器 | 4.6546 | 0.1556 | 4.8102 | -7.23% | +0.00% | 62,417 |
| patch：最少 miss | 4.6959 | 0.1572 | 4.8531 | -6.40% | +0.90% | 63,142 |
| patch：最小 fill 成本 | 4.6968 | 0.1579 | 4.8561 | -6.33% | +0.97% | 63,142 |

| 分配算法 | 迁移保留 / 淘汰 expert | 搬运 GB（十进制） | 含迁移耗时 CV |
|---|---:|---:|---:|
| 均匀分配 | 0 / 0 | 194.426 | 0.0691% |
| 现网分配器 | 5,746 / 644 | 172.571 | 0.0292% |
| patch：最少 miss | 5,805 / 585 | 174.575 | 0.0596% |
| patch：最小 fill 成本 | 5,805 / 585 | 174.575 | 0.0837% |

| patch 方案 | 相对现网耗时的描述性 95% 区间 |
|---|---:|
| patch：最少 miss | +0.803% .. +0.998% |
| patch：最小 fill 成本 | +0.848% .. +1.096% |

### raft

| 分配算法 | 后 512 轮 / s | 迁移 / s | 含迁移 / s | 相对均匀 | 相对现网 | 实际 miss |
|---|---:|---:|---:|---:|---:|---:|
| 均匀分配 | 4.5091 | 0.0000 | 4.5091 | +0.00% | +8.36% | 59,877 |
| 现网分配器 | 4.0085 | 0.1555 | 4.1644 | -7.71% | +0.00% | 52,309 |
| patch：最少 miss | 4.0333 | 0.1568 | 4.1900 | -7.13% | +0.63% | 52,743 |
| patch：最小 fill 成本 | 4.0372 | 0.1565 | 4.1941 | -7.00% | +0.77% | 52,812 |

| 分配算法 | 迁移保留 / 淘汰 expert | 搬运 GB（十进制） | 含迁移耗时 CV |
|---|---:|---:|---:|
| 均匀分配 | 0 / 0 | 165.548 | 0.1273% |
| 现网分配器 | 5,670 / 720 | 144.624 | 0.0326% |
| patch：最少 miss | 5,745 / 645 | 145.824 | 0.0227% |
| patch：最小 fill 成本 | 5,745 / 645 | 146.015 | 0.1279% |

| patch 方案 | 相对现网耗时的描述性 95% 区间 |
|---|---:|
| patch：最少 miss | +0.578% .. +0.673% |
| patch：最小 fill 成本 | +0.623% .. +0.922% |

所有配置五次重复的 hit/miss 和搬运字节完全相同，全部 24,576 次留出段 layer 调用都完成。每次 native copy bytes 都严格等于 miss × 2,764,800；均匀布局完整 GPU miss 与 CPU 参考一致。最大 CV 为 0.128%，上述区间仅描述这两份固定 tape 的重复噪声，不能作为跨业务分布置信区间。

原始五次值（含迁移秒数，repeat 0..4）：

| 工作负载 | 算法 | 五次值 / s |
|---|---|---|
| coding | 均匀分配 | 5.191189, 5.181702, 5.184020, 5.186636, 5.184406 |
| coding | 现网分配器 | 4.812486, 4.810232, 4.811047, 4.808739, 4.809820 |
| coding | patch：最少 miss | 4.853572, 4.851415, 4.853115, 4.858728, 4.852085 |
| coding | patch：最小 fill 成本 | 4.855534, 4.856105, 4.859015, 4.863145, 4.852348 |
| raft | 均匀分配 | 4.509111, 4.517992, 4.506461, 4.507699, 4.518216 |
| raft | 现网分配器 | 4.164843, 4.163061, 4.164393, 4.162057, 4.165363 |
| raft | patch：最少 miss | 4.190722, 4.191139, 4.188854, 4.189305, 4.190014 |
| raft | patch：最小 fill 成本 | 4.205691, 4.194130, 4.194225, 4.193267, 4.193315 |

## 原因与选择

现网分配器相对均匀分配将实际 miss 从 70,322 降至 62,417（coding，−11.24%）、从 59,877 降至 52,309（Raft，−12.64%），分别少搬运约 21.85 / 20.92 GB。一次约 155 毫秒的迁移开销已经包含在主结果中。按本段平均节省速度估算，约 150 / 159 sweep 可摊平迁移暂停；这是条件平均估计，没有逐轮计时来确定精确的累计盈亏平衡点。

**patch 默认候选网格限制了布局选择。** 现网候选为 `[80,96,112,128,133,134,144,160,192,224,256]`；patch 默认是 `[96,112,128,133,134,144,160,192,256]`，缺少 80 和 224。另做的 CPU 同网格诊断显示：把 patch 的 min-misses DP 改为现网候选后，两份路由都选出与现网 **完全相同** 的布局。因此默认方案相对现网的差异包含候选网格影响，不能归因于 DP 求解本身。诊断不改变主 GPU 组的布局。

fill 成本目标使用当前 `mapped_adaptive` 的实测 median GPU ms，而不是上一轮 CTA=8 标定。默认网格下，coding 的 min-fill 与 min-misses 选出完全相同布局，GPU miss 也完全一致；它们的小幅计时差属于重复噪声。Raft 的 min-fill 实际 miss 52,812，比 min-misses 多 69 次，平均搬运/规划耗时也略高；训练集代理目标的优势没有转化为本次留出段收益。

CPU 同网格诊断中，min-fill 仍选出不同布局；静态留出 miss 分别为 62,260 / 51,929，现网静态布局为 62,256 / 52,023。这些同网格 min-fill 布局没有进入本次 GPU 主比较，不能据此排除其他网格下的小收益。本次选择结论限定为两个 patch **默认方案**。

迁移后的 cache 状态与从空按目标容量训练的 cache 状态不同：现网 coding / Raft 静态 CPU 估算 miss 为 62,256 / 52,023，真实迁移后是 62,417 / 52,309。上表已包含这种 resident 历史和迁移后补 cache 成本。

现网继续保持 LFU、`ADAPTIVE_CACHE=true`。本实验没有建立新的端到端吞吐结论；在线分析的 CPU 用时、观察器开销、应用布局的延迟及模型计算重叠需单独做 serving 实验。本次 production CPU 分析约 10.24 / 9.24 秒，提前计算布局避免其完成时间干扰 GPU 算法比较。

## 归档与复现

- [live_capacity_results/summary.json](live_capacity_results/summary.json)：全部重复值、成对比值、区间、计数及最终布局。
- 同目录保存 48 个 GPU worker 的 JSON/log、CPU request/reply、训练布局、同网格诊断、硬件、启动脚本、源码 hash、恢复证据和 `SHA256.json`。原始压缩路由沿用相邻 [live_fp4_results/](live_fp4_results/)，归档时校验解压内容 SHA256 与 GPU tape hash 一致。
- [capacity_experiment.py](capacity_experiment.py)：准备布局、停服保护、随机 GPU 矩阵和恢复。固定路径为本机生产目录，其他机器需调整挂载。
- [capacity_grid_diagnostic.py](capacity_grid_diagnostic.py)：仅做 CPU 同候选网格诊断。
- [summarize_capacity.py](summarize_capacity.py)：CPU 复算汇总和验证归档。

归档的 `capacity_experiment.timed.py` 保存实际计时使用的编排脚本；提交版随后仅删除一个未使用的局部变量。`source-provenance.json` 区分该清理与计时版本，GPU worker 和运行时的源码 hash 保持一致。CPU request 使用 gzip 保存。

从仓库根目录复算已提交结果：

```bash
python -m tools.qwen38_cache_tests.cache_policy_experiments_20261018.summarize_capacity \
  tools/qwen38_cache_tests/cache_policy_experiments_20261018/live_capacity_results
```

重跑 GPU 前先以新的输出目录执行 `capacity_experiment --output DIR` 准备布局，再执行 `capacity_experiment --run --rounds 5 --output DIR`。后一个命令会暂停语言服务；`finally` 自动恢复并检查 health、生产脚本 hash 与 ViT start time。

已完成语法检查、Ruff 格式和静态检查。服务恢复状态及实际 scheduler 环境见归档中的 `restored.json`、`post-restore-health.json`、`restored-scheduler-env.json`。
