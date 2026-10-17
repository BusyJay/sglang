# 同层 verify 跨 token 复用与 reuse 慢因（2026-10-18）

结论：真实请求有跨 token 共享，但当前 kernel 的批次组织把节省的权重解码工作
换成了更多 token-row 计算。问题不只是重合率低。

## 实测路由

在一个隔离的 SM120 worker 上使用现有默认配置，分别采集 CPU cache 短输入、
中文 Raft、两次固定 27323-token 代码审查输入，各生成 1024 tokens。
采集在 target 路由 remap 前执行，仅记录 verify 的逐 token expert ID，带请求边界。
采集使用只挂载到测试容器的源码 overlay，没有修改生产 runtime。
采集的 CPU 同步会影响延迟，因此**不使用这个 worker 的吞吐判断收益**。

共 81984 个 layer-round，即每次 layer 的 verify 调用算一个样本，
全部为 3-token × top-10，共 2459520 条有效路由。48 层、4 个请求，
这些是相关样本，不是 81984 个独立请求。包含所有候选位置，而非仅接受位置。

- 至少有一个专家跨 token 重复的比例：98.19%。
- 真正可合并的重复路由比例：**27.69%**，
  定义为 `1 - unique_experts / sum(per_token_unique_experts)`。
- 每轮 30 条路由平均涉及 **21.693 个不同专家**。
- 不同专家中，**70.77% 只被一个 token 选择**；
  两个/三个 token 共享的分别为 20.17% / 9.06%。
- 没有同一 token 内重复 expert，没有通过非法 ID 增大复用率。
- 各层平均复用比例约 5.76%–42.72%，逐层数据见 route-stats.json。

| 输入 | layer-round 数 | 跨 token 复用比例 | 当前 reuse 的 token-row 工作量模型 |
| --- | ---: | ---: | ---: |
| long-0 | 19296 | 25.59% | 2.977x |
| long-1 | 21120 | 26.84% | 2.926x |
| raft | 21216 | 28.76% | 2.850x |
| short | 20352 | 29.45% | 2.822x |

## 为什么计算量反而增加

`_cache_project_reuse` 按唯一专家选 owner，但 `BT=next_power_of_2(TOKENS)`。
3-token 时 BT=4。每个 owner 都创建 `[BT, BN]` 累加器和 `[BT, BN, BK]`
乘法/归约，包括不选择此专家的 token，以及第 4 个补齐行。
`present` 主要屏蔽 load，未按实际命中 token 数将这些点积拆成独立路径。

原路径每轮计算 30 个 token-expert 行；当前 reuse 按形状构造约
`21.693 × 4 = 86.772` 行，**工作量模型为原路径 2.892 倍**。
这来自源码张量形状，不是 Nsight 测出的 FLOP 硬件计数。
即使三个 token 完全选中相同 10 个专家，仍是 40 行对 30 行，多约 33%。

此外，原 miss planner 已对一轮的专家集合去重后填充 cache。
所以当前 reuse **不会再节省一次 host→GPU expert fill**；它试图节省的是
GPU projection 内的重复 FP4 读取/解码。原路径重复读取还可能命中 L2，
本轮没有测 DRAM/L2 硬件计数，因此不把解码复用比例称为 DRAM 带宽收益。

## 控制重合率和消融实测

固定 SM120、32 个合成驻留专家、hidden=2560、width=640、topk=10，
每 token 内专家不重复。shared 表示所有 token 共有的专家数，其余彼此不重合。
CUDA Graph 每次包含 32 次调用，5 次 warmup replay、20 个计时样本，单位微秒。
所有 72 组均检查 eager/graph 结果与既有 fused 一致（atol=0.004, rtol=0.02）。
两个 token 的完整结果也保存在 controlled-bench.json。

| 共有专家数 | 合并比例 | fused | reuse | 去掉清零 | 向量 scatter | 两者都改 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0% | 117.26 | 383.85 | 382.35 | 375.41 | 374.29 |
| 2 | 13.3% | 117.10 | 337.59 | 336.13 | 332.03 | 330.66 |
| 4 | 26.7% | 118.41 | 291.27 | 289.86 | 286.63 | 285.11 |
| 6 | 40.0% | 118.49 | 244.90 | 243.55 | 241.82 | 240.46 |
| 8 | 53.3% | 118.43 | 200.96 | 199.55 | 197.85 | 196.42 |
| 10 | 66.7% | 118.43 | 182.39 | 180.31 | 179.27 | 177.54 |

贴近真实的 26.7% 合并比例下：fused 118.41µs，reuse 291.27µs。
去掉清零仅节省约 1.4µs，向量 scatter 约 4.6µs，均远小于约 173µs 的差距。
三个 token 完全重合时仍为 118.43µs 对 182.39µs；同时改两者仍为 177.54µs。
因此这两项不是主因，仅改调度/清零不能解决多算 token-row 的问题。

编译器报告中，3-token gate/up 的寄存器数为 fused=115、reuse=128，
shared memory 为 512 对 8192 bytes；down 的寄存器数均为 96。
原 reuse 没有 register spills。不能将慢因直接归结为 spilling，
也没有仅凭资源数推断实际 occupancy。详细资源记录在 controlled-bench.json。

## 下一步实现方向

1. 对只命中一个 token 的专家保留现有 fused 路径，它们占唯一专家约 71%。
2. 对真正共享的专家按实际命中 token 数分组，只计算需要的 1/2/3 行。
3. 解决额外行计算后，再简化 owner 查找、路由扫描和 scatter；用真实路由分布
   重放判断是否值得分派。当前原始 reuse 在控制重合率测试中均未快于 fused。

未改任何生产默认值，reuse 仍关闭。`reuse_no_zero` 仅适用于本次全部路由有效的
诊断输入，遇非法路由不安全，不可直接用于生产。向量 scatter 是诊断消融实现，
不是主张它是最优实现。初次 gather 写法在已安装 Triton 的 lowering 中失败，
保留 initial-gather-compile-failure.log.gz；最终使用 select/reduce 实现并通过全部检查。

## 复现

```bash
python3 tools/qwen38_cache_tests/verify_experiments_20261018/profile_verify_routes.py --results /tmp/verify-route-profile --tokens 1024
python3 tools/qwen38_cache_tests/verify_experiments_20261018/analyze_verify_reuse.py /tmp/verify-route-profile/verify-routes.jsonl --output /tmp/reuse-stats.json
# 在同一 GPU 镜像中，确保 GPU 空闲：
PYTHONPATH=python python3 tools/qwen38_cache_tests/verify_experiments_20261018/bench_verify_reuse.py --output /tmp/reuse-bench.json
```

- route-stats.json：总体、逐输入、逐层的重复比例和 multiplicity/unique histogram。
- service/verify-routes.jsonl.gz：完整逐 token trace（含 warmup/请求边界）。
- service/overlay/、service/provenance.json、command.json：测试 overlay 及实际挂载命令。
- controlled-bench.json、ablations/：全部控制实验与生成的诊断源码。
- 3 项统计单元测试覆盖完全共享、完全不共享、非法和同 token 内重复路由。
