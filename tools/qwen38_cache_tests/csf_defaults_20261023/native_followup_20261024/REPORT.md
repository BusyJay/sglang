# Native planner follow-up（2026-10-24）

上一轮 warp 路由 union 已独立提交为 `55b8ea31b5`。本轮进一步减少 native planner 的重复 lookup 和混合命中的同步开销。固定路由重放确认算子收益，但中文长 thinking 的结果仍在文本、MTP 接受率和缓存迁移波动内，**未证明新的稳定端到端加速**。

## 改动与正确性依据

1. **Native all-hit 直接使用 warp 的原始 miss 数。** 从 warp miss 统计到该判定之间，非 hybrid 路径没有修改路由或 lookup，故不需要由 thread 0 再顺序查一次。Hybrid 会筛走 CPU routes，保留它的原有扫描。
2. **LFU mixed-plan 的 hits 不再通过共享 current_slot 做 block rendezvous。** active union 已排序去重；本次调用的 active experts 不允许被淘汰，所以早先迭代不会改变后续 active expert 的 lookup。每线程可直接读该 lookup。miss 的 victim reduction 和安装后的 barrier 保留，先发布 keys，再允许下一轮扫描。只有 thread 0 依原顺序更新 clock、age、frequency 和 plan；victim scan 排除这些 active slots，因此不需要在每次 hit 更新后增加 barrier。

不改变 LFU/SLRU 规则、升序更新、浮点精度、MTP verify 批量或传输路径，没有新增 kernel / D2H / host 同步。沿用已有工作树内的其他改动；控制和候选的 CUDA 完整源码、SHA256 和实际运行命令均存于本目录，原有 CSF bank 修复未混入这次提交。

## 固定状态重放

`check.py` 从相同状态分别运行上一轮和本轮 CUDA 模块；432 组 remap 与全部缓存策略状态逐位相同。覆盖 32/147 slots、LFU/SLRU、随机连续序列、重复/掩码/32-route 边界。

下表为 µs/次，**包括完全相同的状态恢复成本**，不是纯 planner kernel。每次恢复 miss 状态，避免 replay 将 miss 预热成 hit。每例 9 个 event 样本，中位数：

| 场景 | 上一轮 warp | 本轮 |
|---|---:|---:|
| 10 条全命中 | 9.255 | 8.961 |
| 30 条全命中 | 9.986 | 8.963 |
| 30 route / 混合命中 | 13.573 | 13.058 |
| 全缺失 | 73.250 | 72.673 |
| 重复路由 | 11.781 | 11.777 |
| 全 masked | 8.749 | 8.752 |

全 masked 和单路由基本持平。单独只做 miss-count 复用的结果保存在 `count-reuse-only.json`，组合改动的完整样本在 `check.json`。

## 自然长 thinking：完整请求

沿用上一轮两条中文自然提示、thinking chat template、2048 输出上限、允许 EOS；每条三次，全部达到 2048。使用 native、W4A16 B12X、两步 NEXTN、FP8 target/BF16 draft KV、7489 初始 slots、并发 1、observe=0。纯文本隔离实例关闭 HiCache，原因和边界与上一轮报告一致；生产脚本未改、生产服务未重启。

持续生成速率去掉前 128 输出 token，单位 token/s：

| 场景 | 上一轮中位数 | 本轮中位数 | 本轮样本 |
|---|---:|---:|---|
| 推理服务方案分析 | 104.27 | 102.18 | 102.18 / 101.43 / 109.08 |
| 带约束排程推理 | 111.14 | 112.25 | 111.47 / 112.25 / 118.96 |

分别约 -2.00% 和 +0.99%，未显示一致的整模型收益。控制数据取上一轮已完成的请求，不是本轮随机交错 A/B；MTP 接受率、文本和自适应缓存迁移不同。仅能将其作为完整服务可执行与测量边界的证据，不用它声称稳定加速或回退。

## Kernel table：长 thinking steady trace

最后另发一个 2048-token 请求，在已生成至少 512 token 后采集 25 forward steps；裁剪后保留 23 完整轮 / 1104 个 target layer planner。GPU family 累计时间除以 target planner 数，单位 µs；不代表每个 family 只属于 target 层。

| Family | 上一轮 | 本轮 |
|---|---:|---:|
| Planner | 9.87 | 9.51 |
| 权重 fill | 50.52 | 59.66 |
| MXFP8 GEMM | 79.78 | 80.33 |
| Routed B12X | 63.69 | 63.10 |
| HC mix | 28.66 | 28.71 |
| BF16 GEMV | 35.35 | 35.34 |

Planner 约减少 3.6%；固定状态重放提供了不依赖实际路由变化的补充证据。fill 在两窗口间变化明显，不能把整个窗口耗时变化归因于这次代码。

## Overlap-opportunity table

| 观察 | 证据 | 含义 |
|---|---|---|
| Shared/fill overlap 仍存在 | 本轮 14.7% fill 区间与其他 GPU 活动交叠 | 不重复添加现有 shared overlap |
| 当前窗口的 fill 独占活动仍存在 | 56.17 ms / 372.53 ms span，约 15.1% | 改善真实 miss 轨迹和填充暴露时间仍有价值，不能视作可直接回收的加速上限 |
| GPU 并集空隙约 4.3% | CPU/GPU steady trace | 不是 SM 利用率，也不能从同步 API 累计时长推导可节省 wall time |

## Fuse-pattern table

| 项目 | 结论 |
|---|---|
| Planner 重复工作 | 同一 kernel 内重用已有 miss 归约，没有增加 launch |
| Native hit/miss 拆分 | 本轮没有开启旧 early-cache 路径；MTP verify 需要单独原生实现 |
| HC mix / BF16 / MXFP8 | 本轮没有改动或降低精度，仍是后续独立分析目标 |

本轮为单 trace 加本地源码交叉核对，未进行 graph-off / formal 两 trace 的 Python attribution。

## 验证

- GPU 回归：24 项、55 子测试通过。新增 sorted union 中 resident/missing 交错的路由；策略状态参考扩展到 parallel-miss modes 0/1/2 与 LFU/SLRU。
- Racecheck：上述参考测试 12 个子测试通过，0 errors、0 warnings；保留 reduction 和 miss-install 同步点。
- Memcheck：上述参考测试 12 个子测试通过，0 errors；保留内存检查，沿用本机 `--report-api-errors no` 跳过 import-time API probes。
- Black、改动区间 clang-format 和 `git diff --check` 通过。没有放宽数值容差。

主要剩余空间仍是实际长推理的 fill 暴露时间、原生 verify hit/miss 解耦、常规线性层和 HC mix。本轮局部节省较小，不能据此承诺新的端到端比例。
