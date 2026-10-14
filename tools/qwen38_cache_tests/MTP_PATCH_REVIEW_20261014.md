# MTP patch 包评估（2026-10-14）

包：`qwen38-priority-20261014-v2.tar.gz`。基线：
`9aac562231f53e5bbe0ed5662805338b6ab653e5`。
在原分支 `qwen3.8-flash-next-offload` 按 `series` 顺序应用；11 个 patch
的 SHA256 均与 manifest 一致，每项先做 `git apply --check`。

## 分阶段结果

| 阶段 | CPU suite | 结论 |
| --- | ---: | --- |
| 原始基线 | 193 | 测试、真实模块导入及 ModelOpt runner wiring 通过 |
| 0000–0005：计时、统计、辅助缓冲和 short-conv | 205 | 通过；额外修正容器 fixture 的 runtime 字段 |
| 0006–0007：总槽位和 adaptive 分配 | 212 | 通过；容量边界和预算完整性检查保留 |
| 0008：MTP eager/fixed | 222 | 通过 |
| 0009：MTP graph/fixed | 224 | 通过 |
| 0010：MTP graph/adaptive | 227 | 通过 |
| 后续回归及真实配置投影 | 228 | 通过；包括 steps=1/2/3 的运行配置投影 |

容器 CPU 检查使用本地 `lmsysorg/sglang:v0.5.21`，禁用网络且不挂载 GPU。
真实 MTP loader 和模型模块导入也纳入了容器检查。

4080 SUPER：30 项 GPU 测试通过，SM120 专用的两个测试类按架构跳过。
5090 D：37 项 GPU 测试通过，零跳过，包括实际 MTP 专家形状的 block-FP8
对照、NVFP4 CUTLASS 对照、verify 宽度 2/3/4、缓存迁移与 graph 回放。
GPU 检查使用本地 `swallow-sglang:v0.5.21-qwen38-capacity`。

## 审查中保留的修正

- SwiGLU 提前绑定 runtime 后，原容器 fixture 缺少 `manifest` 和
  `slots_for_layer`；已补齐，继续验证真实 runner payload。
- 初次 GPU 测试暴露了 graph capture 与后台事件同步之间的竞态。
  threaded runtime 的显式 `synchronize()` 现在先排空 job/retirement，
  实际 decode graph warmup 后也调用它。保留原 attention warmup hook。
  修正后迁移与 verify graph 回放测试通过，并增加 CPU 回归测试。
- canary 增加 `--mamba-ssm-dtype`，让 target-only 和 MTP 显式使用相同
  BF16 GDN 状态，避免把精度差异混入对照。
- 用真实 `ServerArgs` 到运行配置 bag 的投影检查步骤数、draft KV 和
  GDN 默认值；不扩展启动 hook 无需使用的模型 override 白名单。

## 真实 checkpoint 验收

使用本地缓存 revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` 和已有
`results/packs/assembled`，不下载权重。测试只启动和清理自己创建的 canary
容器。MTP 保持显式开启，常规 target-only 默认不启用 MTP。

共同设置：threaded、单请求、BF16 GDN、target FP8 KV、MTP BF16 KV、
context limit 4096、输入 64/512 token、chunk 256、最多输出 32 token，
另加 `17 × 19 = 323` 算术探针。512-token 请求复用了首个请求的 64-token
prefix，并跨越 prefill chunk 边界。

原 5040 槽 target-only eager 对照通过；5040 槽 MTP 在
`mem-fraction-static=0.90` 下完成了 draft 的真实加载，但没有足够预算
分配 KV/工作区。随后按包内建议，以 3840 总槽验证 MTP 和同容量 target。
draft 日志中的新增权重约为 2.52 GiB，专家仍为 block-FP8。

最初 `max-mamba-cache-size=4` 太小，按状态比例无法容纳一个请求，已改为
15。graph/adaptive 首轮请求触发 120 秒超时；300 秒请求时限重跑通过。
这些失败记录与成功记录一并保留，不将配置失败算作 patch 通过证据。

验收明细和完整命令位于
`results/mtp-patch-review-20261014/canary-summary.json`、各阶段 `report.json`
以及 `server.log`。

| 真实模型配置 | 结果 | 64-token 请求的平均 verify 接受长度 |
| --- | --- | ---: |
| target-only eager，5040 槽 | 通过 | — |
| MTP eager/fixed，3840 槽，steps=1 | 通过 | 1.94 |
| MTP graph/fixed，3840 槽，steps=1 | 通过 | 1.94 |
| MTP graph/adaptive，3840 槽，steps=1 | 通过 | 1.94 |
| MTP graph/adaptive，3840 槽，steps=2 | 通过 | 2.58 |
| MTP graph/adaptive，3840 槽，steps=3 | 通过 | 3.10 |
| target-only eager，3840 槽 | 通过 | — |
| target-only graph/adaptive，3840 槽 | 通过 | — |

每轮 verify 接受长度按 `1 + correct_drafts / verify_rounds` 推导；保留原始
接口指标，但短请求的接口 `spec_accept_length` 还包含首次 prefill 输出，
不能直接解读为严格的每轮 verify 宽度。

steps=1 的 fixed eager 与 fixed graph 在两个输入上的 output IDs 完全一致。
adaptive 的 64-token 输出与 fixed graph 一致；512-token 输出出现末尾空格
差异。八轮 canary 的算术探针都匹配 `323`。不将短请求的可见正确答案当作
所有模式严格 token 等价的证明。

graph/adaptive 的 steps=1/2/3 两秒 NVML 采样峰值分别为
30540 / 30680 / 30832 MiB。steps=1 实际产生两条 adaptive 分析记录，
steps=2 一条，steps=3 此短请求集未集齐分析窗口；均未达到实际迁移门槛。
`report.json` 的 cache stats 为最近一次周期日志快照，可能早于最后一次
分析或请求结束；完整分析次数以 `server.log` 为准。

## 证据范围

短请求覆盖加载、生成、连续 verify、prefix/chunk、长度停止与 EOS。
steps=2 的实际接受直方图覆盖拒绝、部分接受和完整接受。强制迁移后的
graph 回放由小型 GPU 测试覆盖；真实模型的 adaptive 分析按收益门槛决策，
不能把启用 adaptive 等同于发生迁移。

首次请求包含 JIT、首次专家搬运和 prefill；不同模式的 KV 总分配也会随
预算变化。记录的短请求时延不能作为稳定 decode 加速结论。NVML 显存峰值
为两秒采样值，不是 CUDA allocator 的 allocated/reserved 精确峰值。

本次未证明长上下文边界、取消、全部 stop 模式、所有状态组合的严格数值
等价，也未进行原最佳部署配置的正式性能与碎片 campaign。
MTP 作为默认关闭的实验能力保留。
