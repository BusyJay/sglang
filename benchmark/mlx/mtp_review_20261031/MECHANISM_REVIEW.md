> 原 patch 随附的历史机制审查；最终 M5 修正、测量与默认值见 [README.md](README.md)。

# 实现机制复查 — 2026-10-31

本次按根目录 AGENTS.md 的新原则复查工作区尚未提交的修改。
基准为 `a7a39c8b3d`。本报告是机制审查，不是新的 M5 全模型吞吐结果。

| 项目 | 实际实现 / 限制来源 | 审查结论与处理 |
| --- | --- | --- |
| `serial/head/ffn` 验证模式 | serial 保留逐行参考，head 分离输出头成本，ffn 启用批量 FFN | 是诊断与数值验证接口；不能把三种模式的存在当作完整高效 MTP 已完成。目标机完成误差、生成与吞吐验证后选择默认路径，淘汰无价值候选。 |
| 多行 router / down-combine / FFN mHC | Metal grid 添加 token 维度，数组读写增加每行偏移；1–17 行真实设备测试通过 | 已改进底层 dispatch，不是绕过融合。当前 down 每行仍独立读取专家权重，尚未实现跨行同专家数据共享。 |
| `batch_fusion` | 原候选在一次调用期间写入 MLP/MoE 实例，再在 finally 恢复 | 已修正为显式 keyword 参数并传入 shared FFN，删除临时模块状态。普通调用语义与验证调用语义由参数表达。 |
| attention 逐行验证 | KDA/MLA 的状态在每行执行后形成快照；当前 verifier 未批量投影 | 软件实现缺口。没有证据表明 Metal 禁止批量投影或逐位置快照。完整优化需要批量无状态投影并由 recurrence/索引更新产生可提交的每位置状态。 |
| KDA `1..8` 行 guard / 仅末尾 state | 当前 kernel 对 token 数分配 threadgroup 中间数据，只输出最终 recurrent/conv state | 当前 kernel 合约，不是平台极限。扩展为分块执行或输出每位置 state，需要真实线程/内存资源测量及拒绝回滚验证。不能简单扩大常量或丢掉中间状态。 |
| MoE bits/group/dimension guard | 现有量化 unpack 与 SIMD 行 tile 覆盖 4/5/6/8bit、指定分组、对齐维度 | 实现覆盖范围。checkpoint 当前形状在覆盖内；新增形状应补尾部 mask、unpack 或其他 dispatch，而不是把报错写成 Metal 限制。 |
| 17 行 / 单序列 | MTP 控制器最多 16 个 drafts 加 root；本 kernel 扩展按该合约验证 | 当前服务预算，不是 Metal 最大 batch。跨请求 batching、prefill chunking 和更大验证窗口应分别建立调度/状态合约，不应把 17 当作通用设备上限。 |
| KDA 六路不兼容时回到三路 | 不同投影的 quantization spec 不能直接拼成同一个 QMM；FP32 宽投影还有既有目标机回归证据 | 先保留数值/性能证据；后续可以分组投影或异构融合 kernel。不能由拼接接口不兼容推出融合不可做。 |
| MLA KV-B cache 开关 | 已实现权重身份/spec/dtype 校验和失效后重建，减少重复 dequantize | 是实际复用机制；新增常驻内存且 layout 可改变舍入。小模型没有证明显著收益，须目标机测成本后决定启用或删除。还需评估直接量化 latent 运算以免长期常驻展开。 |
| DOT 与 Metal capture | 模型 async eval 已物化大部分 DOT 图；Metal capture 成功 | 计算图可见性不足不代表算子不能融合。用设备 trace、调用形状与缓存行为补证据，不能仅靠 DOT 节点数宣称消除了全部转换。 |

## 本次实际修正

- 在根目录 `AGENTS.md` 记录软件限制可改、优先机制优化、验证与安全要求和提交规范。
- 删除临时 `batch_fusion` 模块标志，改为显式调用参数；shared expert 同步接收参数。
- 保留 A/B 模式，原因是已实测的数值差异及未完成的目标机验证；不将其定义成模型
  功能或 Metal 能力边界。
- 将 attention 投影批量化、KDA 中间状态、跨行专家权重复用明确标记为未完成机制，
  不宣称本包已经完成全层高效批量 MTP。
- 重新运行最终源码测试并更新便携 patch 包与 manifest。

## 提交边界

建议按三项独立机制组织提交：`[Docs] Record MLX optimization and commit policy`、
`[Perf] Batch GLM MLX verification FFN kernels`、
`[Perf] Reuse GLM MLA latent projection weights`。后两项正文必须写明默认开关、数值变化、
本机验证与 M5 尚未测量；保留未提交状态直到执行实际提交任务。分包/提交不能混入
历史 `dtype_audit_20261030` 的克隆、权重、环境或 capture。
