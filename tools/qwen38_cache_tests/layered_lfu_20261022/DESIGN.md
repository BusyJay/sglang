# 分层 LFU admission 与有界 CPU 分流

本轮按用户确认的定义实现：miss 比本层可淘汰 GPU 专家还冷时，暂不值得换入，才成为 CPU 候选。只比较本层 victim，不引入全局 LFU，不依赖成本校准表。

## 路由规则

1. 已驻留的 hit 一律在 GPU；本次原始路由中的驻留专家受到保护，不可充当 victim。
2. 有空闲 slot 时，优先给最热的 miss 使用；所有 miss 都能放入空位时，全部搬入 GPU。
3. 对其余 miss，取本层非 active GPU 专家中最低的 LFU 热度作为保守 admission 门槛。只有严格低于该门槛的 miss 才成为 CPU 候选；相等时搬入 GPU。
4. 候选数 `C <= layered_cpu_max_experts` 时全部留 CPU；`C > n` 时整个候选集合都搬入 GPU，避免 CPU 承担过长尾延迟。这个 n 限制的是冷候选数，不是该调用所有 miss 的数量。
5. 高热 miss 与非 CPU 候选仍走原 GPU 填充和局部 LFU。候选只是本次不搬入；CPU 专家不占 slot，不提前驱逐 GPU 专家，不后台填充。
6. 保留 B12X 至少一个安全 GPU route、非法路由与容量检查。GPU-only 的 slot 置换仍由原逐层 LFU 执行。

举例：n=1，本次两个 miss 的更新后热度为 1.2、5，本层可淘汰 victim 最低热度为 2，则 1.2 的专家留 CPU，5 的专家搬到 GPU。如果两个 miss 都低于 2，冷候选数 2>1，则两者都搬到 GPU。

门槛取最冷的可淘汰专家，不模拟整批 miss 连续置换后的第 2/3 个 victim。因此是保守 admission：可能将某些本可拒绝的 miss 仍搬入 GPU。这个选择将决策限制为一次 slot 扫描，并保证 CPU 候选低于所有可淘汰 GPU 专家。

## CPU 也参与 LFU

另存 `[layers, experts]` 的 float32 热度，48×512 共 98304 字节，独立于 slot 生命周期。每次有效 decode/verify 调用：

- 用当前 GPU slot 的频次同步 resident 专家，以兼容 prefill 和容量重排后的状态。
- 所有专家热度乘以 0.95；原始路由 union 中每个专家加 1，包含 CPU 专家。GPU 频次原有单位也是每次调用的 unique expert，而不是重复 token route 次数。
- 用更新后的热度比较 admission。CPU 专家频次上涨后不再低于 victim，即正常换入 GPU；迁入 slot 的频次从专家热度恢复，避免重置为 1 导致再次冷落。
- CPU 路由从 GPU fill/LFU 计划过滤，但不从专家热度中抹去。GPU hit、CPU miss、超限后 GPU miss 的记账一致。

CUDA 回归固定初始 victim 频次 3、4、5，重复访问同一 CPU 专家：前两次在 CPU，第三次热度达到 2.8525，超过衰减后的最低 victim 2.572125，迁入 GPU；下一次为 GPU hit，频次保留为累计值。

非驻留热度跨 slot reset/resize 保留。prefill 仍走已有 GPU 路径，resident 频次在下次 decode 同步；本轮不改变 prefill 的计数单位。

## 接口

```json
{
  "cpu_hybrid_mode": "layered_lfu",
  "layered_cpu_max_experts": 1
}
```

其余前提与分层共算一致：threaded、adaptive LFU、W4A16/B12X、device mailbox、禁用 early-cache。n 为 [0,30] 的整数，默认 1；n=0 用于只保留专家热度的 GPU 对照，不创建空转 CPU worker。旧 calibrated/probe/zero 模式兼容，默认模式不因本轮自动开启。

第一次实现的串行热度循环和 slot×route 扫描拖慢了服务。最终改为并行同步/衰减专家热度，slot eligibility 使用现有 histogram，避免重复扫描路由；无 miss 时跳过 admission 扫描。第一次结果保留在 lfu-n1-short，不能和优化后版本合并。最终性能与验证见 REPORT.md。
