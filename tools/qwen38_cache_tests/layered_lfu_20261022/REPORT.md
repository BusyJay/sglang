# 分层 LFU 路由与 CPU 热度：实现和验证

2026-10-22，用户确认的语义：**低于本层 GPU victim 热度的 miss 才是 CPU 候选，候选数不超过 n 时留 CPU，超过 n 全搬到 GPU；CPU 计算也要更新 LFU，热起来后能迁回 GPU。**

## 结论

该规则已实现并真实运行，不依赖成本表。最重要的状态修复是 CPU 专家具有独立于 GPU slot 的 LFU 热度，不再因过滤出 GPU fill 计划而丢失热度。固定 CUDA 用例确认同一 CPU 专家前两次留 CPU、第三次迁入 GPU，且 slot 保存其累计频次而非重置成 1；缩小槽位容量后 CPU 热度仍能继续累积和触发迁入。

**正确性和完整生命周期验证通过，性能仍是混合结果，不能签署所有场景稳定提速。** n=1 比 n=2 更适合作为当前的小规模实验参数，但这不等于已证明 CPU 共算的独立净收益；n=0、当前 GPU-only 与原 Git GPU 基线均保留对照。未部署到正式服务，也没有把默认模式自动切到 layered_lfu。

完整规则和边界见 [DESIGN.md](DESIGN.md)。实现增量 [lfu-routing.patch](lfu-routing.patch) 相对本轮开始时的源码快照；它包含 11 个源/测试文件，不是相对原 Git 基线的独立全量补丁。已从保存快照重新应用并逐文件 SHA256 比对，见 [patch-check.log](patch-check.log)、[source-manifest.json](source-manifest.json)。

## 算法要点

- GPU hit 始终留 GPU，本次 active hit 不能作为 victim。空闲 slot 优先给最热的 miss；全体 miss 放得进空位时全部搬 GPU。
- 对其余 miss，用本层非 active GPU 专家的最低热度作保守 admission 门槛；只有严格低于它的专家可留 CPU。相等热度时搬 GPU。没有模拟连续置换后的更高 victim 门槛，因此可能保守地把更多 miss 放回 GPU。
- 冷候选数 C≤n 则全部留 CPU；C>n 则整个集合走 GPU。n 限制的是冷候选，不是全部 miss。CPU 专家不占 slot、不做 GPU 填充，不后台补入。
- 热度为 `[layers, experts]` 的 float32 状态（本模型 98304 字节），不依赖 slot 的增删或迁移。有效 decode/verify 时先同步 resident 频次，再将所有专家热度×0.95，并对原始 route union 中的 unique expert 各加 1，包括 CPU 专家；与原 GPU LFU 每次调用的计数单位一致。
- GPU 迁入后的 slot 频次从专家累计热度恢复，避免 CPU 热专家到 GPU 后又被当作频次 1 的冷专家。GPU frequency、CPU history 与 CPU/GPU mask 在同一个 planner 中有序更新。
- prefill 仍走原 GPU 路径，resident 频次在下次 decode 同步；本轮没有重写 prefill 的计数规则。旧 calibrated/probe/zero 模式保留。旧成本表会因源码指纹变化而失效，不应修改 metadata 强行复用。
- 首版按 slot 扫所有 route 且串行更新热度，实测明显拖慢。最终使用已有 histogram 检查 active、并行同步和衰减热度，全 hit 时跳过 admission 扫描。首版结果 lfu-n1-short 及 initial-slow/ 快照保留，但不并入最终版本的数据。

## 正确性验证

[tests-final.log](tests-final.log)：**133 passed，340 subtests passed**，包含原全局/分层 policy、dispatch、resize、loader、close 和真实 GPU planner。新 CUDA 用例验证：

- C≤n 留 CPU、C>n 全 GPU、n=0，以及 GPU hit 永不成为 victim。
- CPU 连续命中、自动迁入 GPU、迁入频次保留、之后成为 GPU hit。
- 空闲/部分空闲容量、热 miss 优先入空槽、相等热度留 GPU。
- B12X 全 CPU 无安全 GPU route 时回退，重复 route 不重复计数，CPU 专家与 GPU fill 集合不相交。
- slot reset 与容量 4→2 后保留非驻留热度并继续触发迁入。
- 配置 n 的类型/范围，以及错误模式下的参数拒绝。

[ruff.log](ruff.log) 为本轮源/测试文件检查；git diff --check 通过。CPU 原生计算库未修改，本轮 CPU 算术仍使用此前已执行 scalar/CSF 参考检查的相同二进制。未执行新的全量模型 accuracy 基准或多小时压力测试。

## 真实模型与对照

同一 RTX 5090 D、Ryzen 9950X 物理核 2–5、本地 nightly 镜像与本地 Qwen3.8-Flash-Next FP4 pack；断网容器，未下载模型。单请求、MTP2、breakable decode graph、6388 名义 cache slots，借用 KV 后 stage0 可到 7006；KV stages 80k/160k/220k，context220k。全部命令、输出文本、流式事件、MTP 接受率和日志在各目录。

短组为预热后 3 个 1088 输入/512 输出请求取中位数，加自然聊天/512 输出；完整组再执行 170064 输入/8 输出与之后的短请求。长请求只有 8 个输出，不用它的 decode 速率作持续吞吐结论。

| 版本 | 短请求中位数 tok/s | 自然聊天 tok/s | CPU calls / CPU experts（最近38400调用） |
|---|---:|---:|---:|
| 当前源码，原分层 GPU-only | 103.341 | 99.485 | 0 / 0 |
| LFU heat，n=0（无 CPU 共算） | 104.475 | 98.626 | 0 / 0 |
| LFU，n=1 短组 | 102.355 | 103.222 | 757 / 757 |
| LFU，n=2 短组 | 101.562 | 101.784 | 1219 / 1599 |

n=0 无 CPU worker，不可以把它的结果当作 CPU 共算提速。n=1 自然聊天接受率约 2.0078，n=0 约 1.9033；路由和 MTP 接受率变化会影响吞吐，不能将差异全部归因于搬运量。所有 GPU 测试串行，未锁 GPU 时钟。n=1 的两轮短组还存在波动，因此不选最好一轮签署性能达标。

完整组使用原 Git `9321a3f4504534811676c4d1f39d62341089e610` GPU-only 独立 worktree 对比 n=1；顺序为原 GPU→LFU n=1，前面另外有 n=1/n=2/n=0 短组。原 Git 对照还包含最初优化包的差别，当前源码 GPU-only 对照用于避免把这些差别全部算作新 LFU 算法收益。

| 完整组指标 | 原 Git GPU-only | LFU n=1 | 变化 |
|---|---:|---:|---:|
| 短 decode 中位数，tok/s | 98.100 | 103.770 | +5.78% |
| 自然聊天，tok/s | 101.957 | 99.193 | -2.71% |
| 170k TTFT，s | 29.971 | 30.117 | +0.49% |
| 长输入后短 decode，tok/s | 88.594 | 95.358 | +7.63% |

两组均完成 KV/专家池的短→170k→短生命周期。n=1 最近 57600 runtime 调用快照：CPU calls=1095，CPU experts=1095，original misses=154926，GPU fills=153831；计数满足 misses=CPU experts+GPU fills。

GPU decode copy bytes：原 Git 463571251200，n=1 425311948800，同调用计数周期快照约 -8.25%。这是周期快照，不是全部请求最后一次调用后的总量；也包含后续 LFU 驻留与实际路由的变化，不可全部归为 CPU 直接绕过的 payload。n=1 GPU cache hit rate 为 87.3562%，原 Git 为 86.4006%，同样受调用组成影响。

n=1 完整组自然聊天 MTP accept length=1.939394，原 Git=1.939394；长输入后分别 2.415094 / 2.348624。不宣称所有生成文本完全相同，也不将全部吞吐变化归因于 CPU 算法。

原始结果：[gpu-full](gpu-full/performance.json)、[gpu-current](gpu-current/performance.json)、[lfu-n0](lfu-n0/performance.json)、[lfu-n1](lfu-n1/performance.json)、[lfu-n2](lfu-n2/performance.json)、[lfu-n1-full](lfu-n1-full/performance.json)。[analysis.json](analysis.json) 保存快照、TTFT、跳过前64 token 的速率与接受长度。日志末尾的 SIGTERM/SIGQUIT 清理来自 benchmark 在请求组完成后停止进程组，不是请求期间的 CUDA 错误。

## 启用与复现

在已有 model-loader-extra-config 中加入以下字段即可选择本轮策略，不需要校准表：

```json
{"cpu_hybrid_mode":"layered_lfu","layered_cpu_max_experts":1}
```

保持原有 threaded、adaptive LFU、W4A16/B12X 和 mailbox 前提。n 为整数 [0,30]，默认 1，prefill/draft 沿用原路径。默认未自动启用该实验模式，也未恢复或重建正式推理容器。

用新输出名重放，不覆盖保存的证据：

```bash
cd tools/qwen38_cache_tests/layered_lfu_20261022
bash replay.sh fresh-gpu --mode off --stats-interval 19200 --short
bash replay.sh fresh-n0 --mode layered_lfu --max-cpu-experts 0 --stats-interval 19200 --short
bash replay.sh fresh-n1 --mode layered_lfu --max-cpu-experts 1 --stats-interval 19200
bash replay.sh fresh-n2 --mode layered_lfu --max-cpu-experts 2 --stats-interval 19200 --short
```

本轮未改变全局 cache 的 admission 和 CPU 算术；全局路径的单测/CUDA 回归包含在 133 项测试内。最初严格全局无回退的验收仍见此前 [补丁报告](../patch_validation_20261022/REPORT.md)，本轮不把分层结果当作全局性能验证。
