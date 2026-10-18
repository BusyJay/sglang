# 受控复测方案与当前证据缺口

状态：方案，尚未执行。原来的自由生成服务试验不能区分优化本身的开销变化、
MTP 路径变化和 expert cache 轨迹变化。当前结果应判为**效应尚未识别**。
保持 prepared=0/CTA=8 是临时维持部署现状，不是证明两个优化无效。

## 原试验为什么不能回答这个问题

- 固定提示、temperature=0 和输出长度，没有固定实际输出 token、draft proposal、
  accepted prefix、verify 的 M、路由 ID/权重或逐层 miss 计划。
- `cached_tokens=0` 只检查前缀缓存，不检查 expert cache。相同 slots 数也不等于
  相同 keys/lookup、LFU/SLRU 状态、迁移历史或 H2D 字节数。
- `accept_length / token_per_second` 是派生平均值，不会把不同的路由和搬运工作量
  变成同一份工作量，不能据此消除这些影响。
- 每个配置只有 3 个独立 worker；同 worker 的三个提示不能算作 9 次独立重复。
  顺序反转只能缓和顺序影响，没有补齐上述控制，也没有给出检测小收益的统计功效。
- 大的 fill GPU 时间占比，不证明 CPU 提交优化无法改变关键路径；with-stack profile
  也不能用来证明这个因果解释。旧报告中这部分归因撤回。

操作级固定输入测试确实观察到了 prepared 的 eager 调用收益和部分 miss 规模下
adaptive fill 的收益。它们支持继续验证；既不能直接外推服务收益，也不能被原服务
试验的负向点估计推翻。

## 第一关：录制可核验的同一份工作负载

将录制和计时分开。录制必须包含完整的 step/layer 顺序，不能只保留平均 hit rate。

| 类别 | 必须固定并校验的内容 |
|---|---|
| 调度与 MTP | 输入 token、位置、KV/混合状态起点；draft token、accepted prefix、verify M=1/2/3 及 draft/extend 顺序 |
| 路由 | 每层每 step 的原始 expert ID、router weight，以及映射后 slot ID |
| 缓存 | 每层容量及初始 keys、lookup、frequency、ages、generation、clock；resident 权重与 scale 的来源校验和 |
| 搬运 | 每 step 的真实 miss expert→slot 对、各 segment 字节数、source/destination 连续性及总 H2D 字节数 |
| 执行条件 | 真实原生布局、pack、GPU、stream/event 依赖、Graph 分段、并发计算、scratch 和内存预算 |

缓存权重恢复可以从原 pack 按 keys 重新填充，不必在 GPU 同时复制一整份 cache。
恢复与校验放在计时之外，保持原 cache pool 地址和容量不变；仅恢复 keys 不够，
不能让 metadata 与实际 resident 权重不一致。KV、Mamba、draft 状态也必须恢复。

现有 `route_trace_path` 只记录 layer 和 routes，并同步回 CPU；
`replay_cache.py` 只做 CPU miss-count 模拟。二者都不是满足本方案的 GPU 性能回放。
录制器、状态恢复和回放一致性校验完成前，不开始宣称“受控服务复测”。

## 第二关：同轨迹成对计时

1. 两条路径预先编译和预热；首用编译、内存增长和分配重试另作 cold-path 结果。
   不能把一边冷启动、一边热缓存放进同一个 steady-state 对照。
2. 同一 checkpoint、同一 tape，对 prepared 和 CTA 分别做单因素对照。
   每个独立 block 随机选择 ABBA 或 BAAB；每次回放前恢复同一状态。
   两个变体都要预热，L2/热状态采用同一预先定义的规则，不能固定让一个变体总是后运行。
3. 回放保留真实 MTP 的 shape/计算工作，但固定它的输入与接受序列，防止后续执行分叉。
   单独关闭 MTP 的 M=1 测试只能回答 M=1 路径的问题，不能代替 M=2/3 验证。
4. 除指定开关外，所有状态相同；每个 step 校验路由、slot、miss 计划和字节数的哈希。
   planner 仍实际执行，并校验其计划与 tape 一致，不能只绕过 planner 测 copy。
   分叉就标记该 block 不满足配对条件，报告原因和数量；判定不得依据快慢。
5. 以整个固定 step 序列的 CPU wall time（包含末尾 GPU 完成）作为主指标。
   只在 block 边界同步，不能逐层 synchronize 破坏原有重叠。
   GPU event 时间和每阶段 profile 作为解释指标，profile 不进入吞吐采样。
6. 必须保留 attention、GDN、draft、共享专家与真实事件依赖。
   若实现只回放了 planner/fill/B12X，就明确标为“专家子流水线”，不能称整机收益。

先做 A/A 回放，核验相同状态和计时噪声；再做 A/B。
分配重试、数值差异和回放不一致都要保留，不能事后以“离群点”为由删除。
如需单独分析无内存压力的稳态路径，必须预先定义条件，并同时报告原内存预算下的结果。

## 第三关：什么结果才能支持选择

先约定值得保留的最小收益，例如 2%（这是待确定的决策阈值，不是从旧结果倒推）。
用先导回放估计**成对 block 差值**的方差及相关性，再确定样本量与停止规则。
不能把一个 block 内的数千 kernel 当作数千个独立样本。

- 按固定工作负载分层报告 paired latency ratio、置信区间和 A/A 噪声。
- 若区间仍覆盖有实际意义的收益，结论是证据不足，不能写“优化没用”。
- 要排除“至少 2% 收益”，需要相应收益区间的上界低于 2%；
  要宣称在 ±2% 内等效，需要整个区间落入该等效区间。无显著性不等于等效。
- 若要声明服务吞吐收益，再用足够规模、随机分配的真实自由生成请求做部署验证，
  同时报告接受率、miss 字节和输出差异。这个阶段估计实际业务的总体效果，
  不再把变化的 token 路径伪装成严格配对的 kernel 对照。

受控回放给出“同样的工作做得更快吗”；业务验证给出“实际请求整体获益吗”。
两份证据都需要，且结论只适用于测过的硬件、工作负载与缓存条件。
