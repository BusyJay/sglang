# KV sizing / 专家槽位显存复核（2026-10-19）

已确认存在少量未能利用的显存，但没有复现之前那种大块临时 tensor 残留。
同为 6382 基础槽位时，staged 与 unstaged 在 KV sizing 边界的 driver free
相差 **14 MiB**：约 **6.412 MiB 是专家 VMM 页对齐，7.588 MiB 是普通
allocator 空洞的增量**。前次只归因于 VMM 对齐不够完整。

已将两个很小的 alpha 数组移出独立 VMM 映射，在普通 allocator 中一次性
分配完整最大容量，地址在 stage 切换时保持固定；权重本体仍用 VMM。
这样消除了两张独占的 2 MiB VMM 页，普通 allocator 的现有空洞足以容纳
192 KiB 的完整辅助数组，没有新增普通 allocator segment。

利用这项回收和原有剩余 sizing 余量，生产基础槽位由 6382 提至 **6388**。
三档实际专家预算为 **7006 / 6643 / 6388**，原来是 7001 / 6635 / 6382。
新增的 6 个基础槽位并非全部来自 4 MiB 回收，也消耗了此前没用完的余量。
最终 KV 保持 **220032**；静态比例 0.90、prefill chunk 8192、MTP=2 不变。

## 对照和证据

使用同一源码、checkpoint、6382 槽位、启动共享缓存、TP1 和并发 1，对照
开启/关闭 staged KV。只在隔离端口启用已有 startup memory probe；
它会同步 CUDA 并保存 allocator history，故这些运行不用于性能比较。
第三组启用小数组优化及 6388 槽位，再做同样的快照。最后关闭诊断并恢复
原服务端口，重新核验正常启动。

所有数字都取自 `kv.budget.after_empty_cache`，即已有 `gc.collect()` 和
`empty_cache()` 完成之后，而非权重加载中的临时峰值。

| KV sizing 边界，MiB | unstaged / 6382 | 修复前 staged / 6382 | 修复后 staged / 6388 |
| --- | ---: | ---: | ---: |
| driver free | 7106.500 | 7092.500 | 7080.500 |
| 普通 allocator reserved | 24208.000 | 7388.000 | 7388.000 |
| 普通 allocator active allocation | 24203.773 | 7376.186 | 7376.375 |
| 普通 allocator inactive 空洞 | 4.227 | 11.814 | 11.625 |
| 等待 stream 完成才能释放 | 0 | 0 | 0 |
| 专家 VMM 实际 backing | 0 | 16834.000 | 16846.000 |
| allocator / 专家 VMM 之外的 driver 占用 | 887.813 | 887.813 | 887.813 |

unstaged 的专家数据在普通 allocator 内，因此其 reserved 不能与 staged 的
普通 reserved 单独比较；必须加上 VMM 实际 backing。
第三组 driver free 比第二组低 12 MiB，是增加了 6 个基础槽位后的结果，
并非回收失败。同样的 6388 布局若仍给 alpha 独立 VMM 页，将多占 4 MiB。

两个 GPU 上对 POSIX_FD allocation 查询的 CUDA **minimum 和 recommended
granularity 都是 2 MiB**，因此不能直接把本机映射改成 64 KiB。
小数组原先每个在当前预算下仅约 25 KiB；即使覆盖 48×512 最大总槽位，
每个也只需 96 KiB，远低于一张 VMM 页。

`allocated_bytes` / `reserved_bytes` 的全池统计包含约 **64804 MiB 的 VMM
虚拟保留量**，不能据其与 driver used 的差值推断物理泄漏。
修复后的虚拟池统计为 64800 MiB。分析程序按 `segment_pool_id` 分开统计，
VMM 物理量按 manifest 行字节数和实际映射粒度核算；非 allocator 开销在
三组中完全相同。

普通 allocator 仍有约 11.6 MiB inactive 空洞，被同一页内的活跃对象夹住，
已有清理不能直接释放。此次没有声称消除了这些空洞，也没有发现它们背后
有可直接删除的大块无用 tensor。allocation stacks 随关键快照保留。

## Token 数字的含义

之前的 220608 → 220032 还包含**显式上限**的变化：补丁最终 stage 是
220000，page64 向上取整为 220032；模型 context length 始终是 220000。
本轮 unstaged / 6382 自动计算出 222080，而 staged 会按设计截到 220032，
不能把这整个差值都当成碎片。

现在启动日志同时打印截断前容量：6388 槽位的真实正常启动为
`profiled_tokens=220160 required_tokens=220032`，随后分配 220032。
这项日志保留在实现中，方便后续区分预算损失和 stage 上限。

## 验证与重算

- 完整 CPU 套件 347 项通过，见 `cpu-suite.log`。
- GPU 两种布局各 300 次 grow/return 循环：全 VMM 布局和小 alpha 混合布局。
  检查所有数组的稳定地址、保留前缀数据、物理字节、handle 数和预算差分。
  小 alpha 布局的基础 VMM handle 数从 6 降为 4；见 `gpu-cycles.json`。
- 6388 槽位的完整模型：74K → 170K，跨越 80K/160K，flush 后短请求回借
  到 80K，再次 74K 扩容，全部完成；见 `stage-cycle.json` 和运行日志。
  诊断开启下的耗时不作为吞吐证据。
- isort / ruff 和 `git diff --check` 检查；生产恢复后的配置与日志另行保存。

在本目录运行 `python3 analyze_budget.py` 可重算 `budget-summary.json`。
`*-budget-segments.json.gz` 是 sizing 时刻完整 segment/block/stack JSON；
`*-budget-events.json` 省略逐模块 postprocess 事件，只保留关键阶段。
原始 pickle / 全量 history 仍保存在宿主机：

```text
/home/jay/workspace/prod/homelab-llm/models/startup-memory/kv-sizing-20261019-current/
/home/jay/workspace/prod/homelab-llm/models/startup-memory/kv-sizing-20261019-unstaged/
/home/jay/workspace/prod/homelab-llm/models/startup-memory/kv-sizing-20261019-aux6388/
```

`launcher-before.sh`、`common-before.sh` 为实验前部署；`launcher-final.sh`
为恢复原端口、关闭 probe、6388 槽位的最终 launcher。三个 patch 继续启用。
