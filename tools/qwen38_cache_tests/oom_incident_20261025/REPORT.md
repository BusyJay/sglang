# 2026-10-25 凌晨 OOM 分析

01:21:19 的直接故障是 **paged KV token allocator 返回 None**，随后 Scheduler 抛出 `RuntimeError: Prefill out of memory`。日志与当前未修改的 allocator 代码一致，支持的直接原因是：**分档 KV 的虚拟可回收容量被用于实际分配前的驱逐判断，但所需物理页尚未归还**。这不是该时刻的一条 `torch.OutOfMemoryError`。

## 核心证据

北京时间时间线：

| 时间 | 事件 |
| --- | --- |
| 01:20:18 | KV backing 从更大档回借到 160000，expert 总槽数 6798 |
| 01:20:59–01:21:12 | 一个新的 79124-token 输入分块 prefill；另有请求排队 |
| 01:21:12 | 最后一个 5396-token chunk 完成，日志 pending token=53631 |
| 01:21:13 | expert 层间重分布：搬迁 6313 个 resident experts，淘汰 485 个；总槽数仍是 6798 |
| 01:21:16 | backing 仍是 160000；expert physical bytes 仍是 18796969984 |
| 01:21:19 | 为增量 1343 tokens 分配 KV 页失败，Scheduler 异常退出 |
| 01:22:17 | 服务重启后 ready |

错误原文：

```text
Try to allocate 1343 tokens.
Available full tokens: 167744
  full_available_size=60800 + full_evictable_size_=106944
Available mamba: 6
  available_size=1 + component_evictable_size_=5
```

这里的 60800 并非全部已经 backing 的物理空闲：

| 容量项 | token 数 |
| --- | ---: |
| 最大虚拟 KV 容量（64-token page 向上对齐） | 220032 |
| 当前物理 backing | 160000 |
| 借给 expert、尚未映射到 KV 的尾部 | 60032 |
| allocator 报出的 available | 60800 |
| 当前真正物理空闲 | **768 = 60800 − 60032** |
| 此次增量 | 1343 |
| 所需新页容量 | **1344 = 21 × 64** |
| 至少需要补足的物理页 | **9 页 = 576 tokens** |

树 dump 中锁定的 full prefix 共 **52288 tokens**。加上本次增量 1343，恰好等于此前排队的 **53631**。同时 `52288 + 106944 = 159232`，与 `160000 − 768` 完全一致。这是前缀复用/追加的小增量触发的分配问题；冷启动空缓存下 219K 成功不能覆盖这个状态。

## 代码中的失败链

- `python/sglang/srt/mem_cache/allocator/paged.py:209`：`available_size()` 将已 backing 空闲页与 `loan_pages` 相加。
- `python/sglang/srt/mem_cache/allocator/base.py:181`：驱逐前计算 `num_tokens - available_size()`。约 1407 的保守需求小于 60800，因而不驱逐缓存。
- `python/sglang/srt/mem_cache/qwen38_kv_stages.py:201`：`StagedKVLoan.prepare()` 不立即归还物理 backing，扩档留到既有 result fence。
- `python/sglang/srt/mem_cache/allocator/paged.py:293`：实际 `free_pages` 只有约 12 页，无法满足 21 页，返回 None。
- `python/sglang/srt/mem_cache/allocation.py:239`：把 None 升级为 prefill OOM，Scheduler 退出。

`run_batch()` 中的 `cache_maintenance_before_forward()` 在 `prepare_for_extend()` 分配之后执行。因此新请求/HiCache 恢复若在本轮调度中消耗了前面留下的物理余量，仅依赖“下一个 forward 再标记扩档”不能保证本次分配成功。常规 HiCache `cache_controller.load()` 也直接从同一 allocator 分配页。

已经确认容量口径不一致足以复现本次状态。究竟是哪一次 prefix load-back、排队或 result 处理交错令扩档准备落后，需要更细的分配日志或原请求回放才能确定；现有日志不足以锁定这最后一段时序。

## 对“slots 同时搬运”的核对

这个判断有时间上的依据：搬迁比崩溃早约 6 秒，不能无视。但：

- 搬迁前后总槽数均为 6798，expert physical bytes 都是 18796969984，没有增加一整份 expert pool。
- `_migrate_pool()` 在既有 pool 内原地搬迁，复用启动时分配的单 expert-row scratch。
- 层内重分布要求 target 总量等于当前 pool 总量；它与“归还 KV loan 并从 160K 扩到 220K”是两种不同操作。
- 在完全不执行任何 slot migration 的 CPU 状态回放中，同样会在实际页数检查处返回 None。

因此，**现有证据不支持“搬运临时多占一整份显存，直接导致 CUDA OOM”**。搬迁可能参与维护/调度交错，不能凭现有日志彻底排除间接影响；修复重点应先放在实际分配时的物理容量保证。

## 状态回放与限制

`replay_capacity.py` 从当前源码 AST 提取实际 `available_size()`、`evict_to_free_tokens()`、`alloc_extend()` 和 loan `prepare()` 方法，在 CPU 上代入事故状态；仅替换 tensor 分配/内核执行和外部驱逐接口。

- 无搬迁、无物理驱逐：返回 None，重现分配失败。
- 额外驱逐 576 tokens 的物理页：通过物理容量检查。

这验证了容量判定机制，不是完整模型或原始请求的回放。事故配置 `crash_dump_folder=null`，未找到该请求 dump，不能声称已复现所有异步交错。

## 排除与现状

- 当时 `decode_early_cache=false`、调用数 0；命中早算没有参与该次故障。
- Mamba 还有可用/可驱逐槽；异常栈是 full KV 页分配，不是 Mamba slot 分配。
- 00:52、01:09 等 Triton “avoid CUDA OOM” 是迟加载预警，不是实际 OOM 事件。
- 10/24 20:40 另有一次真实的 CUDA 96MiB 分配失败，是较早的另一事件，不能与 01:21 的 token-pool OOM 混为一谈。
- 当前服务健康 bundle 已保存；本轮未修改生产代码/配置、未重启服务。此逻辑尚未修复，类似长时间前缀复用/排队状态仍有重现风险。

## 修复方向

1. 区分 admission 的逻辑可回收容量和 allocation/eviction 的当前物理可分配容量。实际 paged extend 应先按物理 shortfall 驱逐可释放的前缀页，再验证页数。
2. 可驱逐物理页不足时，先完成已有 result fence 上的 KV 归还/扩档，再重试或推迟调度；不能直接把可恢复的容量缺口升级为全服务崩溃。
3. 检查 HiCache load-back 对下一 prefill chunk 的物理预留，避免本轮恢复前缀消耗此前为下一次分配留下的空间。
4. 回归须覆盖非空 prefix cache、160K backing 边界、排队缓存命中、stage 回借及 adaptive relocation 组合；单一冷 219K、串行短→长→短不够。

本次只分析并保留证据，没有通过调小 memory fraction、关闭搬迁或关闭 HiCache 掩盖该逻辑问题。

产物：`crash-context-full.log`、`night-server-settings.json`、`state-timeline.json`、`migration.json`、`locked-prefix.json`、`capacity-replay.json` 和 `live-bundle/`。Bundle 收集成功；自带文本摘要工具因 `graph_gb` 返回 dict 而非 number 无法渲染，未影响原始 JSON 的采集和本报告的手工核对。


## 已实施修复与验证（2026-10-25）

工作区主目录的代码已修复，正式后端已重启并恢复 ready。生产仍使用原配置：220K context、0.92 memory fraction、原 expert budgets；早算开关未启用。

- `allocatable_size()` 区分已 backing 空闲页和 admission 可计入的 loan。分配前驱逐、decode 容量检查及分片驱逐重试使用物理可分配口径；常规 allocator 保留原口径和分片容量 floor。
- staged-loan prefill 按 CPU 长度元数据计算准确的新页需求。这次状态只驱逐 9 页 / 576 tokens，不额外驱逐保守上界对应的页。
- HiCache 的整段 load-back 在 loan 尚未全归还时保留下一 forward 的 reserve，空间不足则使用已有分块重算回退。扩档/归还仍在既有 result fence 进行，没有新增 CUDA host synchronize 或提前 unmap。
- 注册回归用真实 Paged allocator 和 RadixCache 复现 52288 prefix + 1343 extension、60800 逻辑空闲 / 768 物理空闲、106944 可驱逐状态。修复前同样 OOM；修复后实际分配成功、输出索引唯一、只驱逐 576、锁定前缀完整。HiCache reserve 回归也已验证修复前失败、修复后通过。
- CPU 相关用例共 150 项：146 通过、4 跳过（分片套件原有跳过）；日志见 `kv-tests.log`、`prefill-tests.log`、`paged-tests.log`、`allocation-tests.log`、`interleave-tests.log`。Ruff 和 diff whitespace 检查通过。
- 正式模型顺序运行 52288、79124、53631、159000、53631、219000、53631、54974-token 合成请求，全部生成 8 tokens；前缀追加命中 52288，长请求后的恢复命中 53568；最后算术问答返回 42。完整请求结果见 `serving-validation.json`。这是覆盖状态切换的合成验证，不是缺失的事故原请求回放，也不能代替整夜运行。

独立补丁保存在 `oom-fix.patch`。`replay_capacity.py` 固定读取 Git 2e60da32bd 的修复前方法，保留原诊断状态可复现性；修复后验证使用注册测试中的真实方法。

## 命中提前算同步审计

- 命中划分、planner、FC1 mask 都在 GPU 上；全命中时命中的 FC1 直接提交，不读取 CPU 命中结果。
- fill kernel 在 `miss >= plan[1]` 时立即退出，全命中不会读取 mapped failure flag，更没有 CPU publication/acknowledgement 等待。混合 CPU 模式的 merge 只在实际选中 CPU routes 时等 CPU completion；纯 GPU 全命中绕过该等待。
- 原实现新建 `NativeFc1Stage` 使用 `torch.cuda.graph(...)`，当前安装 PyTorch 的上下文入口会执行 `torch.cuda.synchronize()`。已改为专用 capture stream + GPU wait_stream + 原生 capture_begin/end；首次建图不再使用这个整卡同步入口。
- 5090 实测在新绑定期间将 `torch.cuda.synchronize` 替换为抛异常仍通过；M=1/2/3、miss=0/1/3/10、含 CPU placeholder 的组合，FP32 结果与原 native 路径逐位相同。见 `fc1-no-host-sync-tests.log`，改动见 `early-no-host-sync.patch`。
- **尚存开销**：全命中仍排空 fill kernel、GPU filled event join 和空 late-FC1；它不是单个 fused native launch 的专门全命中 fast path。没有为了跳过这些 GPU 工作而引入 CPU hit-count 检查。mailbox 原有有限队列背压可等待 GPU completion，统计/维护也有既有同步，这些不是新增的每 token CPU 命中确认。
