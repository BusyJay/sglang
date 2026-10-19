# 变慢与 8192 chunk OOM：实测归因

此页保留初版排查证据。后续已降低普通输入 CUTLASS workspace，并在没有 allocator 参数时通过外层 8192 chunk / 170K 请求；最新源码与验证结论见 [优化复验](../optimization/REPORT.md)。

本轮沿用同一基准、5090 D、FP4-Local checkpoint 和镜像。生产源码与上一轮修正版本相同；未保留未通过验证的 workspace 试验，也没有更改生产启动器。

## 变慢的具体原因

对同一组合成权重、固定路由、热重复调用，用 torch profiler 捕获 5 次调用。显式开启 B12X prepared launch；统计只计 `cat=kernel` 的实际 GPU kernel，不把 `gpu_user_annotation` 的范围时长当作 GPU 算时。profiler 有额外开销，以下是归因数据，不用于替代上一轮无 profiler 的延迟测量。

| 场景，每层每次调用 | 关闭 | 开启 | 直接证据 |
| --- | ---: | ---: | --- |
| 单 token decode kernel 数 | 7 | 23 | 提前部分增加掩码、路由、Triton 两次 projection、归约、clone 和输出相加；后续 B12X 仍执行 |
| 单 token decode kernel 累计时间 | 32.3 μs | 144.4 μs | 新增 Triton gate/up、down 合计 97.1 μs；B12X 仍约 24 μs |
| 3-token prefill kernel 数 | 10 | 41 | CUTLASS dispatch 从 1 次变为 2 次，GEMM 从 2 个增至 4 个；额外路由张量操作和合并 |
| 3-token prefill kernel 累计时间 | 86.0 μs | 136.7 μs | 除 GPU 算时外，增加大量主机提交间隙 |
| 3-token prefill 额外 CPU 同步 | 0 | 2 次 stream synchronize | 来自每次把 CPU 命中专家 ID、slot 列表转换为 CUDA tensor；路由拆分每次重建 lookup |

这里的 decode 已全部命中，fill kernel 约 0.85 μs，没有大块传输可以隐藏 97 μs 的额外计算。实现为了不做 D2H 判断，没有在全命中/全缺失时跳过一个专家调用；第二次 B12X 虽然命中路由权重为零，仍做专家计算。

热 prefill 测例反复调用同一层和同一路由，旧路径的 arena 已驻留，原本无需再次复制专家权重。因此新路径付出了第二次 dispatch、路由拆分和相加成本，却省不到权重复制。这解释了上一轮约 0.319→0.620 ms 的小批量热测结果。不能把该比值外推为 8192-token 完整模型的减速比例；大批量和实际 miss 场景的传输收益不同。

原始 traces 在 `profiles/`，机器可读统计在 `component-summary.json`。统一 profiler 工具输出在 `decode-triage.txt`、`prefill-triage.txt`。其自动 fuse 表把 MoE 错配为 NSA indexed-cache-store，属于工具误报，不采纳该融合建议。这里采用单 trace 归因，未把它描述为 mapping/formal 对照证明的重叠收益。

| Kernel 归因 | 已验证事实 |
| --- | --- |
| `_cache_project` | early decode 约占 66.9% GPU kernel 时间；每次两个 projection |
| B12X W4A16 | early 开启后仍有一次完整调用；约占 16.7% GPU kernel 时间 |
| CUTLASS GEMM / routing / finalize | prefill 拆分后两套 dispatch，另有大量 ATen 路由 kernel |

| 重叠机会 | 本轮结论 |
| --- | --- |
| 热 decode 的 fill 与早算 | 无大块缺失传输，收益上限很小；不能抵偿额外 SIMT 计算 |
| 混合命中、大量加载 | 仍需单独测实际路由/传输，不能从热测断言无收益 |

| 融合模式 | 本轮结论 |
| --- | --- |
| 自动 NSA 融合建议 | 与本次 MoE 算子不符，排除 |
| 路由掩码/lookup/合并 | trace 显示独立小算子很多；后续可以针对该路径优化，当前没有宣称已实现融合 |

## OOM 根因：大块 workspace 被长期激活切碎

保存了原请求序列 1024→8192→74000 tokens，chunk=8192、MTP=2、6388 基础 slots、mem_fraction_static=0.90、三个计算/搬迁开关开启。原始 allocator history 和崩溃快照在 `memory-1024/`、`memory-8192/`、`oom-snapshots/`，摘要由 `summarize_memory.py` 从可信本机快照重算，见 `memory-root-cause.json`。

这次在真正报错时：

| 项目 | MiB |
| --- | ---: |
| 普通 allocator 活跃分配 | 10204.35 |
| 普通 allocator 空闲碎片总量 | 1139.65 |
| 最大单个空闲块 | 680 |
| CUDA driver 剩余 | 460.50 |
| 此次需要的新连续 workspace | 822 |
| 等待 stream 完成后才能释放的块 | 0 |

trace 记录了两种 MoE workspace 请求：约 839.76 MiB 和 821.11 MiB（allocator 段分别为 840/822 MiB）。cache 与 arena 的专家容量不同，workspace 大小也不同。

分配历史明确显示：同一个地址先被用于 CUTLASS workspace，释放后被后续 HC/GDN 激活反复占用；最终的 840 MiB 段中，前 160 MiB 是 `fp32_grouped_gemma_norm` 的活跃张量，后 680 MiB 是空闲块。160 MiB 仍活着，所以 680 MiB 不能与其他空闲段拼成 822 MiB。另一个 720 MiB 段也被 HC combine 等活跃张量隔开。

因此，空闲总量够不等于能满足这个连续申请；`empty_cache` 也无法释放包含活跃块的整段。此次快照没有 `active_awaiting_free`，排除了“只要等 stream 完成就能释放”的解释。源码确认 FlashInfer 在未提供 workspace 时按调用申请/释放，不是简单持有两份永久 workspace，也未发现此处显存泄漏证据。

日志里的 75+ GiB PyTorch allocated 包含专家/KV VMM 的虚拟池。已按 `segment_pool_id` 分开统计，不能拿这个数字与 32GB 物理显存直接相减。

## 验证过的处理方式

为实验进程设置：

```bash
PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

它保护大于 512 MiB 的 native allocator 块，使 720/822/840 MiB 这些临时大块不被小一些的 HC/GDN 激活切碎；workspace 仍然按需申请和释放，没有永久占住一份 840 MiB 缓冲。

**未降低 chunk、slots、KV 容量或静态内存比例。** 无 profiler 的对照通过：1K、8K、74K、170K 请求，各生成 8 tokens；算术探针返回 42。报告在 `model-maxsplit8192/report.json`，复现入口 `run_maxsplit8192.sh`。这次是本机具体配置的验证，不是对所有 GPU/负载的保证。

保留 workspace 的私有池、常驻大缓冲两种代码试验没有通过完整模型验证：它们会抢占 GDN/PLE 的瞬时空间，导致 OOM 转移到其他分配点。相关代码已逐字节撤销，原始失败日志保存在 `model-private-pool-failed/`、`model-fixed8192/`；未把这类改动混入最终 patch。

allocator 设置仍会出现分配重试；它是已验证的显存处理方式，不意味着两个计算重叠开关因此获得性能提升。三个开关仍默认关闭。这里没有修改用户的生产部署配置。

修复配置的 allocator 快照对照也通过：`memory-maxsplit-summary.json` 确认 `max_split_size=536870912`，1024/8192/74000 三个阶段中，大于等于 512 MiB 且被活跃小块切开的混合 segment 数均为 0，allocator history 没有最终 `oom` 事件。完整 74K 请求再次完成。空闲缓存总量可能仍较大，但现在是可复用/释放的整块，不能仅凭 reserved-minus-allocated 判断问题未解决。

该参数经 `PYTORCH_ALLOC_CONF` 传入，PyTorch snapshot 的内部键仍显示为 `PYTORCH_CUDA_ALLOC_CONF`；两个名称在本镜像中指向同一设置。生产源码已校验与上轮修正版本逐字节一致，未保留试验性 workspace 改动。
