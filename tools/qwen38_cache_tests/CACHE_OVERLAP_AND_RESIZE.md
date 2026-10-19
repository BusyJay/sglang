# Cache 计算重叠与容量变更保留热点

本次修改包含此前的 prefill direct cache 实现，可直接应用到 patch 包说明中指定的基准提交。新增选项均默认关闭，适用于 threaded runtime。

在完整的 `--model-loader-extra-config` 中合并以下字段，保留已有 `manifest_path`、专家容量、KV stages 和其他设置：

```json
{
  "runtime_mode": "threaded",
  "prefill_reuse_cache": true,
  "prefill_direct_cache": true,
  "preserve_cache_on_resize": true,
  "decode_early_cache": true,
  "cache_overlap_policy": "auto"
}
```

`cache_overlap_policy` 决定这些开关被打开后是否真的走拆分/提前路径，取值 `auto`（默认）或 `always`，需要 threaded runtime。`always` 强制按字面开启，用于 A/B 对照和调优；`auto` 只在实测有净收益的档位上拆分，其余档位回到原有单次计算。

## Prefill 的 cache 计算与 arena 加载重叠

命中专家直接读取 cache；运行时先发布命中列表，再加载缺失权重，计算流只在执行 arena 部分之前等待传输完成。借用 arena 的重叠尾部不能作为有效命中。没有命中则继续使用原有整层/稀疏传输决策。

混合命中目前仍有两次 MoE 调用，arena 预留容量保持原状。独立输出缓冲保护第一部分结果，原路由权重不重新归一化，无效路由使用 -1，避免重复有效专家 ID。错误和全部命中路径也会合并传输依赖，防止下一次执行覆盖仍在使用的内存。详见 [PREFILL_DIRECT_CACHE.md](PREFILL_DIRECT_CACHE.md)。

## Cache slots 变更保留热点

`preserve_cache_on_resize` 改变 KV loan stages 引起的专家池扩容/缩容。原有阶段切换会清空起始地址变化的层；开启后先恢复借用 arena 覆盖的 cache 权重，再搬迁保留的专家并恢复 lookup、频率和年龄。

扩容时保留原有驻留专家。缩容时每层按频率降序、年龄降序选择能放入新容量的专家，其余专家必须淘汰。热点依据当前缓存策略记录判断，不保证未来访问一定命中。固定总预算的 adaptive 重分配已有同样的搬迁机制。

搬迁使用已有的路径/环算法，每种权重张量仅需一个专家行的 scratch，避免分配第二份完整专家池。读取旧地址和写入新地址时使用两者中较大的合法 pool view，搬迁后才发布新布局。

缩容复用已有结果完成边界来获取策略快照；新增的热点搬迁必须完成后才允许 KV 解除专家尾部映射并复用显存。此实验版本在阶段变化时会 drain 读者，并等待搬迁完成，可能造成一次明显的切换停顿。它以搬迁成本换取后续命中率，不能宣称无等待或零成本。扩容也会 drain 并读取策略，后续 GPU 工作按 stream 依赖等待搬迁。

## Decode 提前计算共享和命中专家

`decode_early_cache` 在单 token 的 device mailbox 路径中，先保存 planner 更新前的命中 slot IDs，然后在 fill stream 加载缺失专家。计算流执行共享专家（满足现有 shared overlap 条件时）和命中专家，只在后续缺失部分计算前合并 fill 完成事件。planner 保护本次 active hits，禁止把它们当作缺失专家的 victim slots。

此版本仅在 W4A16 decode、非 inplace、标准输出侧路由的配置下启用提前专家计算；prefill、verify 及 W4A4 decode 不启用该分支。保持已有 W4A16/B12X 环境配置即可，不自动改变激活精度。共享专家沿用现有 `SGLANG_QWEN38_SHARED_FILL_OVERLAP` 开关和融合/并行限制；新开关不强制关闭共享专家融合。

提前部分使用 baseline Triton W4A16，未命中 routes 置为 -1，权重和 alpha 读取均掩码，以免与加载中的 victim row 产生读写竞争。原有 B12X/其他 W4A16 后端在 fill join 之后计算缺失部分，已提前计算的路由 ID 置为 -1，然后合并两部分输出。

planner 在更新 lookup 前直接输出 early/late IDs，消除额外的命中快照张量操作。全部命中和全部缺失时 early IDs 全为 -1，Triton projection 直接跳过 dot product，后半段计算有效专家一次；仍会提交空的 early kernels、归约和合并，不等于零额外开销。混合命中才计算 early hits，late IDs 排除这些路由。B12X 的无效路由处理仍有冗余工作，且 early 部分仍是 Triton，不能保证比原有 B12X 单次调用快。该选项需要实测收益，默认关闭。

强制开启（`cache_overlap_policy=always`）时，每次单 token decode 的 kernel 数从 7 增加到 12，B12X 上低缺失档位实测慢 60%~70%，缺失更多时仍不优于关闭；Triton 最好也只与关闭持平。因此 `auto` 模式让 decode 完全绕过提前分支，连空 kernel、事件和输出合并都不提交，`decode_early_cache_invocations` 应为 0。prefill 侧 `auto` 的判据见 [PREFILL_DIRECT_CACHE.md](PREFILL_DIRECT_CACHE.md)。

## Prefill workspace 上界

开启 `prefill_direct_cache` 且使用 CUTLASS W4A4 时，普通 BF16 输入的 MoE 调用内部最多处理 2048 tokens；外层 attention/GDN 的 prefill chunk 不变，专家规划和填充不按子块重复执行。输出逐块写回，包含不满 2048 的尾块。预量化输入带有特殊 scale 布局，保留原路径，不适用此上界。

GPU 路由拆分直接读取运行时 lookup，并用有效 slot 前缀排除借用 arena 覆盖的尾部；不再从 CPU 字典构建两份 CUDA 元数据，也不再复制两份零掩码路由权重。混合 prefill 仍有两套 MoE dispatch；内部切块增加调用数，以降低瞬时 workspace 需求，需按完整工作负载评估性能。

## 观察指标

- `prefill_copy_stats.direct_cache_runs/direct_cache_experts`：直接读取分支的层调用数及每次调用中不同命中专家数之和。
- `prefill_h2d_bytes`、`prefill_reuse_d2d_bytes`：主机传输和旧路径 cache→arena 复制字节数。
- `decode_early_cache_invocations`：提前命中计算的 mailbox 调用数，包含全部命中/全部缺失，不等于有效重叠次数。
- `prefill_copy_stats.auto_arena_runs/auto_cache_runs/auto_split_runs`：`auto` 每层选中的分支次数（单次 arena、直接读 cache、cache+arena 拆分），只说明选路结果，不代表节省量。
- `adaptive_migrated_experts/adaptive_evicted_experts`：启用后也累计阶段搬迁的保留/淘汰专家数；前者包含留在原地址的保留专家。

共享 fill 计数沿用旧字段；现在该字段也包含仅提前计算 cache 的调用。以上计数不等于测量到的吞吐提升。GPU kernel 与传输争用资源，是否实际重叠以及节省多少时间必须看 profiler 和端到端测试。

## 验证命令

CPU PyTorch 环境：

```bash
PYTHONPATH=python python3 tools/qwen38_cache_tests/run_cpu.py
PYTHONPATH=python python3 test/registered/unit/layers/moe/test_qwen38_prefill_routes.py
PYTHONPATH=python python3 test/registered/unit/layers/moe/test_qwen38_cache_overlap_policy.py
PYTHONPATH=python python3 test/registered/unit/model_loader/test_qwen38_nvfp4_cache_loader_config.py
```

SM120 离线编译（无需 GPU，不证明运行时正确性）：

```bash
PYTHONPATH=python python3 tools/qwen38_cache_tests/cached_w4a16_compile_check.py
```

SM120 的真实内核验证：

```bash
PYTHONPATH=python python3 tools/qwen38_cache_tests/cached_w4a16_gpu_check.py \
  --cache-overlap-only --iterations 10 --output /tmp/cache-overlap-parity.json
PYTHONPATH=python python3 -m unittest \
  tools.qwen38_cache_tests.test_gpu_cache.TestGpuCache.test_staged_resize_preserves_hot_rows_across_shifted_layer_offsets
```

前者使用合成 NVFP4 权重和独立 BF16 激活参考，覆盖 W4A4/W4A16 prefill、W4A16 decode、breakable replay、共享分支和命中计数；后者验证容量搬迁后的所有权重字节及借用 arena 恢复，但不执行真实 VMM 解除映射。真实 KV stages、B12X、完整模型输出误差、跨 stream 性能和切换停顿仍需在目标部署验证。

两部分分别舍入再相加，结果不保证逐位相同。GPU 测试入口的容差用于合成验证，不构成完整模型精度结论。

回退时把三个新增选项设为 false；彻底回退用 patch 包中的 `git apply -R` 命令。不要在服务运行期间直接更换已加载的源码和资源状态。

## 目标 GPU 实验

目标 5090 上的修正、验证范围和开关 A/B 结果见 [实验报告](cache_overlap_20261019/REPORT.md)。三个选项保持默认关闭；小批量热缓存的 prefill/decode 拆分路径均有性能回退。

本机 8192-token chunk 的 OOM 已进一步定位为大 workspace 与 HC/GDN 激活交错分配造成碎片。保持 6388 基础 slots 和原 KV 容量，实验进程设置 `PYTORCH_ALLOC_CONF=max_split_size_mb:512` 后通过 170K 请求；allocator 快照确认大块未再被活跃小块切碎。该设置没有自动写入生产部署。证据及热缓存变慢的逐项归因见 [排查记录](cache_overlap_20261019/investigation/INVESTIGATION.md)。

后续优化已在无 allocator 参数、外层 chunk=8192 的同一完整模型配置下通过 1K/8K/74K/170K 请求。最新证据及剩余性能限制见 [优化复验](cache_overlap_20261019/optimization/REPORT.md)；前面的 allocator 参数是保留的另一种实验方案，不再是该普通输入路径通过本机长请求的必要条件。

在此之上又补了一轮动态选路（`cache_overlap_policy`）：decode 无盈利区间故绕过提前分支，prefill 按缺失数、批量与 arena 状态选路，并用完整模型 canary 验证动态选择与原有 workspace 上界能同时工作。方法与逐档数据见 [动态路径选择复验](cache_overlap_20261019/adaptive/REPORT.md)。
