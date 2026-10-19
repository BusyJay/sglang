# Prefill 直接读取 cache（实验选项）

在现有 `--model-loader-extra-config` JSON 中加入：

```json
{
  "runtime_mode": "threaded",
  "prefill_reuse_cache": true,
  "prefill_direct_cache": true,
  "cache_overlap_policy": "auto"
}
```

保留原有的 `manifest_path`、容量和其他部署字段。这段 JSON 只展示相关选项，不能代替完整启动配置。`prefill_direct_cache` 默认关闭。`cache_overlap_policy` 可选，默认 `auto`（按下文判据决定是否拆分），`always` 复现早期版本的强制拆分行为；非法取值在启动时报错。

当路由选择 arena 路径时，运行时找出本层 cache 中仍然有效的专家。命中的专家直接使用 cache 中的权重，未命中的专家按需加载到 arena。已存在于当前层 arena 中的缺失专家不重复加载。没有任何 cache 命中时沿用原有 arena 路径，包括自动选择稀疏或整层传输的逻辑。

当前 MoE 内核每次只接受一套连续权重，因此混合命中分成两次执行：一次读取 cache，一次读取 arena，再合并输出。路由权重保持原值，不重新归一化；另一部分的专家 ID 设为 -1（GPU 路径复用原路由权重），由内核跳过，避免访问未加载权重。不能复用有效专家 ID 作为零权重占位，否则 CUTLASS 的重复路由可能覆盖有效贡献。全部命中时只执行 cache 部分。该逻辑用于 CUTLASS 和 W4A16 两条计算路径。

运行时先发布命中列表，随后在传输流上加载 arena；计算流立即执行 cache 部分，只在 arena 计算前等待权重就绪事件。退出时仍会等待传输提交完成，并把传输依赖加入计算流，包括全部命中及计算异常路径，以保护下一次显存复用。

如果 arena 借用专家池尾部，重叠的 cache slots 不能作为命中；仅使用没有被 arena 覆盖的前缀 slots。现有计算完成事件覆盖两次计算，arena 归还时沿用原有恢复逻辑。混合路径不从部分填充的 arena 执行 cache seeding/warming，避免复制未初始化行；没有命中的原有路径仍保留这些行为。

`prefill_copy_stats` 增加 `direct_cache_runs` 和 `direct_cache_experts`，分别记录走直接读取分支的层调用数和每次调用中直接读取的不同专家数之和。它们不代表 token 路由次数。缺失专家的主机传输继续计入 `host_missing` 和 `prefill_h2d_bytes`。直接读取不会增加 `prefill_reuse_d2d_bytes`，该指标仍只记录旧路径的 cache→arena 复制。

收益是省去命中专家的 cache→arena 权重复制。代价包括第二次 dispatch、路由处理、占位路由计算、输出合并和临时输出；缺失专家高度碎片化时，强制按需传输也可能比原有整层传输慢。arena 的预留容量保持原状，不会因为这项开关自动变小。两部分分别输出 BF16 后再相加，舍入顺序也与单次执行不同，因此不保证逐位相同。

CPU 测试覆盖混合路由、全部命中、未初始化权重行、输出缓冲复用，以及借用尾部后仅加载缺失专家。原始补丁仅完成 CPU 验证。目标 5090 上的 CUTLASS/W4A16、B12X、breakable graph、VMM 搬迁和性能实验记录见 [实验报告](cache_overlap_20261019/REPORT.md)。该实验发现并修正了重复占位路由导致的数值错误；热缓存小批量场景没有获得性能收益。

相关的容量保留和 decode 提前计算选项见 [CACHE_OVERLAP_AND_RESIZE.md](CACHE_OVERLAP_AND_RESIZE.md)。

后续优化直接使用 GPU lookup 拆分路由，消除每次构建 expert/slot CUDA tensor 的两次 H2D 和同步。开启此选项的普通输入 CUTLASS prefill 还会限制内部调用为 2048 tokens，降低 workspace；外层 chunk、专家池和 KV 容量不变。最新实测见 [优化复验](cache_overlap_20261019/optimization/REPORT.md)。

## cache_overlap_policy：由实测区间决定要不要拆

`auto`（默认）不改变任何所有权，只在已有的 CPU plan（路由计数、cache keys、arena 驻留集合）上做判断，不增加逐 token 的 GPU 状态读回：

- 本层所需专家已经全部在 arena：沿用原有单次 arena 调用，连命中列表都不构建，也不提交第二次 dispatch。
- 没有任何缺失：直接读 cache，仍是一次 dispatch。
- 混合命中：只有 W4A4、已标定几何（hidden=2560、intermediate=640、top-k=10）、总路由 >=10240、命中路由 >=1024、且至少 3 个专家确实需要从主机搬运时，才拆成 cache + arena 两次；其余情况回到单次 arena。
- 未标定的后端或几何：单次 arena。

`always` 忽略上述判据，强制走拆分路径，用于 A/B 对照和后续调优，不建议在部署中使用。

阈值来自目标 5090 上的档位扫描：小批量（3/128 token）混合命中拆成两次计算实测更慢，大批量且存在真实主机搬运时才有可复现收益。方法与逐档数据见 [动态路径选择复验](cache_overlap_20261019/adaptive/REPORT.md)。`prefill_copy_stats` 相应增加 `auto_arena_runs`、`auto_cache_runs`、`auto_split_runs`，记录每层选中的分支次数；它们同样是调用次数，不等于节省的字节或时间。
