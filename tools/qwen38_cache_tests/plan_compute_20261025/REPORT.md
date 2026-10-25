# Decode planner 计算与路由整理优化，2026-10-25

基于 210K decode trace，完成一轮保持 LFU/CPU 分流策略的 planner 优化。以下记录对应 planner 独立验证阶段；当时生产进程未重启，尚无新的端到端 token/s 对照。后续的命中早算及完整模型验证见 [早算报告](../hit_fc1_overlap_20261025/REPORT.md) 和 [全命中融合报告](../all_hit_20261025/REPORT.md)。

## 实现

1. warp 归约携带完整的 `(slot, empty, heat, int64 age)`，并通过共享内存交接四个 warp 的最优候选，避免每轮 shuffle 后重新访问全局元数据。空槽优先、热度、年龄、slot tie-break 顺序保持一致。
2. filtered route 初始化由 route lanes 并行完成。仅在 CPU 分流实际发生时更新 histogram 并压缩 GPU active union。
3. 利用 CPU selector 只移除 unique nonresident experts 的契约，将剩余 GPU union 的 all-hit 判定化为 `original_misses == cpu_count`，去掉串行 lookup 循环。

不修改模型 dtype、MTP、cache 容量或 LFU/CPU 选择策略；没有新增环境变量和配置项。

## 固定状态测量

RTX 5090 D；512 experts / 146 slots / 3 tokens / top-k 10 / CPU limit 1。每个测量点 31 次独立 profiler kernel duration，单位 µs，取中位数。reset 在 graph 内，下面仅统计 `decode_plan_kernel`，不包含 reset。

| 场景 | baseline 两轮 | final 两轮 | 延迟降低 |
| --- | ---: | ---: | ---: |
| 全 hit | 6.368 / 6.368 | 4.320 / 4.320 | 32.2% |
| 冷 miss 留 CPU | 20.000 / 19.936 | 18.688 / 18.688 | 6.4% |
| 热 miss 搬 GPU | 11.392 / 11.392 | 8.960 / 8.960 | 21.3% |
| 冷候选超过 CPU 上限 | 14.144 / 14.144 | 10.720 / 10.720 | 24.2% |
| 部分空槽 | 14.176 / 14.176 | 10.848 / 10.848 | 23.5% |

baseline → final-r1 → final-r2 → baseline-r2 顺序；正式两轮 final 与 GPU 回归串行。早期 `final` 调试轮不用于上述性能结论，避免其与测试启动窗口接近带来的干扰。寄存器版、共享元数据版、路由整理版的阶段性记录也保留，不能用较快的中间版本数值替代最终实现。

GPU 时钟未锁定；生产服务在场但测量时无主动服务负载。未为测量调整生产参数。结果证明这些固定状态的 planner 延迟改善，不能直接外推所有 slot/route 或整模型吞吐。

## 正确性与检查

- 80 组单步场景，slots=16/64/146/256，tokens=1/3，n=0/1，覆盖 hit、CPU miss、热 miss、超过 CPU 上限和空槽。
- 12 组 × 64 步连续路由回放，n=0/1/3/30；所有正式前后轮的 defined 输出与完整状态哈希完全一致。未使用 plan 未定义尾部作比较。
- GPU cache partition / layered LFU / layered plan / hybrid plan：29 passed，73 subtests passed。
- namespace / layered policy / hybrid policy：124 passed，280 subtests passed。
- 增强已有串行 planner 对照测试：33/129 的非整 warp 容量、64 位高年龄位和年龄相同的 slot tie-break。
- Ruff check/format、CUDA 文件 clang-format 20.1.7 dry-run、registered-test validator、git diff --check 通过。已有紧凑格式的 .cuh 只保留语义修改，未进行整文件格式重写。

## 原始 trace 与优化边界

框架 SGLang，Qwen3.8 Flash Next FP4 Local，RTX 5090 D，TP1，220K context，memory fraction .92，Mamba slots 10，自动 expert capacity，MTP 和 CUDA graph 保持原配置。原 trace 为 graph-on 单 trace，无 graph-off mapping 配对。

`/home/user/workspace/prod/homelab-llm/models/profiles/pcie_long_context_20261024/decode210k-trace/pcie-decode210k-TP-0.trace.json.gz`

### Kernel table（原始 trace 的主要项）

| Kernel family | 累计 GPU 时间 | 累计占比 |
| --- | ---: | ---: |
| mailbox fill | 145.49ms | 27.6% |
| MXFP8 CUTLASS GEMM | 97.95ms | 18.6% |
| W4A16 small-M MoE | 74.99ms | 14.2% |
| cuBLAS BF16 GEMV | 41.13ms | 7.8% |
| decode planner | 14.70ms | 2.8% |

### Overlap-opportunity table

| 对象 | 已知证据 | 结论 |
| --- | --- | --- |
| planner | 位于依赖 fill/compute 之前的执行链；仅单 trace | 本轮降低其本身延迟，没有将潜在 overlap 算成实测收益 |
| mailbox fill | 原分析显示仅约 10.3% fill 时间与其他 GPU 工作重叠 | 需要依赖与 mapping/formal 验证，不据此更改执行次序 |

### Fuse-pattern table

| 对象 | 现有路径/先前结果 | 本轮决策 |
| --- | --- | --- |
| MXFP8 small-M | 已有 opt-in tile16；固定新容量下 ABBA 未显示普遍服务收益 | 保持原默认 |
| activation + quantization | 已有融合族；原 trace 自动匹配缺少可靠 Python 归因 | 不把家族匹配作为新增融合依据 |
| LFU victim reduction | shuffle slot ID 后反复读取比较字段 | 改成寄存器 key + 共享内存 warp 交接，固定回放验证 |

完整原三表在同级 `pcie_long_context_20261024/trace-triage.md`。单 trace 自动匹配的融合标签不能证明本模型实际缺少该融合。

原始 GPU span 为 490.47ms，planner 累计 14.70ms。若代表性的 planner 总时间也减少 20%–30%，时间预算约减少 2.94–4.41ms，即窗口的 0.6%–0.9%；这是假设性预算，不是实测吞吐，也未扣除 overlap。主要计算热点仍在 GEMM、MoE 与 GEMV。

## 复现与产物

`bench_planner.py NAME` 读取 PYTHONPATH 指向的实现，写入 `/results/NAME-*`；前后使用独立进程与相同随机种子，分别保留 graph timings、31 次 planner kernel trace 和状态快照。固定状态 benchmark 从上一轮已验证的脚本复用。

`summary.json` 为正式两轮前后测量摘要，`source-manifest.json` 为源码 SHA256，`optimization.patch` 可供审阅。原始 trace、snapshots 和中间候选仅作本地证据。实验容器已停止，生产继续运行；本轮未做新全模型 accuracy 或端到端吞吐验收。
