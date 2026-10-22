# Fill 分桶优化实施记录（2026-10-22）

本轮落实 REPORT.md 中“过粗成本键”和“成本表扩大后查找开销”两项。未修改原始瓶颈报告或历史校准产物，未采集新的 GPU trace，也未声称已经提升服务吞吐。

## 代码行为

成本表 schema 从 1 升到 2，新增必填 `gpu_fill_experts`：对每个 CPU 候选前缀，精确计算 `admitted + rejected - cpu_experts`。这是选定 CPU 子集后仍需搬到 GPU 的不同专家数量；相同 pack 几何下专家 payload 大小固定。路由数相同但搬运数不同的场景不再共用成本。主机与设备路径都传入该信息。未测桶回退 GPU，严格 margin 与所有已测代表必须获胜的规则保留。运行时源码指纹已有覆盖，修改后旧表不会被默默沿用，需要重校准。

设备表按 `(tokens, GPU routes, CPU routes, CPU experts, rejected experts, GPU fill experts)` 排序；每个 CPU 前缀二分找到桶，再在桶内精确比较两组 route histogram。查找从逐前缀全表扫描变为 `O(prefixes × (log(samples) + matching_bucket × histogram))`。小表继续使用 <=32 行 shared-memory 缓存，大表走 global-memory 查找。新增一列使表宽从 70 变为 71，所有调用端同步更新。公开 plan 接口要求输入按上述标量键排序，可使用 `HybridCostModel.device_cost_rows()` 打包。

校准仍使用相同 workload 的 CPU/GPU 配对时间，以同桶最差配对比例代表该桶。准备脚本不再仅按准入数选一个最常见场景：保留不同层、路由与专家组成的代表，只有整桶代表都能装进预算时才选择该桶。默认产物 M3 为 180 个代表 / 80 个完整桶，M1/2 为 72 个代表 / 3 个完整桶；保留更多代表意味着相同预算下覆盖桶数可能较少。

`source_replay_provenance` 保留旧回放的来源指纹，`provenance` 记录本次需要测量的当前代码与输入。准备输出是旧回放导出的分支 fixture，不是修改后连续服务缓存轨迹，也不能证明 CPU cache 冷热、服务 GIL 竞争或 P95。

## 历史样本潜力

`analyze_fill_potential.py` 可重复生成 `fill_potential.json`，只按实际 fill 数重分组历史原始时间，不输出可加载的运行时校准。

| 样本组 | 原始 case | 旧 key 允许的代表 | fill key 允许的代表 | 恢复的代表 |
|---|---:|---:|---:|---:|
| M3 | 180 | 57 | 135 | 78 |
| M1/2 | 72 | 17 | 58 | 41 |
| 合计 | 252 | 74 | 193 | 119 |

原始采样里同一粗 key / 准入数只选过一个代表，因此这些新 fill 桶经常只有一个历史样本。119 是历史代表数，不是在线调用比例或加速百分比；不能据此直接上线新表。

## 验证

- 本地已有 SGLang nightly 容器内：hybrid policy / dispatch / resize 三组 CPU 回归，87 tests、279 subtests 全部通过；日志见 `validation/cpu-tests.log`。
- CUDA JIT 模块在无 GPU 容器中以显式 SM120 目标编译并成功加载；见 `validation/compile-sm120.log`。这是编译检查，不是 GPU 执行或数值验证。
- 新增 GPU 测试覆盖 fill 不匹配的 GPU 回退、scan/indexed 两条路径、100+ 行表及同标量桶 histogram 匹配。当前机器没有 GPU，这些执行检查待目标机运行。
- 保留已有 >=5 配对样本、spread-limit 和 margin gate。未新增 P95 gate；真实服务 P95 与连续 48 层 CPU 交接仍需下一轮观测。

## 目标机复测

修改源码后重新生成 fixture，避免 provenance 过期：

```bash
python3 tools/qwen38_cache_tests/bottleneck_review_20261022/prepare_fill_cases.py
PYTHONPATH=python python3 test/registered/kernels/ops/moe/test_qwen38_hybrid_plan.py -f
```

`run_fill_gate.sh` 接收目标机 repo、已有容器 image、manifest、CPU library、物理核列表、source mode、输出目录七个参数，按原始配置默认使用 11 对样本、4 次预热、5% margin，分别测 M3 与 M1/2 并合并为 schema 2 calibration。GPU 权重 pack 与 CPU library 由显式路径挂载；脚本不部署表。若原测量使用其他 margin 或 tuning，需保持一致。

```bash
bash tools/qwen38_cache_tests/bottleneck_review_20261022/run_fill_gate.sh \
  /target/sglang IMAGE /target/pack/manifest.json /target/cpu.so \
  2,3,4,5 shared_thp /target/results/fill-gate
```

plan 微基准新增 `--calibration PATH`，可以带真实非空成本表检查扩大后查找成本；其路由仍为 synthetic，不能替代服务 plan 时间。

最终应交错运行当前 schema 2 表、相同实现的空 samples 表以及 legacy_gpu，同输入、分别重启、统一预热。记录 decode / 自然聊天 / 长输入后的 tok/s、MTP accept、ms/sweep、fill bytes 与 P95；只有目标机实测通过后才判断性能收益。CPU 原生时间戳、持久 worker、resident-hit/miss overlap 和 prefill 流水尚未实施。


## 第二轮：CPU 原生完成发布

B12X 设备共算现在通过新增的原生 `qwen38_cpu_run_b12x_publish` / `qwen38_cpu_run_csf_publish` 入口，在写完并校验全部 FP32 输出后直接 release-store 完成序号。ctypes CDLL 调用此时仍释放着 GIL，GPU 不必再等待 native 返回后重新获得 GIL、更新 Python stats、调用 mailbox complete。压缩与普通 scale 路径都覆盖；Triton 保留原 GPU stream publication。失败不推进原生完成序号，Python 仍按 poison -> abort -> complete 的顺序释放等待者。

旧同步 C ABI 2 入口签名不变，但 B12X Python adapter 启动时要求新 publication 符号；显式指定的旧 CPU library 需要重新编译。自动构建的源码哈希缓存会构建新 library。成本指纹现在也覆盖 CPU Python adapter，校准需要使用新 library 和当前代码重新测量。

本轮验证日志在 `validation/native_publish/`：

- C++ 库在本机 Ryzen 9950X / AVX512-BF16 成功构建；B12X 和 Triton 各 30 项独立 scalar / partition / error 检查通过。
- 299 项 CSF/native 数值、分区、fallback、routing 与 malformed-input 检查通过，覆盖新的 CSF publication 输出与旧同步入口逐位一致，错误输入不推进序号。
- Python adapter publication 验证通过：成功输出一致，非法 sequence 拒绝，非法路由/非有限输入不发布成功完成。
- CPU policy / dispatch / resize 88 tests、279 subtests 通过，覆盖 B12X 成功不再走 Python publication、统计更新顺序、失败 poison 后释放、Triton publication。

这次只移除了成功路径上的 native-end -> GIL -> stats -> complete 依赖；publish -> observe、poll 结束获取 GIL、Python request staging、每专家 condition-variable dispatch 尚在。没有目标 GPU merge P95 或无 profiler 吞吐测量，因此不能将报告里的全部 merge 累计时间当作已消除收益。

后续优先级：原生交接时间戳定位剩余间隙；多专家批量 dispatch 减少唤醒/等待；resident-hit / miss fill 流水隐藏搬运；全局 LFU hit 快路径；prefill DMA/计算流水。后几项必须计入额外 launch、同步、FP32 合并和缓存轨迹的影响。

## 第三轮：全局 LFU hit 快路径

已实现 indexed planner 的全命中且无需冷回收快路径，保留精确热度与 heap/global-minimum 维护，跳过成本表读取、路由压缩和准入/淘汰流程。详情、编译资源对照和 GPU 配对复测命令见 `hit_fast_path/README.md`。本轮 CPU 回归 88 tests / 279 subtests 通过，两版 SM120f 编译通过，indexed 资源预算一致；GPU 执行与延迟测量待目标机验证。
