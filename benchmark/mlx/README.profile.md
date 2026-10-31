# GLM MLX 性能诊断

入口 `benchmark/mlx/profile_glm_mlx.py`，仅使用 Python 标准库、现有 MLX / mlx-lm / transformers 和已安装 Xcode。工具不下载模型，不改系统 wired-memory 参数；完整模型沿用正常 serving 的进程级 recommended working set。复用持久虚拟环境；独立诊断进程绕过 Torch 根包启动，但执行真实模型、cache 和 kernel。现有 HTTP 服务未插桩；可在独立单线程 serving replay 中使用 `Recorder.instrument`。

## 先测正常吞吐，再定位成本

```bash
PY=/path/to/existing/venv/bin/python
MODEL=/path/to/local/GLM-5.3-Flash-4bit
REV=fa04cbc3622b16aa7475404060b05e506d81bd4e
$PY benchmark/mlx/profile_glm_mlx.py doctor --output benchmark/mlx/doctor.json
$PY benchmark/mlx/profile_glm_mlx.py run --model-dir "$MODEL" --revision "$REV" \
  --steps 1 --tokens 128 --rounds 5 --warmup 2 --output benchmark/mlx/profile-step1
$PY benchmark/mlx/profile_glm_mlx.py run --model-dir "$MODEL" --revision "$REV" \
  --steps 0 --tokens 128 --rounds 5 --warmup 2 --output benchmark/mlx/profile-step0
```

输出目录必须是新的；`viewer.html` 是无网络依赖的可排序/过滤报告。`results.json` 保存首次调用、交替顺序、多轮原始总耗时、prefill/decode 耗时、生成 token hash、接受/复用统计；environment.json 保存实际源码 hash、HEAD、dirty 状态、设备和 swap。`--revision` 是用户声明，工具不声称自动证明权重属于该 revision。`weights.json` 保存实际加载 tensor 的 shape/dtype/bytes，不保存数值。完整权重大小超过 recommended working set 加 2 GiB 预留时拒绝本地载入；这只是容量保护，仍应预留其他应用内存。

用 baseline 的 median_decode_ms 和 generated_tokens 计算 committed tok/s。host 插桩观测每层 MLP/MoE/KDA/MLA、MTP head、verify、sampling、publish、eval/async_eval、QMM 和融合入口。inputs/outputs 描述符可以定位 FP32 首次出现的模块及实际融合入口形状；returned_none 表示入口未采用该融合 kernel。它不是设备 kernel dispatch 的直接证明，编译区域也可能隐藏内部 Python 调用。

`--sync` 增加独立强制同步归因模式。先 drain 输入和所有在途工作，再执行模块、materialize 输出与显式 cache。该模式改变图复用、生命周期和重叠，不能用于报告正常吞吐。drain_before 包含之前待完成的工作，不能归到当前模块。自耗时仅减去记录的嵌套 span，不等于纯 Python CPU 时间。inclusive 表不可相加；采样 `.item()` 等待记录在 sample_and_materialize 中。

## 检查设备事件与转换

短链诊断和限数量计算图：

```bash
$PY benchmark/mlx/profile_glm_mlx.py run --model-dir "$MODEL" --revision "$REV" \
  --workload verify --rows 2 --rounds 5 --sync --graph --output benchmark/mlx/verify2
```

graphs/ 最多保存 64 个 distinct module/shape 输出图，graph-index.trace.json 对应调用。模型内部 eval、已编译区域可能截断图，不能从“图没有 cast”断言没有转换。权重输入 bytes 只描述操作数大小，不能当成实测 DRAM bytes。将 dtype 路径与 Metal capture 的转换 kernel、buffer 形状和生命周期交叉验证。

```bash
MTL_CAPTURE_ENABLED=1 $PY benchmark/mlx/profile_glm_mlx.py run \
  --model-dir "$MODEL" --revision "$REV" --workload verify --rows 2 \
  --capture --output benchmark/mlx/verify2-capture
open benchmark/mlx/verify2-capture/window.gputrace
```

Capture 在 warmup 后单独执行一次，不纳入计时。generate 模式 capture 会包含一次 prefill 和 decode；优先用 verify 模式限制窗口。GPU capture 会保存 buffer 内容，包括权重，可能非常大，不应作为轻量 patch 包或公开共享的默认附件。微型模型的两行 capture 已约 395 MiB，完整 GLM 不保证小。工具不自动解析私有 gputrace 格式：在 Xcode 看 shader 名称、资源、转换和重复 dispatch。

Metal System Trace 使用现有 Xcode，真实 GPU/encoder 时间和 CPU→GPU 延迟可导出：

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun xctrace record --template 'Metal System Trace' \
  --output benchmark/mlx/glm-metal.trace --time-limit 30s --no-prompt --launch -- \
  "$PY" "$PWD/benchmark/mlx/profile_glm_mlx.py" run --model-dir "$MODEL" \
  --revision "$REV" --tokens 32 --output "$PWD/benchmark/mlx/glm-instruments"
$PY benchmark/mlx/profile_glm_mlx.py metal-export benchmark/mlx/glm-metal.trace \
  --output benchmark/mlx/glm-device
```

该模板当前默认未开 shader timeline/counters；导出 metal-gpu-intervals 是 encoder/device activity，不能声称逐 shader timing、occupancy 或带宽百分比。device.trace.json 与 host.trace.json 时钟原点不同，不直接拼接。device-summary.json 筛选目标 PID，计算区间并集避免重叠重复累计。记录包含模型启动/warmup，不能拿整个录制的 busy 比例当稳态 decode GPU 利用率。完整录制若载入慢可用 `--attach PID`，时间窗从实际工作阶段开始；`metal-export --pid PID` 指定附加进程。原始 Instruments trace 可能含系统其他进程元数据，便携包仅包含筛选后数据。

## A/B 和瓶颈决策

对同一个 checkpoint/revision、请求、context、MTP 步数进行机制 A/B。将两次输出分别保存；交换 checkout 的运行顺序重复，避免温度和预热漂移。

```bash
$PY benchmark/mlx/profile_glm_mlx.py compare benchmark/mlx/before benchmark/mlx/after
```

compare 报告配置差异、正常 decode 时间比例及 token hash。配置不同的比较不能证明机制收益。检查 environment.json 的源码 hash；生成 token 一致只是基本保护，不代替 logits、路由、全 cache 及 HTTP 语义验证。

1. verify/attention/FFN 哪段占成本：先看 host 时间线与 sync 分段，再核对 GPU 区间。
2. 大投影重复读取或 cast：用 graphs、权重清单和 Xcode 资源/dispatch 确认；不能用描述符直接推导读取量。
3. CPU submit 间隙或同步：看 Instruments CPU/GPU 时间线，交叉检查 eval、async_eval、sampling。
4. MTP 步数变慢：比较接受率/复用、每轮 verify 行数和 committed decode 成本，区别更快 verify 与更多废弃工作。
5. kernel 算力/带宽限制：需要设备 counter/shader 分析证据；本工具当前不自动给出 roofline 或占用率。

本工具已提供可用定位链路；尚未实现自动 shader counter 分析、跨进程 HTTP request 关联和 host/device 同一时钟的合并时间线。这些是后续机制能力，不是 Metal 平台“不支持”。M4 缩小模型验证只证明工具工作，不能推算 M5 GLM 吞吐。


## M5 采用结果与稠密共享对照

`combined-patch` 的 M5 全模型、HTTP、数值与 step 选择结果见
[combined_review_20261031/README.md](combined_review_20261031/README.md)。
KDA 输入投影和 BF16 输出头的共享权重 kernel 在兼容形状上默认启用，
不需要 profiler 插桩。普通单请求 decode 不会调用多行共享路径；多请求 decode
的逐层合批仍是独立待接入工作。

隔离旧 FFN/cache 基线、仅 head、仅 KDA 与组合收益（同一模型实例，交替顺序）：

```bash
$PY benchmark/mlx/bench_glm5_next_dense_verify_ab.py \
  --model-dir "$MODEL" --rows 2 3 5 9 --repeats 4 --output-length 128 \
  --output benchmark/mlx/dense-ab.json
```

该对照保留原有 FFN/cache 优化，reference 使用旧逐行 attention 循环。
控制仅在 benchmark 进程内生效，不写入模型共享临时状态或新增生产环境变量。
完整模型 profiler 使用进程级 recommended working set，与正常 serving 一致；
它不会修改系统限制。`compare` 同时核对生成长度，避免把不同 token 预算当成性能改进。
