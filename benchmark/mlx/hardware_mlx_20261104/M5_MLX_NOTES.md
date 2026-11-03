# M5 / MLX 性能调查：已知机制、目标机边界与下一验证

2026-11-04。只在目标 M5 做后续性能实验；不再运行 M4 benchmark。
历史性能与采用/移除决定见
[优化账本](../remote_validation_20261103/EXPERIMENTS.md)。本轮完成官方资料、
MLX 固定版本源码及目标机 API/SDK 检查，没有运行新吞吐测试。

## 目标与已完成提交

先前要求是 AR 至少 40 tok/s、DFlash2 至少 90 tok/s，主要验证 300-token 短生成。
这些目标仍未达到。“已测候选回退”不能推出“硬件没有优化空间”。
历史账本和原始证据已提交到目标机 `/Users/jay/workspace/sglang` 的 `metal` 分支：
`0f9ee0de0068d74f320a77fa6dbe416428567c5c`，
`[Docs] Record GLM Metal optimization results and exclusions`。
该提交包含 333 个变更文件，360 个档案文件哈希通过检查；已有生产/pyproject 改动
保持未暂存。没有推送到 Git 托管服务器。本地 `4367e938a7` 是此前位置错误的伴随
提交，不能替代目标机提交。

目标机当前报告：Apple M5 Ultra、256 GiB、`applegpu_g17d`，macOS 27.2，
Xcode 27.2 / 27B52，MLX 0.32.4，解释器
`/Users/jay/workspace/prod/common/sglang-metal/bin/python`。
[环境与模型源码哈希](environment_m5.json) 固定调查身份。
目标 revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`，draft revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a`，来自已提交的实验账本。
本轮未加载第二套完整权重，未停止现有服务，未修改系统内存参数或安装依赖。

## 官方资料带来的可检验区别

| 已确认知识 | 对当前问题的意义 | 尚不能得出的结论 |
| --- | --- | --- |
| M5 的 GPU shader core 内有 Neural Accelerator，TensorOps 能使用它；旧 GPU 使用 shader 实现 | 相同 API 在 M4/M5 的执行机制不同，M4 的 TensorOps 时间不能预测 M5 | 不能把 M5 family 演示速度、核心扩展或带宽峰值换算成当前 Ultra 的 tok/s |
| 窄矩阵/单行的算术强度低；大矩阵更有机会利用矩阵硬件 | AR 与每专家少行 verify 需分别检查权重读取和有效 MMA 工作量 | 当前 NVFP4 仍未被计数器证明确实受主存带宽限制；也不能认为强行 padding 会获益 |
| M5 occupancy 管理会考虑寄存器缓存、L1 压力和内存请求 stalls | zero spill 不排除寄存器缓存 thrashing；live-router 完整图可能改变运行时资源压力 | 静态寄存器数、调度 duty 不足以证明 occupancy 或限速原因 |
| TensorOps 可以通过 cooperative tensor 做自定义解码/融合；输入布局须满足对应 operation | 保持当前量化方案的寄存器内解码是一条机制路线 | 不能把接口可用当数值和速度已通过；以前 threadgroup 解码/NAX 候选的回退仍有效 |
| macOS 27 新增 4/8-bit 浮点 tensor 类型 | 需要查当前 SDK 的具体格式、scale 表示与实际 shader lowering | “支持 FP4”不意味着当前 NVFP4 权重可以直接接原生 scale plane |

来源：[Apple M5 ML](https://developer.apple.com/videos/play/tech-talks/111432/)、
[M5 GPU 架构与 occupancy](https://developer.apple.com/videos/play/tech-talks/111431/)、
[macOS 27 quantized TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)、
[Metal 4 TensorOps](https://developer.apple.com/videos/play/wwdc2025/262/)。
[references.json](references.json) 保存抓取日期、URL 与阅读内容哈希。
资料说明的是机制，不是本 checkpoint 的性能证据。

## 目标机 API / SDK 实查

[device_capabilities.mm](device_capabilities.mm) 在 M5 编译执行；仅查询属性，
没有分配权重、提交 GPU 计算或测量性能。
[原始结果](device_capabilities_m5.json)：Apple GPU family 1–10 返回支持，
threadgroup memory 上限 32768 bytes；threadgroup 各维设备上限为 1024，
具体 pipeline 的 `maxTotalThreadsPerThreadgroup` 还可能更低。
推荐工作集 240 GiB，单 buffer 上限 180 GiB。这些不是可持续带宽或安全额外分配预算。

公开 `MTLDevice.counterSets` 只列出 timestamp / GPUTimestamp；stage boundary
sampling 为 true，dispatch/blit boundary 为 false。这只说明当前公开 API 的暴露范围，
不能推出 Instruments / Xcode 没有更丰富的计数器。
目标机有 Metal System Trace 模板；`--show-recording-options` 只返回 Hangs 与
Time Profiler 配置，没有暴露 Performance Limiters 选择项。
本轮没有再采一份缺少计数器的默认 trace 来解释瓶颈。

[SDK 与模型事实](sdk_facts_m5.json) 来自目标机 `MTLTensor.h` 和实际 `glm5_fp4.py`：

- macOS 27.0 起声明 E2M1 FP4、E4M3/E5M2/UE8M0 FP8。
- Host `MTLTensor` scales auxiliary plane 只允许 UE8M0；block factors 为 32、其余维 1。
- 当前 checkpoint 路径使用每 16 值一组 E4M3 scale 和 global scale，传给 MLX 时乘 2688。
- 因而直接把当前码值和 scales 绑定到这个原生 host scale plane 不满足其表示约束。
  这不是所有 Metal FP4 或 inline/cooperative tensor 的限制。
- 可以研究保持原 E2M1/E4M3/global scale 的自定义解码，再送 cooperative input；
  不可为了接 UE8M0/32 静默重新量化 checkpoint。数值、路由和性能均仍待验证。

## MLX 0.32.4 的具体路径，不能按芯片名称猜

源码身份为官方 tag `v0.32.4`，commit
`36969c094354086e15d2a327a9c50feb20b9cf11`。读取已有官方源码 checkout 的
Git object，并以 URL/哈希记录；不是猜测当前主分支或 wheel 的编译选项。

| 源码位置 | 已确认行为 | 对验证的要求 |
| --- | --- | --- |
| `mlx/backend/metal/device.cpp:978` | NAX gate 包括编译期 `MLX_METAL_NO_NAX`、OS 26.2+、架构代际 | 目标机满足 OS/架构条件；installed build 与具体 operator 是否走 NAX 仍用真实 capture 确认 |
| `mlx/backend/metal/quantized.cpp:89` | `g17d` 走 d 分支；vector/matrix 阈值按 K/N 为 32、18 或 12 | 不套用其他 M5 型号的 33/25/13；阈值是 MLX 选择策略，不是 Metal 窄矩阵限制 |
| 同文件 `GatherQMM:2000`、`GatherQQMM:2190` | selector 分别计算 M/B/E；sorted RHS 有额外平均复用门槛；GatherQQMM 的 matrix eligibility 还要求 dtype/global scales/weight rank/对齐 | token 数 × picks 的扁平总数不是 M；不能看到 4×8=32 就声称触发 matrix 路径 |
| 同文件 `gather_qmm:1298`、`gather_qmm_nax:969` | NAX 还查 transpose、K 对齐、FP32/TF32；当前 gather NAX 固定 BK64，BM/BN64 | 不启用 TF32 或改 dtype 来绕开正确性；窄 tile 是实现工作，不是硬件不支持 |
| `mlx/backend/metal/custom_kernel.cpp:14` | library/pipeline lookup 在 GPU eval；源与选项参与 library cache；默认不连续输入可能 copy | 仅创建 kernel 对象不等于编译完成；setup、copy、编码、提交和等待须分开观察 |
| `mlx/compile.cpp:350` | 图 cache 匹配 owner、默认 device/stream、输入数量、rank、dtype、shape、constants；shapeless 放过 shape | strides 没在这里直接触发 graph retrace，但可能导致 backend 布局/copy/不同 kernel；避免把两层 cache 混为一谈 |

源码链接和哈希见 references。动态长度/缓存索引应作为 runtime 参数；
shapeless 不能自动修复读取 shape 的 Python 分支/reshape。
启动预热需要实际 eval 且复用 serving 的 compile owner，记录首次请求和增长边界 miss。
[MLX compile 文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
说明相关限制。已有首请求 TTFT 约 1.3 秒、后续约 .2 秒的观察仍需 setup/JIT 归因。

## 下一实验必须消除的盲区

现有边界：42 层 isolated FFN native/RPS1 为 14.858/14.271 ms；
captured route + shared 为 18.956/17.768 ms；live router 完整 MoE 为
20.429/20.891 ms；完整 DFlash FP8ref/RPS1 为 51.046/49.095 tok/s。
这些来自历史账本，不是本轮重测。geometry、重排、contiguous、stride-aware、
hybrid 和 submission threshold 都已尝试；结果不支持简单恢复旧候选。

下一步优先测导致 captured→live 反转的硬件/图边界，不能以一层热 micro 收益结束。

| 需要区分的原因 | 采集信号 | 有判别力的目标 M5 对照 |
| --- | --- | --- |
| 寄存器缓存或 L1 压力抑制 occupancy，即使 zero spill | occupancy target；register pressure / L1 cache pressure influence；L1 bandwidth/residency；live register 曲线 | 相同真实权重、route producer 与完整依赖的 native/RPS1，先定位最慢 pass，再改变 tile/live range，一次只改一种机制 |
| 散乱大 bank 访问、LLC/MMU/主存请求堵塞 | memory request stalls influence；可用 cache/带宽计数器；相关 pass 的耗时 | hot 与跨 42 bank 轮转；captured/live route 对照并记录 route 是否相同；调 locality 时保持原码值/scales |
| 少行 matmul 实际没有使用矩阵硬件，或无效 tile 过多 | 实际 shader/pipeline 名称、Neural Accelerator utilization、有效 rows 与 tile geometry | 当前 native、已有 threadgroup dequant 候选、保持数值的 cooperative dequant 候选；禁止只测大 GEMM 后推论 AR/DF |
| graph 依赖、输入拷贝、driver/resource scheduling | CPU enqueue/driver、GPU command buffer/pass、buffer identity 和 copy；编码至完成时间 | 从 captured→live 逐步加 producer，但不能随意插同步改变重叠；原始全图与诊断图分开报告 |
| 首请求/持续增长重编译 | graph miss 与 Metal pipeline miss；setup/JIT 时间；TTFT/steady | 同 owner 预热前后及 context/index 容量增长；startup、首请求、300-token 稳态同时记录 |

Apple 官方演示在 Instruments Metal System Trace 选择 **Performance Limiters**，
随后在 Xcode Metal debugger 查看 per-pass counters/cost graph/runtime statistics。
目标机默认 CLI 尚未暴露此选择；需继续找可复现的配置/保存模板方法，确认 trace
内真正有上述计数器，再做因果判断。公开 counterSets 仅 timestamp 不构成停止理由。
对大模型只 capture 有判别力的短片段；避免全模型庞大 capture 与第二套权重驻留。

无论 micro 是否获益，都要继续到真实 route 的跨 bank 完整 MoE，然后同驻留模型
300-token ABBA、HTTP/缓存生命周期验证。记录原始顺序、耗时、内存、数值、route、
草稿接受率和源码哈希；未测项留空，不把改善或失败外推到未测范围。

## 本轮完成与未完成

完成：目标机历史提交位置纠正；官方资料/固定版本源码检查；M5 设备 API、SDK、
Instruments 模板/选项查询。设备 probe 与 SDK 脚本成功，环境 inventory 首版错误地
使用 PATH 查询 Python，已改用 `sys.executable` 并成功重跑。
未完成：Performance Limiters 实际采集、目标机 cooperative 解码实现、
新的 GPU 数值/性能与全模型测试、新旧 checkpoint 因果对照。
没有生产代码改动，也不宣称本轮提高吞吐。

复现元数据 probe 可在目标机运行 [probe.sh](probe.sh)；原始记录保留于同目录的
`device_capabilities.json`、`sdk_quantized_support.json`、`sdk_quantized_details.json`、
`metal_recording_options.json`、`xctrace_templates.txt` 和本地 `tmux/`。
便携文档提交仅包含报告、probe、结构化事实/哈希；不包含权重、虚拟环境、
整份第三方文档/源码、PDF、大 capture 或传输 scratch。
