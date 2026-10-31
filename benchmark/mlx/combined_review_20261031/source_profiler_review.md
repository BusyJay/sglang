# M4 Pro 工具验证（2026-10-31）

本目录是 profiler 工具的证据，不是当前 M5 全模型 33 tok/s 的瓶颈诊断。

最终源码对应 validated/environment.json 的 source_sha256；完整使用说明见 ../README.profile.md。
真实仓库四层缩小模型：前三 dense、三 KDA、一 MLA、一 routed MoE、MTP；hidden 256、32 experts/top-k 8、BF16 投影、FP16 affine metadata、4/5/6 bit FFN。无算子替身；包 namespace 启动边界绕过未安装 Torch。

- validated/：最终 5 轮交替 baseline/host/sync、各模式 2 次 warmup。总耗时中位数 32.489 / 38.289 / 102.242 ms；host 1.179x、sync 3.147x。不同轮热状态/调度有波动，仅说明仪器干扰，不作为模型速度结论。
- 首次调用在 prepare/base prefill 之后，并非冷启动加载。所有首次和稳态生成 hash 一致，无事件丢失。63 个分模块 DOT 图；图中有 AsType 节点，但未测其单独成本。峰值 183,863,300 bytes（约 175.35 MiB），swap 前后 0。
- unit.log：3 个 registered 单测方法全部通过；本机只替换 Torch-backed CustomTestCase fixture 为 unittest fixture，测试方法、profiler 代码原样执行。这不等于执行完整 CI fixture。
- capture/：两行 verify Metal capture 生成成功（约 395 MiB，保留本机），真实 kernel 逐个成本尚未解析。其源码版本和最终小改动的关系保存在 environment.json，不能称为最终版本完整设备分析。
- metal-system.trace：Xcode 27.2 Metal System Trace 实际录制成功；device/：目标 PID 88621 的 146 个区间，101.475 ms 观测跨度、28.264 ms 区间并集。包括启动/预热和插桩运行；不是稳态利用率。模板 shader timeline/counters 默认关闭，只获得 encoder/device activity。
- device/device.trace.json 可查看真实设备区间；host.trace.json 可查看主机嵌套 span。两者原点不同，不能直接拼接。原始 Instruments 可能含其他进程元数据，便携包排除原始 trace/XML。
- AST/import、py_compile、git diff --check 通过；ruff/black 未安装，未新增依赖。

M5 待验证：当前 checkpoint、完整模型的真实阶段占比、转换 kernel 与 buffer 生命周期、设备队列空隙、kernel 计数/带宽分析、MTP 接受/复用与最终 HTTP 吞吐。工具不会用 M4 结果推算 M5。

早期 smoke/、smoke_v2/、final_smoke/ 是开发迭代证据，最终交付以 validated/ 为准。首次开发运行暴露 staticmethod wrapper 绑定错误，已修复并加入异常恢复测试。

profiler-tool.tar.gz 是新增工具包；combined-patch.tar.gz 叠加当前已有 dense verify 候选，便于同步到目标机器。manifest 列明基准 HEAD 和实际文件 hash。新增工具不修改生产模型；combined 包的生产改动仅来自先前 dense verify 任务，其测试证据见 ../dense_verify_20261031/README.md。应用前在独立 checkout 执行 git apply --check，已有同等 dense 改动的机器使用工具包，避免重复套用。
