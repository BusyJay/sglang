# Qwen3.8 migration：实测与收尾（2026-10-21）

已在 `76406be86465356f92369024a8b1ffabb1306a19` 上应用 `qwen38-migration-20261021.tar.gz` 的 69 个文件，并修复 CPU/GPU 交接、校准、启动流切换及 LFU 调度开销。按用户要求停止继续优化并提交；没有增加额外临时 arena，也没有保留试验性的 decode scratch slots。紧凑 prefill 复用已有 arena。

## 性能结果

RTX 5090 D / Ryzen 9950X；CPU 共算显式开启，固定核心 2,3,4,5；原生 FP4 pack；B12X；原始 6388 个名义缓存槽（起始阶段借用后 7006）。同一性能脚本，短请求 1088 输入 / 512 输出 token，170k 场景只生成 8 token。结果为单次验收中的三次短请求，不是跨多次启动的统计置信区间；MTP 接受率与生成路径有波动。

| 场景 | 迁移前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| 短请求 decode 中位数（token/s） | 57.57 | 80.25 | +39.4% |
| 自然对话 decode（token/s） | 60.91 | 76.91 | +26.3% |
| 长上下文后短请求 decode（token/s） | 58.58 | 77.48 | +32.3% |
| 170k 首 token 时间（秒） | 32.85 | 39.70 | **变慢 20.8%** |
| 170k 后 decode（token/s，8 token） | 42.44 | 42.37 | 基本持平 |

不能宣称全面性能提升：长 prefill 回退仍在。结果摘要见 `result-summary.json`；本地原始结果在 `baseline/performance.json`、`performance-compact/performance.json`，未提交原始响应和大日志。

## CPU 分担 1 / 2 / 3 个专家的拐点

使用录制的真实 verify 路由，在相同帧、相同 admitted/rejected 专家状态下配对测试；每个分支 4 次预热、11 次采样，执行顺序打散，测量包含填充、CPU/GPU 工作和结果合并。下表来自最终 compact 配置的 37 组三选一配对帧：

| CPU 专家数 | CPU 共算 / 全 GPU 分支耗时中位比 | 最快帧数 |
| --- | ---: | ---: |
| 1 | 0.948 | 0 / 37 |
| 2 | 0.841 | 4 / 37 |
| 3 | 0.744 | 33 / 37 |

此采样范围内多数帧分担 3 个最好，部分帧分担 2 个更好，不能固定所有路由都用 3。另补测 1/2 token 形状。最终 `calibration-compact.json` 收录 16 个稳定获胜的键；同键所有代表样本均须获胜，并保留实际配对中的最差比例，要求至少 5% 余量，运行时按表选择。未匹配或未稳定获胜的形状回到 GPU。

这是路由分支级证据；尚未完成同一全局缓存实现、同一配置关闭 CPU 的端到端对照。已有不同缓存组织的 GPU-only 测试不能用于隔离 CPU 共算的端到端净收益。

## 已修复的问题

- 原始融合 FC2 epilogue 的逐输出 CPU 完成轮询造成 3.7–5.5ms 交接；改成 FP32 GPU 输出与独立合并，测得完整分支降至约 0.3–0.7ms。
- 校准脚本缺失 manifest、变量遮蔽、清理错误及错误地拼接不同样本的 CPU/GPU 耗时；校准现在严格使用配对样本。
- 图捕获预热切换 CUDA stream 时错误拒绝；通过事件串行交接共享状态，真实 target verify / draft decode / draft extend 捕获通过。
- GPU LFU 使用共享层状态、堆批量更新、indexed 模板专门化，减少调度与启动显存开销；恢复原始缓存预算。
- Host LFU 在每个候选上扫描全缓存；改成每窗口惰性堆，保留原始策略结果，独立回放测得约 24–194ms 降至 0.7–0.9ms。
- Prefill 将缓存命中稀疏复制至已有 arena，合并一次计算；保持分块上限。
- CSF 数值测试消除 BF16 atomic 累加的非确定性干扰，使用确定性参考路径；未放宽误差阈值。

## 剩余瓶颈与验证边界

最终完整测试累计 CPU 调用 1230 次，CPU 专家选择 3690 次，证明确有共算。缓存仍发生约 26.2 万次驱逐、27.2 万次插入，权重源传输约 876GB（基线约 778GB）。剩余重点是强制 GPU 换入导致的缓存抖动与 H2D 流量；它们与长 prefill 回退一致，但尚未通过进一步消融量化各自贡献。旧 Nsight 轨迹还显示 planner、mailbox fill、merge 开销较大；该轨迹早于最后优化，不能当作最终耗时占比。

已完成 118 个 CPU/runtime 测试（284 个子测试）、13 个 GPU 测试（3 个子测试）、295 项 native/CSF CPU 校验。GPU 覆盖独立 LFU 参考回放、1/64/65 行成本表边界、映射完成标志、非连续槽的已有 arena 复制及 CSF 数值一致性。最后稀疏复制修改后重跑 10 个 planner/复制 GPU 测试通过。完整模型短→170k→短验收通过；未完成双轮生命周期语义 canary 或广泛精度评估。CSF 完整模型转换/部署尚未验收，本次性能使用原生 pack。

撤销未验证 scratch 改动后，当前源码与最终校准表 runtime fingerprint 完全匹配：`092fe21811c22705657595ec666fdfa84bcedab521e9892a0a7e26953e7e5ca4`。未重跑长任务。

## 本机复现

`bash tools/qwen38_cache_tests/migration_20261021/run_hybrid.sh` 使用本机既有镜像、模型/cache 路径、`cpu.so` 与最终配置，结果另存 `performance-replay/`。`run_gate.sh LABEL CASES` 默认使用实测 merge 路径；配对输入和测量一并提交。`run_lifecycle.sh` 供后续双轮 canary 使用，本轮没有执行最终双轮验收。

`cpu.so`、模型、原始 traces 和大日志不进入 Git；本机已编译库保留原位。换机器或重编 CPU 库后必须重新校准，源码/库指纹不一致会使旧成本表失效。CPU 库可由 `cpu_nvfp4/build.py` 自动编译，或以其中 FLAGS 编译 `kernel.cpp`；本机 runner 的显式库路径需指向对应构建产物。

收尾检查：`git diff --cached --check`、runner shell 语法及 Ruff E9/F63/F7/F82 通过；全量新增/修改 Python 的 import 排序检查仍有未整理项，未为整理导入改动已校准源码。
