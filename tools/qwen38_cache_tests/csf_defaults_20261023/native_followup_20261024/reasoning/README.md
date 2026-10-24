# 最新自然长推理 trace（2026-10-24）

对应 native planner follow-up 优化 `ff003549ca`，RTX 5090 D / SGLang / Qwen3.8-Flash-Next FP4-Local。

- `steady/reasoning-steady-TP-0.trace.json.gz`：原始 CPU/GPU Chrome trace，生成至少 512 token 后采集 25 forward steps。
- `steady/window.json.gz`：分析用裁剪窗口，保留原始时间戳。
- `steady/request.json`：该请求的完整响应、流式时序和 MTP 元数据；这是 profiler 请求，不用于无 profiler 吞吐统计。
- `summary.json`：去掉首尾不完整轮后的统计，共 23 轮 / 1104 个 target layer planner 调用。
- `command.json`、`environment.json`、`prompts.json`：实际配置及中文自然提示。HiCache 关闭，target FP8 / draft BF16 精度保留。
- `source-manifest.json`、`cuda-source.cu`：实际工作树的 Python 源码 SHA256 和采样时 CUDA 源码；包含先前尚未提交的其他工作，不能视作 clean checkout 的测量。CUDA 后续仅有格式调整。
- `trace-manifest.json`：压缩文件 SHA256、大小与事件数量。

Chrome trace 文件可在 Perfetto 或兼容查看器中打开。结论与限制见 [上层报告](../REPORT.md)。
