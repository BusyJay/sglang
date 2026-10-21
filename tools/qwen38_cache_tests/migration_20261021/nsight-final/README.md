# 最终 CPU 共算配置的 Nsight 采集

基于 `3803f03a70`，2026-10-21（Asia/Shanghai）采集；RTX 5090 D，CPU 核心 2,3,4,5，使用 `calibration-compact.json` 与 `performance-command-compact.json`，没有额外临时 arena。

先执行 1024 输入 / 128 输出预热及 17+25=42 检查，然后通过 CUDA Profiler API 开启采集，记录 16384+64 输入 / 8 输出的 prefill 请求和 1024+64 输入 / 256 输出的 decode 请求。结束后停止采集，再终止独立测试服务。采集进程成功退出，Nsight 报告生成成功。

`default.nsys-rep` 为完整 trace，可直接用兼容版本的 Nsight Systems 打开；`capture.json` 包含源码版本、文件 SHA-256、启动命令及请求指标。采集开启 CUDA/NVTX/OSRT 和 Python sampling，指标包含 profiler 开销，不用于替代前面的无 profiler 性能验收。本次仅采集和校验生成结果，不做详细分析。

复现：`bash tools/qwen38_cache_tests/migration_20261021/run_profile.sh`。只采 default（CPU hybrid）模式；模型、CPU 库和缓存沿用本机现有路径。
