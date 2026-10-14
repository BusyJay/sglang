# Qwen3.8 同卡 MTP（显式开启）

默认不启用。首版支持 SM120、纯文本、单请求、TP/DP/EP/PP=1、threaded 专家缓存、NEXTN，以及可选 breakable decode graph 和 adaptive cache。target pack 格式不变，draft 只读取 checkpoint 索引的 `mtp.*`，保留 FP8/BF16 混合精度；加载后共享 target 的完整 embedding/head 模块。

本分支的逐项 patch 审查、5090 数值测试和真实 checkpoint 短请求验收见
[2026-10-14 验收记录](MTP_PATCH_REVIEW_20261014.md)。这些结果覆盖 eager/fixed、
graph/fixed、graph/adaptive 和 steps=1/2/3；长上下文及正式性能对照仍需另测。

在现有 target 启动命令上添加：

```bash
--speculative-algorithm NEXTN \
--speculative-draft-load-format qwen38_mtp \
--speculative-num-steps 1 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 2 \
--cuda-graph-backend-prefill=disabled \
--cuda-graph-backend-decode=breakable \
--cuda-graph-bs-decode 1 \
--cuda-graph-max-bs-decode 1
```

target 的 `model_loader_extra_config` 推荐使用以下字段，保留现有 manifest 和其他选项：

```json
{
  "cache_slots_total": 3840,
  "runtime_mode": "threaded",
  "adaptive_cache": true
}
```

`cache_slots_total` 是 48 层总预算。3840 等价于平均 80 slots/层，原来的 105×48 为 5040。启动时均分，余数逐层分配；运行时 adaptive 每次调整都用满总预算。分析先优化采样候选容量，同等预测 miss 时优先使用更多槽；剩余槽按容量从小到大补给各层，同容量时按层号分配，可产生不在候选步长上的容量。例如 5 层总预算 42 会分成 9、9、8、8、8，不留下 2 个未分配槽。剩余槽的额外命中收益不计入当轮采样预测。不能同时指定 `cache_slots_per_layer` 或 `layer_cache_slots`；旧配置继续兼容。

steps 可为 1/2/3，对应 verify 宽度 2/3/4。adaptive 模式下，每层初始及候选容量至少为 20/30/40，覆盖完整 active set，分别需要总预算至少 960/1440/1920；过小配置启动时报错。按整个 verify 轮次采样，包括被拒绝候选触及的专家；prefill 不计入采样窗口。

`bs=1` 指一个请求，target 图按 2/3/4 token verify 宽度捕获，保留全部 logits 和 10240 维 HC 状态。draft decode（steps>1）和 draft-extend 使用现有 specialized breakable graph runners。主专家 route/plan、H2D、CUTLASS、slot/event 生命周期仍在 eager break 中执行；rebalance 后读取新的池视图。图回放阶段通过 break 参数保留，不依赖已经退出的 Python scope。prefill 保持 eager。使用 `--cuda-graph-backend-decode=disabled` 可作 eager 对照，不支持 full/tc_piecewise。

draft model/revision 必须与 target 一致，默认跟随。draft KV 默认 BF16，可显式指定 `--speculative-draft-kv-cache-dtype fp8_e4m3` 作对照；GDN state 固定 BF16。不支持 EPD、LoRA 或独立 draft GPU。

BF16 model dtype 是未量化参数和计算输出的默认类型，专家权重仍为 checkpoint 的 block-FP8，不会整层反量化成 BF16，也不在线重新量化成 NVFP4。MTP 原始独立权重约 2.513 GiB；还需另算 draft KV、推测中间态、工作区、graph 与加载后处理峰值。旧配置中 0.10 GiB 的单 token decode graph 预算不能直接沿用到 verify/draft 图。降低总 slots 释放的是专家池容量，确切 bytes 应读 runtime memory budget；是否保留原有 220K KV 容量需实测。

加载审计要求 512 个专家的三个 projection 和 scale 全部存在、FP8/BF16 dtype 与 128×128 shape 正确、scale 有限且正；实际参数写入覆盖全部参数与 stacked/expert slices。未匹配或缺失的必需权重会导致启动失败。

CPU 检查：

```bash
python3 -m unittest tools.qwen38_cache_tests.test_cpu_mtp \
  tools.qwen38_cache_tests.test_cpu_background_transport
```

在已安装本仓库依赖的 5090 环境执行单层 block-FP8 数值检查：

```bash
python3 -m unittest tools.qwen38_cache_tests.test_gpu_mtp_fp8 \
  tools.qwen38_cache_tests.test_gpu_nvfp4_parity \
  tools.qwen38_cache_tests.test_gpu_cache.TestGpuCache.test_threaded_breakable_verify_replays_after_adaptive_migration
```

完整模型 canary（占位路径需替换为已存在的 checkpoint/pack）：

```bash
python3 tools/qwen38_cache_tests/full_model_canary.py \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --manifest /packs/manifest.json \
  --runtime-mode threaded --total-slots 3840 --adaptive-cache \
  --mamba-ssm-dtype bfloat16 \
  --mtp-steps 1 --decode-graph \
  --context-length 32768 --contexts 8192 32768 --chunk 4096 \
  --output /results/mtp-2
```

分别使用 `--mtp-steps 2` 和 `3` 测试更宽 verify。去掉 `--mtp-steps` 得到相同 cache/graph 配置的 target-only 对照。另需与原有最佳 Graph/cache 配置比较实际输出吞吐、TTFT、接受长度、传输量和峰值；本实现不承诺加速。

设置 `SGLANG_QWEN38_STARTUP_MEMORY_DIR` 可记录 draft 构造、加载、后处理等阶段。绑定后的共享 storage 按原有 memory probe 去重。内存探针会增加同步，只用于显存检查，不用于性能对照。

状态提交沿用 EAGLE v2 的 KV/GDN/PLE/QSA 与 HC 路径。完整接受、拒绝、部分接受、连续轮次、EOS/stop/取消、prefix/chunk/page 边界仍需真实 checkpoint 的 GPU 验收；CPU 测试和 synthetic MoE parity 不能代替端到端状态验收。跨卡仍未实现。Graph、rebalance 与模型状态提交的组合需要真实 GPU 验收，不能把 CPU 协议测试当作 GPU 数值证据。
