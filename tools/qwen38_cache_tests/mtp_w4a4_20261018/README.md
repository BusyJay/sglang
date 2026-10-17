# Qwen3.8 MTP W4A4 GPU 验证与容量修复（2026-10-18）

**已将 W4A4 设为默认，并保留 `SGLANG_QWEN38_MTP_ACTIVATION=w4a16` 重启回退。** 容量下降确实是补丁缺陷：A4 多保留了 150 MiB 的尺度副本。修复后 KV 容量从 209344 恢复至 **221248 tokens**，与 A16 完全一致；没有改 slots、静态比例或 KV 预算公式。

## 缺陷与修复

1. **重复路由破坏 graph/eager 一致性。** 原补丁内核测试失败，见 kernel-32.log。同一 token 的重复 expert ID 在 CUTLASS 前必须合并；新增一个 Triton kernel 合并路由权重、应用 up/gate 比例并屏蔽重复/非法 IDs。新增 GPU 回归覆盖独立 CPU 参考、1/3/128/257 tokens、输入和路由改变后的多次 graph replay、全部无效路由。
2. **尺度布局副本挤占 KV。** 原实现保留 checkpoint-layout scale 参数，又为 CUTLASS 分配完整 swizzled 副本。512 专家的两张表共多驻留 150 MiB。A4 在模型加载后只使用 swizzled 布局，故逐专家重排回原参数存储，CUTLASS 直接别名引用。A16 保持原布局，模式切换仍要求重启。GPU 测试验证随机尺度的每个字节与原重排函数一致，且数据指针未改变。

完整 512 专家测量：新增常驻 tensor **5120 bytes**，准备阶段额外峰值 **1638400 bytes（1.56 MiB）**；原来的 150 MiB 副本已消除。

## 实测结果

RTX 5090 D / SM120，现有生产镜像 `sha256:f712e3118c757894f77ebab49081badab75e47fb16523a088435ec1910a37f59` 和 FP4-Local 权重。配置始终为 6390 slots、8192 chunk、MTP=2、FP8 KV、context=220000、mem_fraction_static=0.90。

先完成 A16/A4/A4/A16 四个全新 worker 的 20 个请求，再对容量修复后的 A4 启动新 worker 复测。固定 seed=103、无 profiler、每轮先 warmup；请求前 flush_cache?timeout=30，所有 cached_tokens=0。短 prompt 各输出 1024 tokens；长 prompt 实际为 32015/32017 tokens，各输出 1024 tokens。

| 指标 | A16 基线（2 个 worker） | 修复容量后的 A4（1 个新 worker） |
|---|---:|---:|
| 短上下文样本数 | 6 | 3 |
| 短持续解码 token/s | 90.21 | 88.42 |
| 约 32K 样本数 | 4 | 2 |
| 约 32K TTFT 均值 s | 6.106 | 4.335 |
| 约 32K 持续解码 token/s | 88.81 | 89.82 |
| 约 32K 端到端 token/s，含 prefill | 58.10 | 65.12 |
| 短接受长度 | 2.507 | 2.469 |
| 长接受长度 | 2.412 | 2.438 |
| KV token 容量 | 221248 | 221248 |

容量修复前，两轮 A4 的 32K TTFT 均值为 4.50s；修复后为 4.33s。长 prefill 收益持续存在。持续解码没有可证明的显著提升；短单轮复测约慢 2%，而 A16 自身逐样本为 86.68–101.19 token/s，因此保留全部样本并明确标注样本数，没有筛除慢样本。默认切换的依据是长 prefill 收益、恢复后的相同容量和实际容量验证，不是声称整体 decode 大幅加速。

## 接近容量上限的验证

在另一个全新 A4 worker 上发送 **218015-token prompt + 128-token generation**：完整通过，cached_tokens=0，num_retractions=0，TTFT=31.07s，结束后 health=200。服务日志中 full token usage 达 0.99，见 capacity-fix/near-context/。这验证了实际请求，不只是服务器报告的容量数字。

修复后的常规复测和 218K canary 各只有 1 次已恢复的 allocator mapping 重试，与 A16 基线每轮 1 次一致；没有原候选某轮的 140 次密集重试，没有 fatal OOM 或请求回撤。100ms GPU 采样只能代表采样峰值，不能代替瞬时峰值或 allocator 计数。

## 验证与适用范围

- 容量修复后的 314 项 CPU 回归及真实依赖导入通过；默认切换后再次运行 CPU suite，见 capacity-fix/cpu-default-a4.log。
- 容量修复后的 66 项 GPU 回归全部通过，无跳过，包含随机尺度布局/存储别名、重复/非法路由、graph replay；完整 512 专家内核/显存检查通过。之前的 32 专家检查也保留。
- 初始四轮 16/16 已知答案 smoke 检查通过；修复容量后的 4/4 通过。算术、字符串和 Python 表达式只是 smoke 检查，不是完整质量评测。
- A16 两次运行的长输出也不逐字相同，故未将 A4 的不同长文本直接判作质量下降。所有文本、接受率、stream 时刻和元数据均保存，可复核。
- 该开关只改变驻留 NVFP4 MTP 专家；target decode/verify 精度仍由原有独立开关控制。

## 工件与复现

- `capacity-fix/`：最终修复的 CPU/GPU 日志、kernel-512.json、provenance.json、A4 新 worker 的 command.json/service.log/result.json/gpu-memory.csv，以及 near-context/ 的实际 218K canary。
- `comparison.json` / `capacity-fix/comparison.json`：初始 ABBA 与最终 A4 分别汇总。持续吞吐按 stream token 总数除以 stream 区间总时间；TTFT 取算术均值。
- `INITIAL_CANDIDATE.md` / `initial-validation.json`：容量修复前的诊断结论与证据，供追溯；当前结论以本页和 validation.json 为准。
- `run_ab.py --results DIR --modes w4a4`：在 DIR/launch/ 放入相同生产脚本后复测；near_context_canary.py 复现固定后的 218K 请求。
- `initial-flush-race/`：最早因 stream 刚结束、scheduler 尚未释放请求而立即 flush 返回 400 的中断尝试，不计入正式结果；脚本已改为等待空闲。
- `runtime-tested.patch`：第一轮 GPU/A/B 时的源码；`runtime-final.patch` 与 final-source-sha256.json 记录最终实现。没有修改原始补丁压缩包。

恢复的生产语言服务会使用源码默认 A4；最终 health、实际模式、KV 容量记录在 deployed-default-a4.json。ViT 和其他服务未重建。
