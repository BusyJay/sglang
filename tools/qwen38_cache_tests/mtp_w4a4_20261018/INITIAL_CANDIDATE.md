# Qwen3.8 MTP W4A4 本机 GPU 验证（2026-10-18）

默认保持 **W4A16**。W4A4 缩短长 prefill，但没有可辨认的持续解码收益；当前配置的 KV 容量下降约 5.4%，且一轮出现频繁显存映射分配重试，因此没有提升为通用默认。

## 实测汇总

RTX 5090 D / SM120，生产镜像 `sha256:f712e3118c757894f77ebab49081badab75e47fb16523a088435ec1910a37f59`。沿用实际 FP4-Local checkpoint、6390 slots、8192 chunk、MTP=2、context=220000、FP8 KV、静态显存比例 0.90。仅切换 `SGLANG_QWEN38_MTP_ACTIVATION`。

| 指标 | W4A16 | W4A4 |
|---|---:|---:|
| 短上下文持续解码 token/s（6 个样本/模式） | 90.21 | 90.57 |
| 短上下文 TTFT 均值 s | 0.316 | 0.308 |
| 约 32K 上下文持续解码 token/s（4 个样本/模式） | 88.81 | 88.18 |
| 约 32K 上下文 TTFT 均值 s | 6.106 | 4.501 |
| 约 32K 端到端 token/s，包含 prefill | 58.10 | 63.59 |
| 短上下文接受长度 | 2.507 | 2.478 |
| 长上下文接受长度 | 2.412 | 2.424 |
| 自动分配 KV token 容量 | 221248 | 209344 |
| 简单任务检查 | 8/8 | 8/8 |

32K TTFT 降低 26.3%，端到端吞吐提高 9.5%；短 decode 变化 +0.40%，长 decode 变化 -0.71%。这些是合成诊断工作负载的结果，不代表生产请求的固定收益。

## 流程和限制

按 A16/A4/A4/A16 顺序启动四个全新独立 worker；生产语言容器临时停止，ViT 和其他服务保留。使用复制的生产 launch 脚本，固定 seed=103，每轮先 warmup 128 tokens。每个请求先等待 scheduler 空闲并 flush prefix cache；实际 cached_tokens 全为 0。每轮三个短 prompt、两个实际长度 32015/32017 tokens 的重复文本 prompt，各输出固定 1024 tokens，temperature=0、ignore_eos=True，无 profiler。每种模式有 6 个短样本和 4 个长样本，没有筛除慢样本。

持续吞吐用所有 stream 区间的 token 数/区间总秒数汇总，排除首 token 等待；TTFT 为算术均值。端到端包括 prefill。GPU 显存每 100ms 采样，只能代表采样峰值，不能代表瞬时分配峰值。

四轮映射分配重试次数依次为 1/140/1/1；请求均完成，没有 fatal CUDA OOM、请求回撤或测试容器自动重启。第一轮 A4 的重试是已恢复的分配尝试失败，不能称为请求 OOM，但暴露了余量/分配行为的不稳定。两轮 A4 的 KV token 容量均为 209344，低于配置的 context=220000，不能把声明的 context 上限当作可用 KV 容量。

每轮另外执行四个已知答案任务（乘法、反转字符串、Python 表达式、代数），共 16/16。所有文本、元数据和原始 stream 时刻保存在各轮 result.json。A16 两次运行的长文本也不逐字相同，因此没有将 A4 与 A16 长文本不相同直接判为质量下降；4 类简单任务只是 smoke 检查，不能替代完整质量评估。

## 补丁缺陷与修复

原补丁自带 GPU 检查在 graph/eager 比较失败，见 kernel-32.log。去掉重复 expert ID 后一致；同一 token 的重复路由需要在 CUTLASS 前合并。新增单个 Triton 路由准备 kernel：按 expert 合并权重，应用 up/gate 比例，重复/非法 ID 变为 -1 且权重为 0。CPU 参考路径同步更新。

新增 test_gpu_mtp_w4a4.py：独立 CPU 字典参考验证路由准备；在实际 2560/640 专家几何下，对 1/3/128/257 tokens 修改输入、权重和 expert ID 后重复 graph replay，并覆盖非法 ID、重复 ID、全部无效路由和分块边界。

314 项 CPU 回归及实际依赖导入通过，66 项 GPU 回归通过且无跳过；完整 32/512 专家内核检查通过。512 专家时 128/257-token MTP 单层约快 9 倍，但模型的其余层占主要耗时，不能将单层收益当成整体解码加速。

## 复现与工件

- run_ab.py / analyze_ab.py：实际执行的 A/B 脚本和汇总逻辑；launch/ 保存生产脚本副本（测试副本额外固定 random seed）。
- provenance.json / runtime-tested.patch：测试初始源码与镜像信息；最终源码另存 runtime-final.patch 与 final-source-sha256.json。
- round-*/command.json、service.log、gpu-memory.csv、result.json：四轮原始命令、日志、采样和输出。
- comparison.json / validation.json：最终汇总。
- gpu-suite/gpu-small.json、gpu-suite.log、kernel-32-fixed.json、kernel-512.json、cpu.log：回归与内核证据。
- initial-flush-race/ 和 ab-initial-flush-race.log：首次脚本因 stream 刚结束就 flush、scheduler 尚有 pending request 而返回 400 的中断尝试；未计入正式结果。正式脚本使用 flush_cache?timeout=30 等待空闲。

默认开关未修改。需要手动选择时，在模型 worker 设置 `SGLANG_QWEN38_MTP_ACTIVATION=w4a4` 并重启；`w4a16` 显式回退。目标模型的 decode/verify 精度仍由原来的独立开关控制。
