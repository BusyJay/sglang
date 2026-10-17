# A16 + MTP 的持续 decode 测量

用户确认后续保持主 MoE decode/verify A16 + MTP。prefill 仍是 A4，
MTP draft 为 resident W4A16。没有切换到 A4 decode preset。

之前的 60.10 token/s 是三条短 prompt 各生成 192 token 的端到端吞吐，
包含 prefill / 首 token 等待。不能将它与历史日志中的 A4 decode 峰值直接对比。
A4 日志确有 74.37 token/s 的历史峰值；不同请求/cache/context 的峰值并非受控基准。

新脚本支持 `--stream`，从首个非空 token 事件到最后一个增长的 token 事件，
用 token 增量 / 时间增量计算 decode 吞吐，并单列 TTFT 与端到端吞吐。
全部原始 token count / elapsed points 保存在 JSON，未保存生成文本。

## 每条生成 1024 token，短上下文，无 profiler

| 请求 | 持续 decode token/s | TTFT 秒 | 端到端 token/s | 接受长度 |
| --- | ---: | ---: | ---: | ---: |
| 英文 CPU cache | 70.98 | 0.249 | 69.84 | 2.53 |
| 中文数据库索引 | 77.22 | 0.291 | 75.63 | 2.60 |
| Python LRU | 65.47 | 0.297 | 64.31 | 2.59 |

合计 token 增量 / 合计 decode 时间约 **70.90 token/s**。这不是瞬时峰值，
也不是原 31K-context 生产负载的回放。文件：`a16-mtp2-stream-cta8.json`。

## 长 prefill 显存修复

第一条 32K 合成请求在 PLE short-conv prefill 处申请 160 MiB 时 OOM；
当时仅剩 156.5 MiB。服务自动重启。原始错误是模型内部显存不足，
benchmark 工具原先因空 SSE stream 再报 UnboundLocalError，现改为明确的
“stream ended without a result” 异常。

`Qwen4ExpPLE._short_conv` 在 cat 完成后释放 `state/padded_seq`，
递归/track state 更新全部入队后，在 gather 之前释放 `conv_input`。
算术和 CUDA stream 顺序不变，inference 下可复用已结束生命周期的分配；
autograd 所需的引用由计算图继续保留。

8192×10240、BF16 的 GPU 检查中：原峰值增量 671334912 bytes，
修复后 335709184 bytes，减少约 320 MiB。输出和 recurrent state 完全一致。
CPU prefill/verify/decode/gradient 四项检查通过。证据：`ple-prefill-memory.json`。

新部署快照 `fp4-local-verify-ple-ab0963e00024` 基于已验收 verify mailbox
快照，仅增加短卷积临时量生命周期修复。220K context、5340 expert slots、
MTP=2 和精度配置保持。

## 32K 重测通过

输入 32015 token，prefix cached_tokens=0，输出 1024 token。
首 token 等待 6.148 秒，持续 decode **65.67 token/s**，
端到端 47.13 token/s，MTP 接受长度 2.46。
4 个 prefill chunk 为 8192/8192/8192/7439。文件：`a16-mtp2-stream-32k-ple-fix.json`。

这是重复合成上下文，不是原始线上 prompt 的回放；不能从中保证每种接受率下
都超过 A4 历史峰值。没有更改 KV token 上限、mem_fraction_static 或上下文配置。
本次没有改动 fill CTA 策略，仍为固定 8；profile 的单层热点不等于端到端收益。
