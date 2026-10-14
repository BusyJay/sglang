# MTP 吞吐与 0.92 显存预算验证（2026-10-14）

补充原 patch 验收缺少的吞吐对照。模型代码与原验收相同，本轮增加可复用的
`mtp_throughput.py`，以及覆盖流式合并 token 计时的 CPU 回归测试。

## 测量条件

5090 D，使用本地 `swallow-sglang:v0.5.21-qwen38-capacity` 镜像、checkpoint
revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` 和 assembled pack。
每组单独启动进程，顺序为 target-before → steps=1 → steps=2 → steps=3 →
target-after，只清理测试自身创建的进程和容器。

共同配置：5040 总专家槽，`mem-fraction-static=0.92`，threaded runtime、
packed prefill/decode、异步 pack prefetch、GDN projection 预分配、固定专家
缓存（adaptive 关闭）、breakable decode graph（bs=1）、prefill graph 关闭、
chunk=1024、最大并发请求 1、FP8 target KV、BF16 GDN、Mamba 状态池 10。
所有候选都将 context/KV 上限设为 8192；MTP draft KV 为 BF16。
脚本读取 `/server_info` 检查实际生效配置。

输入为相同 token IDs 的中文 Raft 解释问题，长度 512 / 4096。
每个长度先完整生成 512 token warmup，再顺序测量三次；温度 0、seed 42、
`ignore_eos=true`，每次实际输出固定 512 token。每次请求前清空 prefix KV，
保留专家缓存。首次加载、编译和 warmup 不计入统计。

decode 吞吐使用客户端流式事件：从首个含 token 的事件到末次含 token 的
事件，token 数也减去首个事件的全部 token。这样不会把 MTP 首次成批输出
错误地算入后续 decode。TTFT 为首个含 token 事件的时间；E2E 吞吐包含
prefill 和客户端请求全程。原始事件、输出 IDs、输入哈希均保留。

## 结果

这组输入下 steps=2 的两个长度分别提升 decode 中位数 12.8% / 14.8%，
端到端吞吐提升约 10.6% / 8.1%。steps=3 在 512 输入上与 steps=2 接近，
但 4096 输入的增益降至 5.5%，因此优先选 steps=2 作为后续验证候选。
该结论限于本轮输入和配置，范围重叠的短输入结果不能证明宽度 3 优于宽度 2。

decode 为 tok/s；范围为全部重复的最小–最大值。target 合并前后六个样本，
MTP 每项三个样本，表内为中位数，增益相对合并 target 中位数。

| 输入 token | 模式 | decode（范围） | 增益 | TTFT 秒 | TPOT 毫秒 | E2E tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | target | 36.58（32.15–38.33） | — | 1.838 | 27.34 | 32.51 |
| 512 | mtp-steps1 | 40.11（35.28–42.91） | +9.7% | 1.758 | 24.93 | 35.29 |
| 512 | mtp-steps2 | 41.25（38.91–41.58） | +12.8% | 1.851 | 24.24 | 35.95 |
| 512 | mtp-steps3 | 41.51（36.93–41.91） | +13.5% | 1.791 | 24.09 | 36.31 |
| 4096 | target | 37.10（32.04–37.93） | — | 6.667 | 26.95 | 25.09 |
| 4096 | mtp-steps1 | 41.85（40.87–41.91） | +12.8% | 6.891 | 23.89 | 26.81 |
| 4096 | mtp-steps2 | 42.60（41.98–42.71） | +14.8% | 6.824 | 23.47 | 27.13 |
| 4096 | mtp-steps3 | 39.16（38.23–40.14） | +5.5% | 6.687 | 25.54 | 26.04 |

前后 target 中位数：

- 512 输入：36.74 → 36.41 tok/s，漂移 -0.9%。
- 4096 输入：36.79 → 37.20 tok/s，漂移 +1.1%。

接受率和 verify 长度由三次样本的原始计数合并计算；verify 长度为
`1 + correct_drafts / verify_rounds`，排除首次 prefill 输出。H2D 为周期日志
计数差除以 512，**包含 prefill**，边界可能相差一个 layer invocation，
用于观察传输量，不能当作纯 decode 流量。

| 输入 token | MTP steps | draft 接受率 | 每轮 verify 接受长度 | H2D MiB/输出 token |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 67.8% | 1.678 | 256.13 |
| 512 | 2 | 54.9% | 2.098 | 302.88 |
| 512 | 3 | 45.1% | 2.354 | 310.12 |
| 4096 | 1 | 70.6% | 1.706 | 442.86 |
| 4096 | 2 | 54.6% | 2.093 | 463.87 |
| 4096 | 3 | 43.2% | 2.296 | 497.73 |

一秒 NVML 采样的整轮（启动及请求）峰值：

- target-before：26900 MiB。
- mtp-steps1：29882 MiB。
- mtp-steps2：30164 MiB。
- mtp-steps3：30424 MiB。
- target-after：26900 MiB。

采样峰值不代表 CUDA allocator 的精确峰值。原始重复数据存在波动，
不删去慢样本，也不把这组中位数增益推及所有任务。

## 0.92 容量验证

另行执行未限制 `max-total-tokens` 的 5040 槽 MTP steps=1 canary，0.92 下
完成真实加载、graph 捕获、生成与 `17 × 19 = 323` 探针。实际 KV 容量为
**25,408 token**。这轮使用 direct 传输、Mamba 状态池 15，未开启 projection
预分配，与上面的统一吞吐配置不同，不能混用时延或显存数据。

0.92 在这个配置下解决了 0.90 时 5040 槽 MTP 无法分配 KV 的问题。
25,408 是启动分配容量，尚未以该长度做生成验收；也不能据此声称保留了
原 target-only 的 220K KV 容量。吞吐对照刻意固定 8192 KV，以比较相同
输入、状态精度和专家容量。

## 复现与证据范围

运行入口：`python3 -m tools.qwen38_cache_tests.mtp_throughput`，参数
`--checkpoint`、`--manifest`、`--revision`、`--output`、`--steps 0/1/2/3`。
其余共同配置为脚本默认值；`--steps 0` 为 target-only。

本地完整证据在 `results/mtp-throughput-20261014/`：`campaign.json` 保存
容器命令与退出状态；各候选的 `run/report.json` 保存实际配置与测量指标；
`request-*.json` 保存原始流式事件；`throughput-summary.json` 保存对照汇总。
另保留容量 canary、GPU 时钟/温度采样、执行及汇总脚本、代码 provenance。
最初 readiness 解析失败的启动记录单独保留，修正 HTTP 状态检查后全组重跑，
失败轮不参与吞吐统计。

新增计时测试通过；包含真实模块导入与配置投影的 CPU suite 共 230 项通过。
新脚本的 Ruff、格式、isort 和 diff 检查通过。五轮真实 benchmark 均正常退出。

这是单卡、单问题、两个长度、每进程三次重复的 warm decode 对照。
target 的前后两轮用于观察漂移，不能替代多个独立进程、更多问题和完整
部署负载的性能 campaign。专家缓存随重复请求演化，未清空主机 page cache。
固定长度请求用于吞吐测量；此前短请求与 GPU 数值验收负责正确性证据，
本轮不声称各模式所有输出 IDs 严格等价。长上下文、adaptive 吞吐和更高
并发仍未覆盖。MTP 默认继续关闭。
