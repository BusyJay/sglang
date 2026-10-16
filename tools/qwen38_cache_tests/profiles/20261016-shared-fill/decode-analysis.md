# Shared graph replay 部署后的 profile

raw `/tmp/nsys-root/nsys-report-a959.qdstrm`，capture 本地时间
2026-10-16 09:41:14（Asia/Shanghai）。取回后在独立无 GPU、无网络、
2 CPU / 4 GiB 容器中使用 Nsight Systems 2026.4.1 import/export。
没有线上请求、停止 profiler、重启或调整服务。SHA256 见 metadata.json。

## 结论

05d6feeef7 的 shared 小 graph 路径正常执行，上一轮整段 MoE eager 引入的
大部分提交空隙已消除，shared/fill 的实际重叠也提高。
但相对最早未并发的稳态 capture，尚不能证明净吞吐收益。

各份均记录 6144 次 planner，选择 127 个完整 planner-to-planner step：

| ms/step | 未并发 | 整个 MoE eager | shared 小 graph |
| --- | ---: | ---: | ---: |
| 整步墙钟 | 17.890 | 22.698 | 18.090 |
| GPU 无 kernel/copy 活动 | 1.194 | 8.223 | 1.416 |
| fill kernel 累计 | 5.275 | 3.473 | 5.486 |
| 全部 BF16 GEMV 累计 | 5.324 | 5.412 | 5.500 |
| GEMV/fill 区间交集 | 0 | 0.487 | 0.688 |
| shared GEMV 累计 | — | 0.771 | 0.843 |

相对整段 eager，整步耗时减少 20.30%；窗口吞吐从 44.06 到 55.28 step/s。
相对未并发 capture 的 55.90 step/s，当前耗时仍高 1.11%。输入、上下文、route、
cache 布局并非受控 A/B，不能据此断言并发版本一定更快或更慢。

## 调度修正证据

fill 仍全部位于 stream 33，shared GEMV 与其相交 0.688 ms/step，约为本次
shared GEMV 累计时间的 81.7%。6092/6096 次 fill 与 GEMV 有交集，包含
all-hit 的极短 fill；不能将这个次数当成 miss 次数或 PCIe 吞吐指标。

图 launch 从 50 次/step 变成预期的 98 次（额外 48 个 shared graph）。
`cudaLaunchKernel_v7000` 从整段 eager 的 310 恢复到 166 次/step，
`cudaLaunchKernelExC` 从 481 恢复到 433，`cuLaunchKernelEx` 从 102 恢复到 6。
说明 shared/router 不再每步由 Python 逐 kernel 提交。

以紧接空隙的下一 kernel 分类，routed prefix-sum 前无 kernel 活动从
2.604 降至约 0.103 ms/step；gate/add 前从 0.929 降至 0.185。
这是 GPU 时间线的空隙归类，不是对应 Python 函数的独立耗时。
本次 graph launch API 累计 1.947 ms/step，未并发版本为 1.236；
CPU API 时间与 GPU 执行可重叠，不能直接从 0.688 ms 重叠收益中相减。

## 比较限制与下一瓶颈

>5 us 的长 fill：未并发 3902 次、整段 eager 3013 次、本次 4015 次。
本次比未并发多约 2.9%，平均长 fill 172.17 us（未并发 171.25 us），
并非等 miss expert 数/等搬运字节的比较。长 fill 也只能近似识别含 miss 调用。

当前最大两项依然是 fill 5.486 ms、普通 GEMV 5.500 ms，planner 为约
1.397 ms/step，expert grouped GEMM 为约 1.305 ms/step。
shared 之外普通 GEMV 还有约 4.657 ms/step，具有原有上下游依赖。
已隐藏约八成 shared GEMV 后，剩余可挤空间有限，应优先分析 miss/fill
及 planner，避免以扩大 eager 范围换取微小重叠。

09:40:54 已完成首次 adaptive evaluation/rebalance，capture 位于其后。
09:41:09 / 09:41:19 stats 均为 1 次 evaluation、1 次 rebalance、丢样 0；
后者 overlap 调用和 device gate 均为 95663。未把迁移暂停或 sampling 窗口
混入本次稳态比较。持续返回结果并不等于完成模型数值对照验收。

KV 仍为 223616 tokens，context_len=220000；日志 graph capture 使用约
0.10 GB，上一次约 0.09 GB，日志有取整，不能当作精确显存分项。
本轮没有缩减 KV 容量。没有修改任何源代码或部署设置。
