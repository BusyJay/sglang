# FP4-Local MTP 吞吐诊断（2026-10-17）

以下为初次诊断的历史记录。用户明确要求保留 MTP 开启，MTP 默认值现已恢复为 2；
正在验证 verify fused mailbox 修复，最终结果见 `VERIFY_FIX.md`。

当前证据支持：主要问题在 target 的多 token verify。它失去了单 token decode
的 mailbox/fused planner/shared-fill overlap 快速路径，扩大了专家工作集和搬运量；
W4A16 的计算量也随验证 token 数增加。线上接受长度约 1.65，无法摊薄这些开销。
没有发现 packed 记录代表异常或重复报错的证据。没有复现原线上请求的精确 60% 降幅。

## 线上日志与代码对应

原服务为 steps=2 / draft_tokens=3 / 5340 expert slots / pinned PLE / prefill A4、
decode 和 verify A16。16:12:38 的生产日志：context 30976、accept len 1.65、
accept rate 0.33、31.85 token/s。不要混淆 accept length 和 accept rate 的分母。

`qwen38_cache_threaded.py:run` 中，device mailbox、fused decode plan 和 all-hit
fast path 均要求 `topk_ids.shape[0] == 1`。verify 是 3 个 token，不满足。
此时 `during_fill()` 在普通路径上先执行，失去了 mailbox 路径的 shared/fill overlap。

生产累计计数：191328 次 cache layer 调用、636495 次 miss，device gate、fused decode
plan 和 shared-fill overlap 均为 0。关闭 MTP 后，同样的功能恢复：某次累计快照中
三项均为 57167 次。见 `service-before.log` 和 `service-off.log`。

MTP draft 使用的 resident `_project` 在诊断 trace 中约 7.8 ms 总计；target 的
`_cache_project` 约 214 ms 总计，不能把退化简单归咎于 draft 网络太大。

## 为什么 packed_cpu_staging_and_enqueue 特别多

`_fill()` 在 **每个发生 miss 的专家** 上调用 `PackedExpertTransport.fill()`。
该范围包括读取 pack 的 memoryview、等待 pinned ring 可重用、CPU 打包、enqueue H2D
和 packed scatter；它不是一个 token 一条，也不是一整轮 batch 一条。
`qwen38_observation.py` 的默认值是 `SGLANG_QWEN38_OBSERVE=1`，首条及每 256 次调用
打印一次；名称虽写 opt-in，实际默认启用。因此大量专家 miss 会持续刷屏。
日志里的 `calls` 是累计专家调用数，不是打印行数。范围的 host_seconds 不是纯 GPU
拷贝时间，而且可能包含 ring backpressure；不能据此独立判定带宽瓶颈。

生产快照中 packed decode_experts=636495，与 miss 数一致；每专家 payload 2764800
bytes，累计约 1.76 TB。MTP verify 的多 token union 工作集增加 miss，并绕过了
直接从映射主机内存取专家的 mailbox fill，改走 CPU staging/H2D/scatter。
当前 FP4 入口默认 `SGLANG_QWEN38_OBSERVE=0`；可按需显式开启。

## CPU/GPU trace

采集是在生产服务端点上通知后启动的，完成窗口主要由短合成请求驱动，
不是对 31K-context 原生产请求的精确回放。所有 trace 位于部署目录：
`models/profiles/mtp-investigation/`。

| 指标 | MTP=2 | MTP off |
| --- | ---: | ---: |
| 完整 decode/verify steps | 19 | 20 |
| 平均 CPU step span（含 profiler 扰动） | 80.16 ms | 12.70 ms |
| pinned H2D / step | 35.96 ms，约 0.952 GB | 无显式专家 H2D memcpy |
| target `_cache_project` / step | 11.22 ms | 3.49 ms |
| planner `plan_kernel` / step | 5.58 ms | mailbox fused plan |
| mailbox fill / step | 无 | 4.35 ms |

关闭 MTP 后，专家从 host memory 的流量在 `mailbox_fill_kernel` 内发生，
**并不意味着不再经过 PCIe**。不能仅比较 memcpy 事件数判断实际数据传输为零。
MTP 步会产出多个接受 token，以上每步数字不是每 token 数字。两段输入、cache
状态不同，profiler 对 CPU staging 路径也有额外扰动；表格用于确认执行路径和热点，
不能把 80.16 / 12.70 当作吞吐收益比。事件类别存在重叠，不能相加。

## 不开 profiler 的短请求对照

三条相同短 prompt，各生成固定 192 token，temperature=0、ignore_eos=true。

| 请求 | MTP=2 token/s | MTP off token/s | MTP 接受长度 |
| --- | ---: | ---: | ---: |
| CPU cache 英文解释 | 40.25 | 55.57 | 2.34 |
| 中文数据库索引解释 | 38.93 | 52.69 | 2.49 |
| Python LRU 实现 | 39.39 | 50.80 | 2.67 |
| 合计 token / 合计时间 | **39.51** | **52.95** | — |

关闭 MTP 后该组吞吐增加 34.0%；反向表述为 MTP 比 off 低 25.4%。这组接受长度
高于原线上约 1.65，不能据此否定更大幅度的线上退化。需要完全复现 60% 时，应使用
同一长上下文及输出负载做回放。没有对原始生产文本做拷贝或持久化。

限制：两次独立服务进程，cache/adaptive 历史不同；MTP on 是 BF16 GDN state，off
恢复 FP32；on 的观察日志启用、off 关闭。因此这是部署配置的诊断对比，不是严格
隔离单个 MTP 开关的实验，也不是完整质量测试。第一次 off 测试第三条与 profile
重叠，整组排除并重跑，原始文件保存在 `bench-off-overlapped-profile.json`。

## 已落地

- 共享 FP4 入口默认 MTP off，保留 `--mtp 1|2|3` / `MTP_MODE` 的显式选择。
- 默认关闭运行期观察计时；保留 `--profile-nvtx` 对接 Nsight Systems。
- `llmctl profile start --steps 20 --prefix name`：CPU+GPU、自动停止、持久化 trace
  和 server 配置。`stop` 提前结束；无请求时不会自动凑满步数。
- `llmctl profile summary <trace.json.gz> --output <summary.json>`：按完整 DECODE/VERIFY
  span 裁剪，排除 idle/prefill，GPU busy 用每设备区间并集，输出拷贝字节及热点。
- 原入口 `--profile-start` / `--profile-stop` 可直接从容器控制现有进程。
- 新增 stdlib-only `profile_service.py`、`bench_mtp_service.py`，三项 trace 汇总测试通过；
  部署控制器原有七项测试、FP4 preflight、真实 start/auto-stop/summary 均通过。

没有改动运行模型的 frozen source、权重或 pack，也没有修补多 token mailbox 内核。
后续若要让 MTP 真正获益，应先扩展 verify 的 fused planner/device fill/overlap，
再评估多 token W4A16 的 grouped/Tensor Core 实现，最后按真实接受长度算收益。
