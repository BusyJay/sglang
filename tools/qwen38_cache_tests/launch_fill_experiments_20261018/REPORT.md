# Launch/fill 实验选择（2026-10-18）

最终选择：`SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH=0`、
`SGLANG_QWEN38_FILL_CTA=8`、`prefill_copy_mode=auto`。
继续使用 nightly 镜像和既有 B12X W4A16 后端。包内实验代码保留，准备式调用
维持 opt-in；本次没有发现同时改善短提示与长上下文的稳定吞吐收益。

用户包的原始提交 `9b09d5682e6fafb5bb38fa85ec4503c69d5d6eb8` 已直接快进合入
`qwen3.8-flash-next-offload`，保留作者、提交和完整历史。包的 SHA256 校验、
bundle 前置提交校验、patch apply-check 均通过。
`results/provenance.json` 记录归档校验和、实际镜像 ID 和 GPU/驱动。
实验使用用户的 `swallow-sglang:nightly-dev-20261018-b234b5c3`，不是作者仅做 CPU
验证的另一份镜像。

## 服务 A/B

每个候选重建 3 次语言 worker，轮次顺序正向、反向、正向；每次预热 192 token，
随后三个固定短提示各生成 1024 token，再对固定的 40,017-token 上下文生成
512 token。长请求均检查到 `cached_tokens=0`。ViT 没有重启。
所有版本使用相同 6390 expert slots、MTP=2、8192 prefill chunk、0.90 静态显存比例、
FP8 KV、BF16 Mamba、单请求调度。服务实际 KV 容量为 220608 token。
吞吐取首末流式 token 间的持续 decode 速度，采样期间没有运行 profiler 或 GPU 微基准。

第一组：3 个候选，共 9 次 worker 实验。

| 包内配置 | 短提示中位 token/s（9 样本） | 对照变化 | 长上下文中位 token/s（3 样本） | 对照变化 |
|---|---:|---:|---:|---:|
| prepared=0，CTA=8 | 97.78 | — | 94.28 | — |
| prepared=1，CTA=8 | 100.91 | +3.20% | 90.90 | **−3.59%** |
| prepared=1，CTA=adaptive | 100.91 | +3.20% | 93.57 | −0.75% |

第二组：关闭 prepared，单独隔离 CTA 的贡献，共 6 次 worker 实验。
两组对照分开统计，避免用不同时间段的对照计算增益。

| CTA | 短提示中位 token/s（9 样本） | 对照变化 | 长上下文中位 token/s（3 样本） | 对照变化 |
|---|---:|---:|---:|---:|
| 8 | 100.43 | — | 94.24 | — |
| adaptive | 101.88 | +1.44% | 92.46 | **−1.88%** |

这些是同一包内开关的比较，不是 `bed9c124a2` 与整个新提交的源代码回归比较。
原始输出、每轮结果、MTP 接受长度、TTFT、端到端吞吐见两个 `results/service*/`
目录及 `results/service-summary.json`。原始 warmup/short/long JSON 无损压缩为
`.json.gz`，复算脚本直接读取压缩文件。复算：

```bash
python3 tools/qwen38_cache_tests/launch_fill_experiments_20261018/summarize_results.py
```

## 为什么没有直接启用

准备式调用的隔离 eager 测量有明确收益：1/2/3 token 的单次中位耗时分别从
169.40/178.95/191.37 µs 降到 110.86/120.85/133.50 µs；Graph 重放约为
28/40/52 µs，两条路径基本相同。它省的是 Python 包装和描述符构造，
不改变计算 kernel 或搬运字节数。微基准的事件区间含 CPU 提交空隙，不能将每次节省
直接乘 48 层后当作整机收益；服务中的提交、搬运和计算会重叠。

完整服务没有证明这部分 CPU 节省能稳定缩短关键路径，长上下文反而在三轮中均下降。
这是关闭 prepared 的直接实验依据。GPU fill 仍占很大时间窗口，故仅缩短包装器不足以
承诺吞吐提升；精确解释每轮下降仍需要同路由回放与 CPU/GPU 调度联合分析。

自适应 CTA 的长上下文估算 verify 周期中位数从 25.50 降到 24.55 ms，但 MTP
接受长度中位数从 2.359 降到 2.276，输出 token/s 因而没有同步改善。
该周期用 `accept_length / decode_tokens_per_second` 估算，保留流式边界误差；
它不是独立测得的 kernel 延迟。基线自身跨轮次也会改变输出与路由，所以不能将接受率
变化断言为 CTA 的直接作用。短提示 +1.44% 与长上下文 −1.88% 的组合不足以支持
默认启用。此次保留 CTA=8；没有回退镜像、合并代码或既有 B12X 后端。

## Fill、DMA 与 prefill 的选择依据

`results/fill-prefill-overlap.json.gz` 含 684 个 fill、270 个独立 GEMM 重叠、108 个
prefill 记录，均执行了精确字节检查。计划覆盖连续和碎片化 expert/slot。

| miss 数 | CTA=8（µs） | adaptive（µs） | DMA 含计划回读（µs） |
|---|---:|---:|---:|
| 1 | 88.5 | 88.1 | 102.0 |
| 2 | 149.3 | 134.4 | 189.5 |
| 3 | 221.5 | 207.1 | 276.9 |
| 10 | 678.6 | 608.4 | 910.9 |
| 30 | 2077.6 | 2076.0 | 2638.7 |

预先知道计划的 DMA 省去了路由回读、描述符生成和生产 worker 交接，不是可直接替代
当前 device mailbox 的实现。回读版本仍省略了生产 worker 交接，却已经更慢。
独立 GEMM 与 copy stream 的重叠中位收益只有约 1–4%，且没有实现模型的依赖链。
因此不接入 decode DMA，也不凭这份实验脚本启用完整模型流水线。

整层 1,415,577,600 字节 DMA 实测约 33.35 ms，约 42.44 GB/s。
48 层全量流式传输至少约 **1.60 秒**；要塞进旧 profile 的 26.263 ms 轮次需要
约 **2.59 TB/s**。增加缓冲区不能减少总流量，decode 不选全 expert 整层预取。

prefill 则适合按缺失布局选择：448 个连续缺失 expert，合并稀疏 DMA 约 27.98 ms，
整层约 31.94 ms；448 个碎片化缺失 expert，合并稀疏约 33.88 ms，整层约 31.93 ms。
384 个碎片化缺失时稀疏仍较快（30.26 对 31.92 ms）。这些结果支持保留现有
`auto`，不改为无条件 dense/sparse，也不从独立缓冲区测量直接重调生产 staging 阈值。

## 正确性与显存记录

- 当前镜像 CPU 回归 334 项、prepared binding 单测 2 项通过。
- baseline/prepared 各 3 轮、每轮 24 个 GPU 组合通过独立 dense reference、
  改变输入/路由/scale 后 Graph 重放、交替 stream 与缓存替换检查。
- 补充同张量检查 54 组：容量 32/133/512、M=1/2/3、int32/int64 ID，
  两条路径和各自重复执行均逐位一致。见 `results/paired-launch.json`。
- 全部 15 次长上下文请求完成；没有抛出 `torch.OutOfMemoryError` 或请求失败。
  但每次均出现 **23 次** expandable allocator 的非致命 OOM 映射重试，
  当时尝试新增 20 MiB、设备仅余约 6–11 MiB。部分 QSA Triton kernel 也在服务启动后
  才加载。这说明 8192 chunk 的长 prefill 临时内存仍紧张，不能宣称显存问题已经消失。
  两组所有配置次数相同，不能归因于 prepared/adaptive。
  原始警告、时间和按配置汇总见 `results/memory-warnings.json` 与压缩服务日志。
  本次保持 KV 容量、静态比例与 chunk 一致，没有用缩小容量掩盖 A/B 中的内存压力。

最终配置快照在 `deployment/fp4-local.yml`。复现入口为 `service_ab.py`、包内手动
GPU 脚本及 `qwen38_b12x_launch_paired.py`。这些单请求和固定提示结果不代表所有
业务分布或并发情况；样本规模不足以证明百分之一量级收益的统计显著性。

## 选定配置的 profile

最终独立采样结果和三张分析表保存在 `results/profile/`，不计入上述吞吐结果。
`capture_selected.py` 在至少 64 个输出 token 后启动 16-step CPU/GPU 采样，保留
真实 breakable Graph、MTP 和生产后端。单 rank trace 包含完整的 16 个 VERIFY。

| 时间线项目 | 每次 verify |
|---|---:|
| GPU envelope | 30.96 ms |
| expert fill 区间并集 | **11.88 ms（38.36%）** |
| fill 未与其他已记录 GPU 工作重叠 | 10.55 ms |
| B12X expert kernel | 2.86 ms |
| MXFP8 GEMM 区间并集 | 3.49 ms |
| expert planner | 1.41 ms |

GPU active 区间并集占 envelope 73.96%；同时存在可见的提交空隙。
CPU B12X custom-op 的 inclusive 时间为 11.13 ms，含嵌套与采样开销，不能作为
另一个可直接相加的“可省时间”。这份 with-stack trace 放大了主机开销，
不能用其 34.36 ms 的轮次间隔替代无 profiler 的吞吐测量，也不能与旧 trace
直接解释为服务性能变化。没有采集 PCIe/DRAM 利用率计数器，未宣称链路已饱和。

`results/profile/triage.txt` 是 `llm-torch-profiler-analysis` 对单 trace 生成的
kernel、overlap opportunity、fuse pattern 三张表；`analysis.json` 给出区间并集和
重叠计算。自动 fusion 表识别的是实现族，B12X 本身已有融合计算，不能把整个族的
GPU 占比再当成额外可消除的开销。这里没有 graph-off mapping/formal 双 trace
或依赖正确的流水线验证，重叠机会属于保守时间线观察，不是可保证的加速比。

首次采样控制请求已 HTTP 200 成功，但客户端误将纯文本控制响应按 JSON 解码，
因此本地脚本报错；服务端正常停止、导出完整 trace，生成继续到 1024 token。
解析器已修正并验证 JSON/非 JSON 两种响应，分析复用这份完整 trace，没有重复采样。
`capture.json` 明确记录这个客户端问题与未保存生成响应 metadata 的限制。

最终服务通过文本计算、远程 ViT 图片、工具调用与 OpenAI SSE 验收；三种 preset
预检通过。完整结果与服务恢复状态随提交保存。
