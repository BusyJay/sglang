# 当前线上 B12X 瓶颈 profile（2026-10-18）

采样代码版本：`46cdc10b6f`，位于 `codex/qwen38-main-b12x-20261018`。
镜像、模型、6390 expert slots、MTP=2、B12X 与完整 breakable graphs 均保持当前线上配置。
直接采集已有业务请求，没有重启服务、修改参数或注入测试请求。
本次提交增加采样资料和离线分析工具，没有修改推理实现。

## 结论与建议优先级

1. **优先看 expert cache miss 的 mapped-host fill。** A/C 两次轻量采样中，它占
   target verify GPU 时间窗口的 **38.1%–47.9%**，每个 verify 共
   **7.34–10.94 ms**。其中 **6.47–9.71 ms** 没有与其他已记录 GPU 工作重叠。
   这是时间线观察，不等于消除 fill 就能直接节省同样时长。
2. **B12X 的主机调用路径也值得优化。** 每个 verify 有 48 次
   `b12x::w4a16_small_m_direct_launch`，CPU inclusive 时间稳定在
   **6.17–6.24 ms/verify**；对应 B12X GPU kernel 合计约 **2.96–2.98 ms**。
   带栈 trace 显示 adapter → `run_w4a16_moe` → direct-launch wrapper →
   `compile_plan.__call__`，以及每次 launch 的指针包装/参数准备。
   先评估固定几何下的 prepared-launch 快速路径及重复检查、scale refresh 的成本。
   保持 cache slot 替换、alpha 更新、stream/event 依赖正确。
3. **MXFP8 dense GEMM 是下一块显著 GPU 工作。** 约 240 次/verify，
   kernel 时长相加为 **4.15–4.47 ms**，区间去重后 **3.15–3.42 ms**。
   可继续定位哪些 attention/GDN/shared 投影适合小 M kernel 或合并 launch。
4. **planner 和 HyperConnection 还有中等规模开销。** planner 48 次/verify，
   **1.25–1.43 ms**；HC kernel 去重后 **1.54–1.56 ms**。
   QSA attention/indexer 在这两段采样约 **0.3 ms/verify**。

这些是后续实验方向，本次没有在线应用新的优化或做参数 A/B。

## 采样和主要数据

均为 TP=1 / batch=1 的线上长上下文 decode，附近日志的上下文大约 25k–32k，
MTP accept length 约 1.5–1.8；与迁移时短提示样本的约 2.44 不同。
不要把下面 profiler 时长直接换算为先前 100.79 token/s 的吞吐结论。

| 采样 | 上海时间 | target verify 数 | stack / shapes | CPU verify 均值 | GPU verify 窗口均值 |
|---|---|---:|---|---:|---:|
| A，主要依据 | 16:04:01 | 24 | 关 / 关 | 18.77 ms | 22.84 ms |
| B，调用路径定位 | 16:04:58 | 16 | 开 / 开 | 32.90 ms | 32.67 ms |
| C，轻量确认 | 16:07:49 | 32 | 关 / 关 | 18.00 ms | 19.26 ms |

带栈采样 B 有明显观察开销，GPU active 区间占窗口的比例降至约 54.7%，
主要用于调用路径和实际形状定位。A 中该比例约 97.2%；active 包括等待主机映射
内存 load 完成的 kernel 时间，不能解释为达到 97.2% 的算力利用率。

下面 GPU 时间按 kernel 区间 union 去重；**不同组之间仍可重叠，不可相加为总耗时**：

| target verify GPU 工作 | A：ms / 窗口占比 | C：ms / 窗口占比 |
|---|---:|---:|
| Expert mapped-host fill | 10.94 / 47.9% | 7.34 / 38.1% |
| MXFP8 GEMM | 3.42 / 15.0% | 3.15 / 16.4% |
| B12X expert kernel | 2.96 / 13.0% | 2.98 / 15.5% |
| HyperConnection | 1.56 / 6.8% | 1.54 / 8.0% |
| Expert cache planner | 1.43 / 6.3% | 1.25 / 6.5% |

A 中 fill 与其他 GPU 工作重叠约 1.23 ms/verify，C 中约 0.86 ms。
每个 verify 的 host 开始时间平均间隔分别约 25.27 / 22.51 ms，包含其间的
speculative/scheduler 工作。A 的 draft 和 draft_extend GPU 窗口分别约
1.10 / 1.21 ms；这些也有流水重叠，不能机械相加。

A 的 CPU 记录还有 97 次 `cudaGraphLaunch`/verify，inclusive 约 1.76 ms；
B12X launch、CUDA API、aten 的 CPU 时间有嵌套，不能再相加。
带栈采样中 B12X `ptr` / `make_ptr` 各 9984 次，即 13 次/launch；
`_small_m_direct_supported` 为 1536 次，即 2 次/launch。
`compile_plan.__call__` 是已编译程序的调用路径，函数名和调用次数不代表每次都在 JIT 编译。

## 新请求补采 D 与业务请求边界澄清

第三次采样后的日志空档，经用户确认是业务推理已经结束；用户随后重新发起请求。
之前仅凭 GPU 空闲、没有新 decode 日志和单次 event 同步栈，推断为 scheduler
卡住，依据不足，已撤回该判断。没有重启或取消任何请求。

新请求上于 **16:14:30** 开始采集 16 个 verify，**16:14:32** 导出完成。
导出后 **16:14:33 / 16:14:42** 均观察到新的 decode 日志，分别为
62.42 / 77.68 token/s（前者窗口含采样/导出开销），确认业务继续推进。

D 的 CPU verify 均值 18.19 ms，GPU 窗口均值 22.87 ms；mapped-host fill
11.06 ms/verify，占 GPU 窗口 48.4%，其中 9.78 ms 未与其他已记录 GPU 工作重叠。
MXFP8 GEMM 去重后 3.48 ms，B12X GPU 2.93 ms，B12X launch CPU inclusive
6.04 ms。瓶颈排序与 A/C 一致。

原始数据是 `live-new-request-16steps.trace.json.gz`，统计见
`new-request-analysis.json`，采样前配置见 `new-request-server.json`。

## 为什么普通 HtoD memcpy 统计看不到主要搬运

`python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu` 的
`mailbox_fill_kernel` 直接从 `cudaHostRegisterMapped` 的 immutable expert pack
地址做 `ld.global.cv.v4.u32`，写入 GPU cache slot。
因此这部分跨 host/device 的流量在 trace 中表现为 **kernel**，不是
`Memcpy HtoD`；仅看 copy engine 或 memcpy 汇总会漏掉主要传输路径。
这个 fill kernel 也没有等待 CPU acknowledgement 的自旋循环。

相邻统计区间 **16:03:54–16:04:06**：

- 19200 个 layer invocation，47531 个 miss，385529 个 hit，区间 hit rate 89.02%。
- 平均约 2.48 miss/layer invocation，累计复制 131413708800 bytes。
- 按每个 target verify 48 层折算，约 **313.3 MiB/verify**。

该累计计数差覆盖约 400 个 verify，**不是 A 的精确 24-step 窗口**；
见 `cache-interval.json` 和 `live-cache-stats.json`。
本次没有采集 PCIe/DRAM 带宽、Nsight Compute occupancy 或其他硬件性能计数器，
不能仅凭这些时长证明 PCIe 已饱和，也不能把 Kineto 的 estimated occupancy 当实测值。
下一步可以围绕 miss 数量、cache policy/slot 分配、fill CTA/合并访问和 shared-fill
重叠做控制实验，并结合硬件计数器判断瓶颈。

## Reviewer 建议从这些位置开始

- `python/sglang/srt/layers/moe/qwen38_cache_mailbox.py`：
  `DecodeMissMailbox.run`、planner、`submit_fill` 和 `fill_with_overlap`。
- `python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu`：
  `mailbox_fill_kernel`、`decode_plan_kernel`。
- `python/sglang/srt/layers/quantization/qwen38_cached_b12x.py`：
  `CachedB12xW4A16.run` 的 scale refresh、weights descriptor 和 launch。
- 镜像内 FlashInfer 0.7.2：
  `flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel.py` 的
  `run_w4a16_moe`、`_w4a16_small_m_direct_launch_flat`，以及
  `_lib/compile_plan.py::__call__`。
- `python-b12x-stacks.json` 保留上述函数的调用次数和 inclusive 时间；
  trace B 包含完整 Python 调用树及 shape。例如实际 expert 调用是
  BF16 `[3,2560]`、top-k `[3,10]`，样本首层 cache capacity 已自适应到 256。

## 其他现场观察

`live-service-timing.txt` 保留了采样开始/结束和附近的 decode 日志。
其中有服务启动后加载 `_sparse_gqa_chunk_prefill` 的记录，当时 free device memory
约 0.08 GiB。附近也有低吞吐窗口。本次 trace 没有覆盖该加载瞬间，
无法把停顿归因于它；建议后续单独调查 QSA 新形状懒加载/预热和显存余量。

采样未重启容器，所有 capture 都由 `num_steps` 自动结束。
第三次采样后的 decode 日志空档，经用户确认是该业务请求已经结束，
随后用户发起了新请求。GPU 空闲和没有新 decode 日志不能单独证明 scheduler 卡住；
单次线程栈停在 `copy_done.synchronize()` 也不能证明持续死锁。
新请求已观察到持续 decode，并补采了轻量 profile D。
stack trace 导出需要约 11 秒，轻量 trace 导出约 3 秒，相关时间已记录。
这些 profiler/导出窗口的日志吞吐包含观察开销，不能当线上稳定性能基线。

## 文件与复现

- `live-24steps.trace.json.gz`：A，原始 CPU/GPU trace。
- `live-stacks-16steps.trace.json.gz`：B，原始 trace、完整 Python 栈和 shape。
- `live-confirm-32steps.trace.json.gz`：C，第二次轻量采样。
- `live-new-request-16steps.trace.json.gz`：D，用户重新发起请求后的轻量采样。
- `live-analysis.json`、`stack-analysis.json`、`confirm-analysis.json`：
  每组 kernel/CPU/API 统计、重叠、每步时长，保留原始 kernel 名称。
- `live-server.json`、`stack-server.json`、`confirm-server.json`：采样前运行配置。
- `manifest.json`：来源、采样设置、trace 大小和 SHA256。

将 `.trace.json.gz` 解压为 `.trace.json` 后，可用 Perfetto 的 Open trace file 打开。
建议先看 A/C 的 `step[VERIFY bs=1]` 和 mailbox fill 所在 stream，
再用 B 查 Python 调用树。原始 trace 已随提交保存。

从 repo 根目录重新计算：

```bash
P=tools/qwen38_cache_tests/nightly_migration_20261018/profile
python3 "$P/analyze_trace.py" "$P/live-24steps.trace.json.gz" --output /tmp/live-analysis.json
python3 "$P/analyze_trace.py" "$P/live-confirm-32steps.trace.json.gz" --output /tmp/confirm-analysis.json
python3 "$P/analyze_trace.py" "$P/live-stacks-16steps.trace.json.gz" --output /tmp/stack-analysis.json
python3 "$P/extract_python_frames.py" "$P/live-stacks-16steps.trace.json.gz" --output /tmp/python-frames.json
python3 -m unittest discover -s "$P" -p 'test_*.py'
```

分析按 CPU step 的 External id 关联 GPU annotation，再裁剪/合并区间，
不使用 CPU 提交范围去截断排队的 GPU 工作。关联缺失时直接报错。
4 项离线测试覆盖区间重叠、间隙、CPU/GPU 时间窗口不同及缺失关联；全部通过。

重新采集轻量 profile（需线上请求持续运行）：

```bash
python3 tools/qwen38_cache_tests/profile_service.py start \
  --steps 24 --prefix b12x-live \
  --output-dir /prod/homelab-llm/models/profiles/nightly-b12x-20261018 \
  --metadata /tmp/b12x-live-server.json
```

B 的完整 API 请求保存在 `stack-profile-request.json`。
