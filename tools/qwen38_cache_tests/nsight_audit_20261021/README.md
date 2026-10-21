# 共算性能验收：独立 Nsight 原始数据复核

2026-10-21，工作区基准 `76406be86465356f92369024a8b1ffabb1306a19`。

审计后的执行策略修改、历史门限问题及验证边界见
[COST_SELECTION.md](COST_SELECTION.md)。下面的 trace 和数据均为修改前结果。
新增真实路由搬运量重放见 [TRANSFER_REPLAY.md](TRANSFER_REPLAY.md)，
其基线为非全局、逐层非共算 LFU；不是优化后吞吐报告。
动态非共算的最佳已测配置对照见 [ADAPTIVE_TRANSFER_REPLAY.md](ADAPTIVE_TRANSFER_REPLAY.md)。
同一全局缓存下 CPU 参与的诊断对照见 [GLOBAL_HYBRID_REPLAY.md](GLOBAL_HYBRID_REPLAY.md)。

**没有通过“共算带来端到端加速”的验收。** 本轮重新导出并分析原始
`.nsys-rep`，未将旧报告或它的统计 JSON 作为分析输入。确认 CPU 尾部等待、
主机 GPU 提交开销、全命中仍走 CPU 协调三类问题；尚未取得优化后的真实
GPU A/B 数据，不能声称已经解决吞吐回退。

## 证据与口径

输入为 `../async_admission_20261021/captures/{baseline,async,detailed}.nsys-rep`。
三份 SHA256 均与提交中的 manifest 相符；来源哈希见 `provenance.json`。
使用本地镜像 `lmsysorg/sglang:nightly-dev-20261018-b234b5c3` 内 Nsight Systems
2026.4.1 CLI 导出 SQLite，原始采集版本为 2026.3.2。

`audit.py` 独立查询 NVTX、CUDA runtime、kernel、memcpy、memset 和 OSRT。
以最后一次 prefill 后第一个 `HybridCache.run` 至最后一个 run 返回为 decode
区间，剔除长 prefill，按进程/线程与 CUDA correlation ID 关联每次 resident
kernel，使用真实时间区间的并集与交集。5424 个 detailed 层调用均唯一关联到
一个 resident kernel；CPU native 每层至多一次。包含所有记录到的 GPU 活动，
并非 SM 利用率。最后一次 run 之后可能仍有异步设备活动，不在这个 host
窗口内；此窗口时长也不等于客户端 decode tok/s。

文件 `baseline.json`、`async.json`、`detailed.json` 是紧凑结果，完整逐层记录
可由同一脚本不带 `--compact` 重建。`median-e8.json` 保存图中实例的绝对
时间戳和关联事件。图只画该主线程提交的设备工作，其他线程活动计入汇总
GPU busy/idle，但未画入此图。

## 实锤一：CPU 计算没有被 GPU 工作充分覆盖

Detailed 的 decode host 窗口为 **5.281363759–10.564451727 秒**，持续
**5.283088 秒**，5424 次层调用。

- 4848 次有 CPU native 计算，其中 **3022 次（62.3%）CPU 晚于本层 GPU
  resident kernel 完成**。
- 主线程 pending result join 累计 **1.055687 秒**。
- 主线程在 join、CPU native 尚未返回、同时 GPU 没有记录活动，三者的交集
  累计 **0.852703 秒**。这是观测到的关键依赖，不能直接当作优化可回收量。
- 4848 次 native 累计 2.266257 秒，仅 0.203146 秒与各自 resident kernel
  重叠；native 包含工作线程等待和可能的调度延迟，不是纯算术 CPU time。
- native 内观测到 **39720 次 `pthread_cond_wait`**，与该区间 distinct
  miss 专家数乘二完全一致。原实现每专家 gate/up、down 各一次派工。
  等待包含实际工作线程计算，不能声称它们全是无用同步。

选取 E=8 层调用中按 host 时长排序的中位实例（序号 1742）：

| 事件 | 相对层入口时间，µs |
| --- | ---: |
| remap | 100.91–213.98 |
| GPU dispatch host | 215.24–465.16 |
| CPU native | 196.57–1026.71 |
| GPU resident kernel | 451.756–510.220 |
| 主线程 result join | 474.91–1038.28 |
| CPU 结果 H2D | 1069.81–1074.92 |
| 层返回 | 1122.13 |

GPU 专家 kernel 约 **58.5 µs**，CPU native 约 **830.1 µs**，GPU 完成后
CPU 又持续 **516.5 µs**。这条依赖链来自原始事件，不是由旧报告推断。

![Median E=8 invocation](median-e8.png)

长尾不能全归因于专家数量：序号 362 只有 E=3，native 却耗时 5.943 ms，
其中两次条件变量等待各约 2.73/2.86 ms。采集关闭了 CPU 指令采样和
context-switch trace，无法判断是线程调度、抢占还是算术/访存。需要补采。

## 实锤二：GPU 提交前的主机成本很大

| detailed decode 指标 | 总量 | 中位数 |
| --- | ---: | ---: |
| remap host | 0.772642 s | 112.94 µs |
| remap 中 CUDA API 之外的时间 | 0.702483 s | 100.80 µs |
| GPU dispatch host | 1.278009 s | 224.44 µs |
| GPU dispatch 中 CUDA API 之外的时间 | 1.182328 s | 206.40 µs |
| resident launch 前、上一 kernel 提交返回后的间隔 | 0.690511 s | 119.70 µs |
| resident CUDA launch 开始至设备 kernel 开始 | 0.062213 s | 5.86 µs |

CUDA API 之外的区间包含 Python/CuTe 参数处理、框架工作、调度、GIL 和探针
等可能因素。原 trace 未细分 `_PreparedSmallMLaunch.run` 及其内部步骤，
因此只能定位到 **提交前的主机路径**，不能实锤某个 Python 函数为全部根因。
也不能把 224 µs host dispatch 错写成 GPU kernel 执行时间或驱动 launch 时间。

以下是按互斥主线程 scope 分解的 **GPU 无活动时间**，合计 3.187175 秒：

| 主线程所在区间 | GPU 无活动时间 |
| --- | ---: |
| remap | 0.410139 s |
| GPU dispatch | 1.010221 s |
| pending result join | 0.886063 s |
| run 内其他步骤 | 0.442579 s |
| 两次 run 之间 | 0.438172 s |

这张表可以相加。它与 CPU native 表跨线程重叠，不能再与 native 时间相加。
即使只把 0.853 秒 join/native/idle 交集理想化为全部消除，保持其他工作不变，
这个窗口也仅约 1.19 倍；不能据此承诺填平约 61→106 tok/s 的差距。

## 实锤三：全命中也付出共算固定成本

Detailed decode 中 **576 次没有 CPU miss/native**：

- 每次都读回 15360 字节 BF16 激活，以及两份 120 字节路由数组；原始
  memcpy 计数分别为 576 和 1152。
- 每次仍经过 CPU worker 的 input fence 和结果 Future，虽然不调用 native。
- 全命中 host 层耗时中位数 **388.0 µs**，GPU 无活动中位数 **236.6 µs**。
- 其中 507 次还上传了 2048 字节 lookup，反映当前映射快照缓存的命中情况。

因此仅优化 CPU 算术不会解决全命中路径。应实测设备侧 all-hit 快路径、
复用稳定 lookup 和减少逐层主机提交；这些修改涉及动态路由、异步 admission
和 graph replay，不能在无 GPU 情况下直接宣告可用。

## 对旧报告的修正

1. `baseline.nsys-rep` 是 `1c44f94100` 的上一版共算，**不是纯 GPU legacy**。
   三份原始报告不能直接完成 106 tok/s legacy 与当前共算的因果对照。
2. 旧 HANDOFF 写 top-k=8，与原始记录不符。M=3 的两份路由读回各 120 字节，
   dtype 为 int32/float32，对应 **top-k=10**；原 trace 确有 E=25/26 的调用。
3. 将长 prefill 与 decode 合并得到约 60% GPU busy 会掩盖 decode：独立
   detailed decode 窗口 busy 只有 **39.67%**。这里包括 kernel/copy/memset，
   仍不是 SM 利用率。
4. 原始 baseline→async 的 decode 窗口为 4.551→4.972 秒，但层调用数也从
   4992→5184 变化；均值约 911.7→959.0 µs/层。接受率和路由不同，不能把
   两个总时长直接当作同工作量 A/B 的加速比。

## 本轮产物与后续实验

生产 runtime/kernel 保持工作区起始版本。已准备的 CPU 多专家实验保存于
`artifacts/qwen38-cpu-batch-20261021/candidate.patch`，未作为修复落入源码。
它通过 B12X 和 Triton 各 30 项独立标量/路由/错误检查，以及 FP32 抵消
回归；14 组 H=2560/I=640 的相同输入对照均 FP32 逐字节一致。

四核 native 微测中 E=8/12/16/24 不均匀路由约 1.42/1.28/1.36/1.38 倍，
但 M=3/E=2、E=6 全共享路由分别约 0.95/0.93 倍，存在回退。M=1/3、
合成权重、无 GPU 竞争，不能证明服务加速。可重现入口为
`test/manual/kernels/moe/bench_qwen38_cpu_batch.py`；原始结果和检查日志保存在
同一 artifacts 目录。它是后续实验候选，不是已验收优化。

下一步需要可访问 GPU 后进行：

1. 同镜像/请求/容量口径下重新采 legacy_gpu、当前共算和候选，分别保留
   无 profiler 吞吐、多轮 A/B、实际 MTP 接受率和路由/miss 分布。
2. Nsight 打开 `--cpu-sampling`，细分 host GPU adapter；查清 worker 长尾、
   Python/CuTe 提交与 GIL/调度成本。新增 `functions.json` 提供细分入口。
3. 针对 all-hit 快路径和 CPU 专家批量执行分别做消融，测结果等待与 GPU
   空闲是否下降，并跑 resize、异步发布、精度和 graph replay 回归。

已给现有 `profile_nsight.py` 增加 `--nsys` / `--functions-trace` 参数；默认
优先从 PATH 找 Nsight，便于在原机器不可访问时迁移采集环境。容器内示例：

```bash
python3 tools/qwen38_cache_tests/nsight_optimization_20261021/profile_nsight.py \
  --output /results --modes legacy_gpu default --annotations --detailed \
  --cpu-sampling --nsys /usr/local/bin/nsys \
  --functions-trace tools/qwen38_cache_tests/nsight_audit_20261021/functions.json
```

原脚本仍依赖指定的模型/缓存挂载和 serving 环境；此命令本轮未执行。
用户确认原 GPU 机器不可访问，本机 `torch.cuda.is_available()` 为 false。

## 离线复核

```bash
nsys export --type sqlite --output /tmp/detailed.sqlite \
  tools/qwen38_cache_tests/async_admission_20261021/captures/detailed.nsys-rep
python3 tools/qwen38_cache_tests/nsight_audit_20261021/audit.py \
  /tmp/detailed.sqlite --output /tmp/detailed-full.json
python3 tools/qwen38_cache_tests/nsight_audit_20261021/plot.py \
  /tmp/detailed-full.json --output /tmp/median-e8.png
```

另外两份报告同法导出并分析；加入 `--compact` 生成当前目录的紧凑 JSON。
脚本检查每层 resident kernel 唯一关联、单主线程、scope 不重叠、GPU idle
分解守恒。格式和 Ruff E9/F/I 检查通过；新采集参数的 `--help` 在依赖镜像
内执行通过。采集脚本本轮没有启动模型服务。
