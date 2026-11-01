> 2026-11-01 机制追审修订：本文件是上一版源码的历史记录，不能作为当前源码或“所有优化已完成”的验收结论。4bit gate/up、head 1/3行、mHC 和 stream/down 归因已重新打开；当前结果及未解决项见 `mechanism_revision/RESULTS.md`。

# 最终源码 HTTP 吞吐对照

M5 Ultra 256GiB，MLX0.32.4，GLM-5.3-Flash 原混合4/5/6bit，checkpoint `fa04cbc3622b16aa7475404060b05e506d81bd4e`，TF32=0。输出128token、greedy、chunk256、context32768、token pool65536、max-running4、deterministic。每场景 baseline/candidate 各一次；按用户要求收敛测试，未完成五组HTTP ABBA，不能宣称统计显著。

Aggregate tok/s = 全部请求实际completion tokens / 批次完成耗时，包含prefill；single-request decode tok/s根据首尾流事件计算。首个请求含首次布局成本，模型加载不计入。

| Prompt | 模式 | 并发 | 基准 tok/s | 最终 tok/s | 变化 |
| --- | --- | --- | --- | --- | --- |
| 26 | AR | 1 | 33.93 | 34.77 | +2.49% |
| 26 | AR | 2 | 61.96 | 62.65 | +1.11% |
| 26 | AR | 4 | 92.80 | 93.30 | +0.54% |
| 2050 | AR | 1 | 20.10 | 20.50 | +2.02% |
| 2050 | AR | 2 | 27.75 | 28.60 | +3.05% |
| 2050 | AR | 4 | 33.68 | 34.80 | +3.34% |
| 8192 | AR | 1 | 5.33 | 5.46 | +2.27% |
| 8192 | AR | 2 | 5.65 | 5.67 | +0.52% |
| 8192 | AR | 4 | 5.85 | 5.82 | -0.42% |
| 26 | MTP2 | 1 | 51.77 | 51.78 | +0.02% |
| 26 | MTP2 | 2 | 64.52 | 64.71 | +0.29% |
| 26 | MTP2 | 4 | 75.24 | 75.54 | +0.40% |
| 2050 | MTP2 | 1 | 25.55 | 25.63 | +0.33% |
| 2050 | MTP2 | 2 | 28.47 | 28.83 | +1.26% |
| 2050 | MTP2 | 4 | 30.56 | 30.58 | +0.05% |
| 8192 | MTP2 | 1 | 5.37 | 5.41 | +0.84% |
| 8192 | MTP2 | 2 | 5.38 | 5.43 | +0.90% |
| 8192 | MTP2 | 4 | 5.41 | 5.43 | +0.43% |

## 单请求 decode 与 TTFT

每模式/长度只有一次请求；TTFT包含prefill，decode排除首个流事件。

| Prompt | 模式 | 基准 decode tok/s | 最终 decode tok/s | 基准 TTFT s | 最终 TTFT s |
| --- | --- | --- | --- | --- | --- |
| 26 | AR | 35.09 | 36.03 | 0.154 | 0.156 |
| 2050 | AR | 32.22 | 33.15 | 2.427 | 2.411 |
| 8192 | AR | 31.45 | 32.40 | 19.956 | 19.539 |
| 26 | MTP2 | 54.99 | 54.96 | 0.157 | 0.161 |
| 2050 | MTP2 | 50.89 | 51.31 | 2.514 | 2.518 |
| 8192 | MTP2 | 57.73 | 58.26 | 21.640 | 21.462 |

## 流事件 ITL

p50/p95在单次请求内部的事件间隔上计算，每间隔除该事件新增token数；MTP批量事件并不提供逐token设备完成时间。多个请求不构成重复的同配置样本。

| Prompt | 模式 | 并发 | 基准 ms/token p50/p95 | 最终 ms/token p50/p95 |
| --- | --- | --- | --- | --- |
| 26 | AR | 1 | 28.48/29.67 | 27.72/28.89 |
| 26 | AR | 2 | 30.51/31.10 | 30.13/30.55 |
| 26 | AR | 4 | 39.64/40.07 | 39.40/39.92 |
| 2050 | AR | 1 | 30.99/32.42 | 29.98/31.46 |
| 2050 | AR | 2 | 35.79/36.46 | 35.44/36.23 |
| 2050 | AR | 4 | 45.53/46.85 | 45.22/46.07 |
| 8192 | AR | 1 | 31.76/32.77 | 30.88/32.06 |
| 8192 | AR | 2 | 36.89/37.74 | 37.03/38.60 |
| 8192 | AR | 4 | 46.53/51.37 | 45.42/48.52 |
| 26 | MTP2 | 1 | 1.05/41.25 | 1.01/41.34 |
| 26 | MTP2 | 2 | 1.51/68.63 | 1.51/68.57 |
| 26 | MTP2 | 4 | 1.71/114.85 | 1.75/118.88 |
| 2050 | MTP2 | 1 | 1.11/47.21 | 0.98/46.83 |
| 2050 | MTP2 | 2 | 1.52/82.45 | 1.42/80.23 |
| 2050 | MTP2 | 4 | 1.73/135.02 | 1.80/143.08 |
| 8192 | MTP2 | 1 | 1.10/47.32 | 0.94/47.20 |
| 8192 | MTP2 | 2 | 46.10/79.83 | 45.81/83.16 |
| 8192 | MTP2 | 4 | 48.07/138.83 | 46.97/150.40 |

## 正确性、源码与资源

18场景全部128token；基准与候选输出hash比较失败数：0。这只证明本次greedy请求输出一致，不能代替所有生成质量、逐层路由、grammar/logprob/cancel/prefix契约。最终相关M5 QMM3/model2检查通过；此前扩展回归保留各自源码范围。

基准swap记录：`vm.swapusage: total = 2048.00M  used = 1274.88M  free = 773.12M  (encrypted)`；最终：`vm.swapusage: total = 2048.00M  used = 1625.75M  free = 422.25M  (encrypted)`。系统已有swap，若发生增长按原始数据报告；无系统wired参数修改。第一次g200端口冲突采集已排除，保留`http_failed_g200_candidate.json`。最终30012重跑只有一个模型服务，模型加载与请求吞吐分开。

早期engine ABBA加载后active为189.11GiB，基准/候选相同；第一组peak约191.84GiB，原始active/cache/peak存于各JSON。最终HTTP没有逐请求active/peak采样，因此不能声称已验证最终峰值内存无回退。

新kernel已纳入 `current_source_manifest.json`；候选HTTP内嵌source哈希。原始数据 `remote_evidence/http_g0_baseline.json` / `http_g200_candidate.json`，脚本 `http_matrix.py`。

## 首轮完整模型 engine decode

五组ABBA、每场景baseline/candidate各10样本，输出128，token hashes一致。源码早于最终4bit down/任务表复用，见`prestream_remote_source.json`。

M5 Ultra GLM-5.3-Flash original mixed 4/5/6-bit, TF32=0, engine decode, output128
| Prompt | Mode | Baseline tok/s | Candidate tok/s | Change | Samples B/C | Token hashes |
| --- | --- | --- | --- | --- | --- | --- |
| 26 | AR | 35.99 | 37.19 | +3.35% | 10/10 | 1 |
| 26 | MTP2 | 61.40 | 62.30 | +1.48% | 10/10 | 1 |
| 2050 | AR | 33.51 | 34.42 | +2.72% | 10/10 | 1 |
| 2050 | MTP2 | 51.36 | 52.18 | +1.61% | 10/10 | 1 |
| 8192 | AR | 32.41 | 33.33 | +2.83% | 10/10 | 1 |
| 8192 | MTP2 | 50.61 | 51.63 | +2.02% | 10/10 | 1 |

## MTP预算和深度

2050 prompt/64输出的完整checkpoint提交扫描，group4、无新stream，各10样本：MTP2 **55.12 tok/s**，MTP4 **39.17 tok/s**。MTP2 depth1/2 accepted/proposed为260/270、110/270；MTP4为240/240、120/240、30/240、10/240。eligible分母单独保留在`mtp_depth_summary.json`，不与全草稿接受率混用。这是早期源码诊断，不是最终MTP4 HTTP结果。

## 交付范围

全部review机制已实现/评估并记录采用决定。默认采用图观察、dense几何、5/6bit packed加载、prefill worklist与4bit直写、target token-only、pending head统一提交和MTP预算统计。grouped down/mHC/shared-routed stream经多轮改进仍有回退，移除候选保留复现证据，详见DECISIONS.md。未改checkpoint/量化/模型方程。

最终32K真实并发、MTP4 HTTP、完整profile及每个独立机制的全模型吞吐归因未重测；不宣称这些已验证。已完成的基准AR32K/服务契约属于基准，MTP32K中断。没有提交或推送。
