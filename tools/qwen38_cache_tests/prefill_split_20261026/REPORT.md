# Prefill split 计算与输出组装优化

本轮保持 `prefill_direct_cache=true`、`prefill_reuse_cache=true`、`cache_overlap_policy=auto`。只优化已选中的大块 CUTLASS mixed-hit prefill 的输出组装，命中计算与未命中权重拷贝的并行顺序不变。

## 改动及边界

旧路径：命中子块结果复制到完整 cached 输出 → clone 完整 cached → 未命中子块结果复制到完整 missing 输出 → 分配完整 sum 输出并相加。

新路径：命中子块结果仍复制到独立 cached 输出 → 等待未命中权重就绪 → 每个 missing 子块直接加到 cached 对应行。每个 CUTLASS 子块仍先输出 BF16，再做 BF16 相加；最终转回原输入 dtype，保留 FP16 调用的输出约定。2048-token workspace 上界不变。

只有 prefill、原始输入（非预量化）、非 W4A16 / 非 B12X W4A4 且 M>2048 时使用新路径。该路径明确分配独立完整输出，因此可以安全累加，即使后端复用子块输出 buffer。小输入、其它后端和预量化输入继续保留原有保护性 clone 与合并。全部命中直接返回命中结果，全部未命中仍走原单分支。

减少的是输出 clone、missing 完整输出及最终完整 sum 的组装开销；两套权重的 dispatch、MoE GEMM 和各自的量化仍然存在，不宣称已经合并为一次 MoE。

## 固定输入计算验收

`test/manual/kernels/qwen38_prefill_split_ablation.py` 使用真实原生 E512 pack 的第0层、H2560/I640/top-k10、128个 cache 专家、固定随机无重复 top-k。两臂使用同一输入和路由，旧实现固定到 `a03f36b70d`。权重全部驻留，传输/计划由只读 fixture 隔离；结果只代表拆分计算与组装，不包含 H2D 收益。

每档预热后进行三组 ABBA，CUDA event 和主机 wall time 分别记录。peak_extra_bytes 是实际 allocated 的峰值增量，不是 allocator reserved 显存或整服务峰值。`calculation.json` 是初测，`calculation-verified.json` 为增加相对 L2 检查后的最终验收。基准按生产 ABI 把 layer activation scale 保持为0维，且以只读文件映射按需读取一层，不创建整包匿名源副本。

## 服务对照方法

`run_serving.py` 用同一个冻结 Hermes 请求的25560个 input_ids和原采样参数，仅把输出上限缩为256，以聚焦 prefill。顺序 old/new/new/old，每轮重启并固定 server seed=42；运行后恢复原部署 launcher 和最终源。完整原请求另在恢复后回放，输出上限仍5895、保留自然EOS。

原始 request/response 和完整日志留在 `/home/user/workspace/prod/homelab-llm/models/profiles/prefill-split-20261026`，仓库仅提交脱敏统计和脚本。

## 正确性

扩展原有 `test_prefill_workspace_bound_preserves_tail_with_reused_backend_output`：4101 tokens跨三个子块（尾块5），混合命中/未命中，未加载权重先填NaN、join后才变有效，后端反复复用同一scratch；独立加权和验证每个token，无容差，覆盖BF16/FP16及重复调用。该用例保护子块copy/add混淆、尾块覆盖、未等待fill、输出buffer别名以及最终dtype约定。原有小输入buffer复用和全部命中测试仍保留。

## 最终计算测量

| tokens | 旧/新计算 ms | 计算耗时减少 | 峰值额外 allocated 减少 | 最大相对 L2 误差 |
| --- | --- | --- | --- | --- |
| 984 | 1.5811 / 1.5914 | -0.65% | 0.00 MiB | 0 |
| 2048 | 1.7198 / 1.7112 | 0.50% | 0.00 MiB | 0 |
| 2049 | 1.8860 / 1.8636 | 1.18% | 10.00 MiB | 0 |
| 4101 | 3.2023 / 3.1741 | 0.88% | 20.02 MiB | 0 |
| 8192 | 5.5846 / 5.5314 | 0.95% | 40.00 MiB | 0 |

984/2048未进入优化分支，表内差异作为测量波动对照。GPU数值检查沿用逐元素 rtol=atol=0.02，并额外要求相对 L2 <0.001，旧版重复调用也记录误差作为控制。

## Hermes 输入 ABBA

| 轮次 | 状态 | TTFT s | 输出 tokens |
| --- | --- | --- | --- |
| 0 | old | 6.8723 | 256 |
| 1 | new | 6.8380 | 256 |
| 2 | new | 6.8363 | 256 |
| 3 | old | 6.8258 | 256 |

平均 TTFT：旧 6.8491 s，新 6.8372 s，约 0.17% 正向差异，小于两次旧版基线间的波动，服务层面判持平，不宣称已证明整体加速。4/4完成256预算、cached_tokens=0、retractions=0，无推理错误。

最终保留：大块路径固定输入计算小幅改善，显存峰值降低；完整服务未确认回退。没有修改 auto 阈值或强制其它档位拆分。

## 最终验证与部署

- 路由/组装单测7项通过，decode graph回归9项通过。GPU固定输入30次比较的相对L2均为0，覆盖五个档位。format/isort/定向lint/注册与package检查均通过。
- 恢复原部署启动参数后回放原始 Hermes 请求：25560输入、5895输出，TTFT **6.849 s**，128 token后 **101.26 token/s**，最大流间隔 **0.248 s**；无推理错误、cached_tokens=0、retractions=0、/ready=200。该完整输出的吞吐不与256预算的prefill对照换算成加速比。
- 最终运行时仍为 direct_cache/reuse_cache 开启、auto选路，记录到180次split调用。A/B/C1/C2/U5、MTP/KV精度/context/chunk等原部署设置保留；本轮只改仓库内的输出组装实现。
- 全部原始正文留在私有prod目录；没有推送远端。

复跑固定输入计算需要空闲GPU及同一镜像，开启 `SGLANG_OPT_QWEN38_CUTLASS_PREFILL_TUNE=1`：

```bash
python test/manual/kernels/qwen38_prefill_split_ablation.py \
  --manifest /path/to/native-E512-manifest.json --output /new-results/calculation.json
python tools/qwen38_cache_tests/prefill_split_20261026/run_serving.py \
  --request /private/frozen-request.json --output-dir /new-private-results
```
