# 原生命中 FC1 提前计算：2026-10-25

已实现、验证并放回当前工作区。完整模型 ABBA 中，短请求中位数两轮均值提高约 **6.75%**，四轮同文本、同 MTP 接受长度的自然问答提高约 **4.79%**。长 thinking 与 210K 场景结果混合，不宣称普遍加速。新路径仍由 `decode_early_cache` 显式开启，正式服务按原配置恢复，默认关闭。

## 执行链

- plan 前用现有 GPU lookup 记录 hit slots；LFU、CPU 选择和 cache 所有权规则保持原样。
- 缺失权重在 fill stream 搬运；compute stream 沿用 shared-expert overlap，并执行命中专家的原生 FC1。
- fill join 后，补算其余 FC1，再按原始 route 顺序统一 FC2。复用同一份原生 pair-swizzled FP16 intermediate；没有拆成两份 MoE 输出后改变归约顺序。
- CPU safe-mask 的零权重占位留在 late FC1，保证其 intermediate 已初始化。命中阶段不读取填充中的 slot zero/其他 miss 权重或 alpha。
- 两个阶段使用缓存 CUDA graph，逐次更新输入；按容量/存储/route-weight 绑定缓存，最多 8 份。淘汰时登记跨 stream 缓冲区使用；没有长期保留输入 activation 对象。
- 编译按 capacity bucket 复用，实际 slot extent 仍作运行时 route 边界。没有修改 checkpoint、dtype、MTP、KV、SSM、HiCache 或量化误差预算。

原生 FC2-only 公共接口接受普通 BF16 intermediate；本实现保留原 FC1 内部的 FP16 pair-swizzled 数据和计算，不能直接套公共 FC2-only 的解码方式。旧 Triton early-hit + 完整 late MoE 仍不是此方案。

## 完整模型 ABBA

RTX 5090 D，TP1，同一 FP4 checkpoint、源码与种子 42；HiCache ratio 6、target FP8 KV / draft BF16 KV、NEXTN 两步 / 三 draft tokens，context 220000、memory fraction .92、Mamba slots 10，并发 1。GPU-only serving；CPU LFU 组合只做了局部兼容验证，没有完整 CPU worker ABBA。

四轮均先冷 219064→8，再 flush KV、三项答案检查、warmup、三个 1088→512 短请求、自然问答、两条中文 thinking、210064→1024 和扩容后的短请求。A/B 只改变 loader extra config 的 `decode_early_cache=false/true`，未开 MXFP8/CTA 候选，容量自动策略一致。

单位 token/s；下面为完整 decode，不含 TTFT。均值变化以两轮 baseline 与两轮 candidate 各自均值比较，不是逐请求稳定加速承诺。

| 场景 | A0 | B1 | B2 | A3 | 两轮均值变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短请求中位数 | 108.63 | 119.11 | 118.17 | 113.64 | +6.75% |
| 自然问答 | 109.16 | 113.80 | 114.36 | 108.58 | +4.79% |
| Thinking 0 | 104.20 | 110.06 | 109.34 | 104.16 | +5.30% |
| Thinking 1 | 109.39 | 105.54 | 115.12 | 109.78 | +0.68% |
| 210K → 1024 | 114.89 | 120.19 | 114.64 | 116.98 | +1.27% |
| 扩容后短请求 | 96.89 | 105.60 | 102.43 | 97.24 | +7.16% |

自然问答四轮文本 SHA256 相同、`spec_accept_length=2.048`；这是较干净的固定输出对照。
210K 的 A0/B1/A3 文本相同，但 B2 文本与接受长度不同；接受长度为 2.885 / 2.885 / 2.790 / 2.917，不能将所有吞吐变化归因于早算。短请求与 thinking 也存在文本/接受率差异。Thinking 1 的 B1 较 baseline 慢，完整保留该结果。

4/4 冷 219K 完成，12/12 精确答案检查通过；无 CUDA OOM/崩溃。冷 TTFT 为 42.144 / 40.944 / 40.932 / 40.947 秒，不将首轮偏慢当作本轮 prefill 收益。四轮实际 expert 三档总预算均为 **7161 / 6798 / 6543**；层内 LFU 轨迹和后续 adaptive rebalance 仍随路由变化，没有宣称固定每层命中状态。GPU/CPU 时钟未锁。

候选运行时统计确认 `decode_early_cache_invocations` 非零，baseline 为零。性能请求未启用 profiler。原始文本、MTP 接受率、stream events、冷请求和 server logs 均在 `r0-baseline/`、`r1-hit-fc1/`、`r2-hit-fc1/`、`r3-baseline/`。

## Trace 验证（三表）

SGLang，单层真实 checkpoint 权重，M=3、top-k=10、3 个 miss；真实 mailbox kernel 和 mapped CPU source。独立 trace 为 `real-staged-overlap.trace.json`，不是完整模型 trace，也不用于上述吞吐数字。原模型 210K 三表仍见同级 `pcie_long_context_20261024/trace-triage.md`。

### Kernel table

| Kernel | 此调用 GPU 时间 |
| --- | ---: |
| mailbox fill | 208.86µs |
| 提前 hit FC1 | 26.11µs |
| late FC1 | 14.43µs |
| unified FC2 | 11.42µs |

### Overlap-opportunity table

| 依赖 | Trace 证据 | 结论 |
| --- | --- | --- |
| hit FC1 与 miss fill | 不同 stream；26.11µs hit FC1 全部落在 fill 窗口内 | 已实现实际重叠，不将其重复当作潜在收益 |
| late FC1 与 fill | late kernel 在 fill 完成后提交 | 保持权重就绪依赖，无 device spin 等待 |

### Fuse-pattern table

| 路径 | 本轮处理 |
| --- | --- |
| 两份完整 MoE + 结果合并 | 未采用；复用原生 intermediate，FC2 只做一次 |
| FC1 activation / native intermediate rounding | 保留原 B12X 计算与转换 |
| alpha refresh + late-route preparation | 合为一次准备 kernel，只读取受保护的 hit alpha rows |

单层合成/真实权重 sweep 的 graph 与 eager 对照保留在本目录；部分全 miss 档位持平或略慢。CPU graph 提交优化也贡献收益，不能将整个实现的收益全部归为 PCIe overlap。

## 验证与边界

- 最终 GPU 回归：**33 passed，105 subtests passed**。覆盖 M=1/2/3、全部 hit/全部 miss/混合、跨 token 重复、动态 activation/route weights、CPU 零权重占位、原 planner/分层 planner 行为。
- CPU/namespace/loader/既有 dispatch 兼容：**104 passed，26 subtests passed**。
- 真实 checkpoint + 真正 GPU plan/fill/alpha scatter/NativeFc1Stage：60 次连续 mailbox 回放，含 cache install/eviction，输出逐位等于直接按原专家 ID 执行的原生 FP32 参考。
- 同一 60 次回放经 compute-sanitizer memcheck：**0 errors**。
- Ruff check/format、相关 registered-test validator、CUDA clang-format 20.1.7 和 git diff --check 通过。
- 当前仅原生 scale、B12X、H=2560/I=640、M=1..3。CSF + FP32 原基线在此测试环境出现 DSL 编译错误，本轮没有修复或接入该组合；显式拒绝 CSF 早算，普通 CSF 默认路径保持原状。

最后补充了同步 runtime 缺省属性兼容和 stage cache 冷路径的 stream 生命周期登记；这些不改变测量中的算术或 steady-state 路径，最终 GPU/CPU 检查及 memcheck 均在这两项修改后执行。

## 启用与产物

新路径默认关闭。经用户批准临时停机测试；开始时正式容器已不存在、GPU 0 已空闲。测试结束后仅恢复 `fp4-local` 的 llm 服务，不重启 GPU 1 的 vit；正式 loader 配置不添加 `decode_early_cache`。

显式试用时，在已有 threaded/B12X/native-scale loader 配置中设置 `"decode_early_cache":true`。分层 CPU 配置限 `cpu_hybrid_mode=layered_lfu`；其他 CPU mode 保持原互斥限制。没有新增环境变量。

`serving_ab.py`、`reference-command.json`、`reference-environment.json`、`summarize_serving.py` 可复现服务对照；需要独占 GPU 0 并串行执行。`source-manifest-serving.json` 记录 benchmark 源码，`source-manifest-final.json` 记录最终源码。`hit-fc1-overlap.patch` 只含本轮新增优化；上轮 planner 修改原样保留。
