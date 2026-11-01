> 2026-11-01 机制追审修订：本文件是上一版源码的历史记录，不能作为当前源码或“所有优化已完成”的验收结论。4bit gate/up、head 1/3行、mHC 和 stream/down 归因已重新打开；当前结果及未解决项见 `mechanism_revision/RESULTS.md`。

# GLM MLX review 优化采用决定

基准 commit：`fb5d8fe5bad8c9086ea94bfd6341097fe1cd81a0`。M5 Ultra 256 GiB，MLX 0.32.4，GLM-5.3-Flash checkpoint `fa04cbc3622b16aa7475404060b05e506d81bd4e`，原混合 4/5/6bit，TF32=0。保留远端原有 pyproject 改动；没有安装依赖、调整系统 wired-memory、提交或推送。

## 默认采用

| Review 建议 | 最终机制 | 证据与范围 |
| --- | --- | --- |
| 图命中主机开销 | 预建观察路径和参数树；每事务读取标量、成员/容器/数组身份及类 flags；变化时重建 epoch。MLA layer/attention 共用观察。 | M5 scalar/weight/layout/cache 回归通过；完整 checkpoint 命中检查 3KDA 175.97→68.53µs，MLA layer 86.27→41.62µs，attention 54.99→27.14µs。无权重存储副本。 |
| Dense 几何 | K4096/N24896 三行采用 BM4/RPS8；K4096/N154880 九行以上采用 BM8/RPS8；其他保留原几何。 | 原生 lane/reduction 次序保留；M5 实际形状 ABBA 和 dtype/view/NaN/tie 检查。 |
| Routed gate/up 权重加载 | 连续 packed-column 的 5/6bit 采用对齐 uint32 加载，覆盖跨字节偏移；任意 stride 保留。 | M5 packed 数值检查通过；旧4bit候选回退并保留字节加载；新修订已实现4bit两word mask路径，见当前报告。 |
| Prefill 紧凑任务表 | 设备 expert ranges 和有界 worklist，最多 ceil(M/BM)+E−1 项；一次排序与任务表供 gate/up/down 共用。 | 原 NAX 累加及专家内行序不变；M5 worklist/view/rounding 相关检查通过。算子实测 13–30% 缩短不能换算 TTFT。 |
| Prefill down 消除重排 | 新公开 kernel `gemm.mlx_sorted_qmm_scatter` 将4bit NAX结果直接写到原 pick 槽。 | M5 FP16/BF16/stride 和复用任务表数值检查；4bit算子提升约2–18%。5/6bit三轮加载改进仍回退，使用已有 native compact NAX。 |
| Target head token-only | 显式 greedy/no edits/processors/logprob/sanitize 契约；能力变化丢弃尚未消费的未来行；保留 logits 服务契约。 | M5 sampling/cache/私有能力组回归通过；超过17行按有界tile执行。 |
| 多请求 pending MTP head | 同一事务统一提交多个独立请求 head replay，消费后才发布各自状态。 | M5 MTP/runtime/cache 与 HTTP 普通场景；不宣称未测的独立机制并发收益。 |
| MTP4 budget 可观测性 | 记录每 budget/depth proposed/accepted、cycle秒数和 committed tokens；adaptive 策略不变。 | 完整模型 submission scan 包含 MTP4；group4无stream各10样本：MTP2 depth1/2 accepted/proposed为260/270、110/270；MTP4为240/240、120/240、30/240、10/240，深层大量工作未被消费。记录eligible分母避免把条件命中率混作全草稿接受率。未新增固定深度。 |

## 实现、改进后移除的候选

- **Decode grouped-down**：动态分组、静态1/2/3行展开与直接字节加载、原生单行 qdot/匹配行共享三版均 exact。该分拆实现发生回退；旧结果未充分区分加载、跨行计算结构与 dispatch/partial 成本，不能将它们视为唯一原因。完整 checkpoint 记录的实际路由中，MTP2专家匹配组1/2/3行频数为2892/836/548，MTP4为4347/1316/674/205/103（1至5行）；对应样本不是全模型所有请求的统计。实际5bit operand：3行 .2601→.2936ms、2行 .2568→.2598ms、5行 .3177→.3538ms。保留原融合 down/combine；`down_candidates.py` 与 `remote_evidence/route_probe.json` 等保存复现证据。未把实现回退归为 Metal 限制。
- **mHC 一次 RMS**：单组24输出、拆 inverse/projection、8/12输出tile均 exact，但五组 ABBA 编译45层链未胜过原4输出融合；原算子约17–19µs，24输出约30–32µs、拆分约19–21µs。保留原机制，不改 Sinkhorn/模型方程。候选在 `mix_candidates.py`。
- **Shared/routed stream 与 eval group**：保留 KDA/MLA/FFN 编译子图、显式 fused weights 后，完整 checkpoint 五组 ABBA、240例 token hash 一致，2050 prompt/64输出：AR **34.45→27.79 tok/s**、MTP2 **55.12→47.34**、MTP4 **39.17→35.66**。更大 group8/12/16 仍慢。移除新候选，保留原诊断接口，默认 group4。`stream_candidate_glm5_next.py` 保存实现；重新采用需减少额外提交且同模型 tok/s 无回退。

AR lookahead 源码核对：现有 native GLM、稳定批次、无待 prefill/grammar/custom CPU processor、overlap scheduler 时可走已有 lazy-token chain。MTP仍按已消费事务边界发布。没有新增未经证实的未来依赖重叠。

## 吞吐与最终检查

首轮完整模型五组 ABBA，20进程、每场景 baseline/candidate 各10样本、输出128，token hashes 相同：AR26 **35.99→37.19 tok/s**，2050 **33.51→34.42**，8192 **32.41→33.33**；MTP2分别 **61.40→62.30**、**51.36→52.18**、**50.61→51.63**。该轮源码见 `prestream_remote_source.json`，早于最终4bit down与任务表复用，不能冒充最终源码对照。

最后 prefill 整合前 M5 QMM7、runtime59 测试通过；此前 sampling26、release4、auxiliary5、scheduler8 通过。末次任务表复用后仅补 QMM3、model2 相关检查，均通过。完整旧测试不重复运行。最终 HTTP 基准18普通场景与候选一轮对照已完成，退出0、全部128token且输出hash一致。AR aggregate约+0.5%至+3.3%（8K并发4单次-0.42%），MTP2约+0.02%至+1.26%，见FINAL_RESULTS.md。单次样本不宣称稳定收益。

按用户“不要过度测试”要求，停止原计划20 worker HTTP ABBA及剩余32K资源矩阵。已完成基准 AR32K c1/2/4 和服务契约保存，MTP32K中途终止，不计通过。最终版本32K并发、MTP4 HTTP、完整 profile 和独立机制全模型归因仍未验证；已有旧版本证据不替代最终版本。

## 资源与复现

脚本和原始JSON/log均在本目录及 `remote_evidence/`；`current_source_manifest.json` 包含新未跟踪kernel。持久环境 `/Users/jay/workspace/prod/common/sglang-metal/bin/python`，远端 repo `/Users/jay/workspace/sglang`。首次服务启动/模型加载不计入 HTTP 请求吞吐；第一次布局与稳态样本不混称。

完整模型按正常runner进程资源默认值运行。停止旧 orchestrator 后仍存活的 baseline 服务已显式清理；第一次候选采集启动曾短暂与旧服务重叠，健康检查误命中旧端口导致采集失败，结果保留为 `http_failed_g200_candidate.json` 且排除性能结论。最终候选在独立端口30012重跑，只有一个scheduler。系统已有swap，最终记录有增长，不能宣称峰值内存无回退；未采逐请求active/peak。服务退出后无scheduler、memory_pressure free95%，swap1545.75MiB。
