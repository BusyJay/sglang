# Stage A review：6194db936a

日期：2026-10-13（Asia/Shanghai）。Review 对象为提交 `6194db936af5867d6074093f441b738b5097fb54`，以及 `/data/workspace/qwen38-stage-a-20261013` 中同一 target SHA 的交接包。

结论：**需要修订；暂不冻结正式性能基线、不进入 06。** 本次只 review 并补充文档，没有修改实现。完成以下三项后重新运行 CPU 检查、出包，再做 GPU smoke/correctness 和正式基线。

## 已核验的交付

- 提交只增加 benchmark 工具、配置、测试和交接材料，没有修改 runtime 或启用 06/05/01。
- 重新执行 `python3 -m unittest tools.qwen38_cache_tests.test_cpu_benchmark -v`：21 tests，全部通过。
- 对交接目录执行 `sha256sum -c SHA256SUMS`：全部通过。`package.json` 的 target 与当前提交一致。
- 检查了打包脚本的隔离应用/tree 校验逻辑；本次没有重新执行完整打包流程，其隔离运行结果来自交付的验证记录。
- 对照本地服务实现检查 `/server_info` 和 `/generate` 的累计 output_ids 结构；未在 GPU 上验证服务启动和实际响应。
- CUDA 编译、真实模型、GPU payload/MoE 正确性、性能与 profiling 仍为 `not_run`，本次 CPU 结果不能替代它们。

## R1 [P1] 普通 replay 的正式计时包含逐层 GPU profiling 埋点

位置：[replay.py](replay.py:83)，尤其 85–87、115–118 行。

当前普通 benchmark 无条件为每个 layer event 创建两个 `torch.cuda.Event(enable_timing=True)`，分别 record，并保留到回放结束。它们位于 `started` 和最后 `elapsed` 之间；因此报告的 `wall_s` 包含 event 创建/提交和每层记录数据的成本。仅默认 2048 个 decode token × 48 层就额外引入 196,608 个 timing events，尚未计入 prefill/startup trace。

本项目要比较的恰好是 Python dispatch、event 和少量 copy/kernel launch 的改进。大量额外埋点会改变被测路径，并可能掩盖或改变 patch 的相对收益。称 event 为“辅助指标”不会使其开销从 wall time 中消失；完整模型有单独 profile 模式，但 replay 没有。

修订要求：

1. 普通 replay 默认只保留完整测量区间的墙钟边界与必要的起止同步、计数快照；不逐层创建/record timing events。
2. 单独提供 replay profile 模式或明确的采样选项，记录逐层/逐 token 时间。profile 数据、指纹和汇总与 benchmark 分离，不参与正常速度比值。
3. GPU 前可用 CPU fixture 验证 normal/profile 分支及 event 调用数量；GPU smoke 后补充两种模式的开销对照。
4. 明确 runtime 自身的 `stats_log_interval` 在计时中的作用，避免将高频日志负担误认作搬运成本。若正式 replay 关闭周期日志，应记录实际生效值及边界计数。

## R2 [P1] 环境指纹没有覆盖会改变 GPU 执行行为的继承变量

位置：[core.py](core.py:18) 的 `ENV_KEYS` 与 [runner.py](runner.py:105) 的子进程环境构造。

服务子进程继承 `os.environ`，但 run identity 只记录很小的固定白名单。`CUDA_LAUNCH_BLOCKING` 未被记录也未被清理；例如先正常跑一部分，然后设置 `CUDA_LAUNCH_BLOCKING=1` 续跑，resume guard 仍可能接受相同指纹，却合并执行条件不同的样本。

本地复现：保持 `command()` 返回的硬件信息相同，分别在 `CUDA_LAUNCH_BLOCKING=0` 和 `=1` 下调用 `machine_identity()`；两者 digest 完全一致。无需 GPU 即可确认这项遗漏。

修订要求：

1. 定义明确的 benchmark 子进程环境策略：影响执行的设置要么被规范化，要么被保存并加入指纹；完整模型和 replay worker 使用一致策略。
2. 至少覆盖 CUDA 同步/连接设置、allocator 设置和本实验相关的 SGLANG/缓存开关；阶段 B 的 `SGLANG_QWEN38_ELASTIC_L2` 也必须可显式设置并审计。不要为了补齐指纹将凭证等全部环境变量原样输出。
3. 测试：改变 `CUDA_LAUNCH_BLOCKING` 或相关缓存开关时，应拒绝旧目录续跑，或明确规范化为相同的子进程有效配置；无关变量不应导致虚假的配置变化。
4. 保存有效子进程环境，而不只保存调用者原始环境。

## R3 [P2] replay 续跑不验证已完成 unit 的指纹

位置：[__main__.py](__main__.py:257)。

完整模型 `full_run()` 读取旧 unit 后会校验 `sample.fingerprint`，但 `replay_matrix()` 只 `read(unit.json)`，随后直接纳入 wall 汇总。父 `run.json` 的检查不能证明目录内每个 unit 都属于该实验。

本地使用真实 `replay_matrix()` 的续跑与汇总代码，模拟 identity/trace 校验输入：父 run 是当前实验，`r0-baseline04/unit.json` 设置 `fingerprint=FROM_ANOTHER_EXPERIMENT`、`status=ok`、`wall_s=0.001`。函数返回 0，并把 0.001 汇入 baseline median，没有拒绝。该复现不运行 GPU 或修改实际结果目录。

修订要求：

1. replay 读入旧 unit 时验证 fingerprint、round、candidate、mode/trace 关联及必需结果字段；不匹配或不完整时失败，不能直接使用或静默当成通过。
2. 新 worker 返回成功也要验证 `replay.json` 存在且状态/模式与本次运行相符，避免仅以退出码判定结果有效。
3. 增加 replay 的 foreign fingerprint、错误 candidate/round、结果缺失、合法中断恢复测试。现有 mixed-summary 测试仅覆盖完整模型路径，不能覆盖本问题。

## 非阻塞澄清与 GPU 前检查

- `--warmup-events N` 当前含义是先执行 trace 前 N 项，再从头测量整个 trace；不是丢弃前 N 项再测尾部。这可以作为“重复前缀预热”协议，但需要在 CLI/help/结果中明确标出 measured range，不能据此宣称测量阶段已经排除了 warmup 前缀对应的工作量。若要测生产顺序中的 steady decode，增加独立的测量范围。
- `generate()` 的 `urlopen(timeout=min(timeout, 30))` 给响应头阶段设置了单独的 30 秒上限，随后才用总 deadline。当前服务返回 StreamingResponse，不能仅据此断言长 prefill 一定超时；仍建议明确 header/connect timeout 与 request timeout 的关系，并补充延迟响应头测试。
- 示例 suite 的 mem fraction=0.8 是未验证的启动提议，与历史 0.94 不同。GPU AI 先 smoke；不能静默改低 slots 来启动。任何必要改动应生成新的冻结 suite，并重跑基线。
- transport-only 正确性检查与完整模型数值/语义正确性继续分开，不把 completion/count 检查称为模型语义验收。

## 给实现 AI 的下一任务

> 请基于 6194db936a 修复 STAGE_A_REVIEW.md 的 R1–R3，补充对应回归测试、结果 schema 和 GPU_HANDOFF 说明。不要开始 06/05/01。保留原有测量/失败记录，明确 benchmark/profile 与有效子进程环境。完成后提供修复提交、每条 review 对应的验证证据，并重新生成阶段 A 包；CUDA/GPU 未执行项继续标 not_run，交回 review 后再跑正式基线。

后续涉及 README/review 文档提交时同步维护 `package_stage_a.py` 的 stage A 允许路径；保持白名单精确，不放宽为打入所有未跟踪文件。重新出包必须记录新的 target 和校验和，不能继续使用旧包宣称修复已交付。
