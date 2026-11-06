# GLM FP4-Local：KDA 投影与验证状态优化

2026-11-07。本机 Apple M5 Ultra，256 GiB，MLX 0.32.4。
基准 commit：`12003aa133a311ae72e9f1b9dc81d61e600398db`。
代码已分机制提交：投影链 `2dc2535e0a`，延迟状态 `700d710ab5`，
singleton/pair 测试 `45a6746df1`。未推送；原有 pyproject 改动未修改。

## 默认采用的机制

1. 三个独立 BF16 窄门共享一次 dispatch，权重不拼接；保留原生
   TN=4、shuffle-down，以及窄输出的 BN=8 归约和输出舍入。
2. FP4 KDA 传递独立连续的 QKV、fa、ga、beta；f_b/g_b 共用一次
   dispatch，recurrence 独立读取 beta。避免 24896 列拼接和小门切片 copy。
   保留 checkpoint 的原始权重、量化和所有计算边界。
3. 多位置验证只写最终 FP32 recurrent state，并输出已舍入的紧凑
   K/V、衰减门和 beta。中间状态是依赖这些数据及验证前状态的惰性重放数组。
   重放先将小输入读入 threadgroup memory，避免每个 value 行重复全局读取。

卷积边界仍逐位置保存。正常 decode 不重放；拒绝后的后续计算、显式状态读取、
prefix snapshot/restore 都能物化对应边界。求值标记随 clone、发布、辅助缓存
快照传递，正常更新后清除。普通 HTTP 提交不提前物化只读的中间重放节点；
完整 `cache_state_arrays` 接口仍默认返回全部数组。

以上生产路径默认启用，没有新增环境变量或默认关闭的实验开关。
窄门新 operator 当前覆盖 1..17 行、对齐的 dense GEMV；更大 prefill 的窄门
仍由既有 helper 处理，未声称已解决全部大批量投影。重放对应现有宽度 128、
最多 8 个位置的 recurrence 分段；这些是实现覆盖范围，不是 Metal 平台限制。

## 单请求吞吐：按阶段报告

所有下表请求均单独运行。目标 checkpoint revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`，DFlash2 revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a`；D3，BF16 KV/A16，TF32=0，
eval groups 2/4/2，MLX cache 1 GiB，max MB/ops per buffer 1024/200。
进程 budget 沿用 recommended working set 的 .78，没有调整系统 wired-memory。

第一阶段只比较投影机制 1/2；每条请求输出 192 token，按 191 个 decode 区间
计时，排除 prefill/首 token。每臂预热，顺序 A/B/B/A，每臂两次测量。
总吞吐用总 token 数除总时间，不平均各次 tok/s。

| 单请求场景 | 基准 tok/s | 投影优化 tok/s | 变化 |
| --- | ---: | ---: | ---: |
| 三条短 prompt，顺序运行后汇总 | 71.1987 | 72.8040 | +2.25% |
| 英文解释，20-token prompt | 70.7674 | 72.0325 | +1.79% |
| Python 代码，28-token prompt | 73.6350 | 75.1822 | +2.10% |
| 中文解释，27-token prompt | 69.3275 | 71.3119 | +2.86% |
| 固定 1024-token history | 60.8806 | 62.1083 | +2.02% |
| 固定 2048-token selected history | 58.2228 | 59.3248 | +1.89% |

短请求峰值分配约 173.231 → 173.230 GiB。生成 token、接受数、逐深度接受统计、
offset 和最终完整缓存逐位一致。另有三条 32-token eager 路由诊断，每臂
1302 次选择，专家 ID 和路由权重完全一致；它不冒充编译路径的完整路由 trace。

第二阶段在投影优化已启用的前提下，单独比较状态机制 3；一条 Python prompt，
128 输出 token，每臂等长预热，A/B/B/A。最终源码结果：

| 状态方案 | 单请求 decode tok/s | 峰值分配 GiB |
| --- | ---: | ---: |
| 全边界 eager state | 74.1454 | 172.8061 |
| 紧凑输入 + 按需重放 | 73.9438 | 172.2899 |

吞吐 -0.27%，两次采样/臂不足以证明增益；不把本项计入加速幅度。
采用它的已测收益是约 **0.516 GiB** 峰值分配降低，吞吐短测基本持平。
原始样本分别为 A=[74.1205,74.1704]、B=[73.8664,74.0214]。
生成 token、接受统计和最终缓存均相同。后续仍需把拒绝边界的重放融合进下一次
recurrence，减少额外 dispatch；此项尚未实现。两阶段百分比不能相加。

## 局部证据与被撤回的候选

实际 checkpoint 第 0 层四行投影链 inventory：33 → 3 次 dispatch。
新路径包含 QKV、三窄门、f_b/g_b 三次投影；未将该数字外推为吞吐百分比。
原始微基准见 `micro.json`，它与全模型结果分开保留；`first_eval_seconds`
只是该进程首次求值诊断，不是匹配的完整冷启动 A/B。

第 4 项 singleton 做过完整独立循环和共享循环/条件归约两版：
128-token 单请求短测均约回退 3%，token、接受统计和缓存没有变化。
两版均撤回，最终生产 mixed kernel 与基准相同；没有留下关闭开关。
BF16/FP32 合成微基准有局部收益，但不能替代全模型。
分支改变了编译控制流；寄存器、代码体积、指令缓存和调度的时间份额未分离，
未将其中任何一个猜测写成已确认根因。实验源码及原始数据保留在
`singleton_duplicate.source`、`singleton_conditional.source`、`singleton*/fullmodel.json`
和 `singleton_tune*.json`。

状态机制第一版也曾较慢：重放重复全局读取小输入，并且提交边界会提前物化状态。
已改进为 threadgroup 复用和显式延迟求值，最终结果见上表。

## 验证与已知边界

- 最终 84 项相关 MLX 设备测试通过：GLM、KDA/replay、DFlash。
  覆盖 FP16/BF16、有/无初始状态、各个前缀、9/17 行分段、拒绝后续跑、
  clone、辅助快照恢复和求值标记。窄门测试另逐位对照原生 singleton GEMV，
  含 FP32、BN=1/8、strided 输入和尾行。没有放宽数值阈值。
- registered namespace 模块的 3 项 CPU 测试通过；修改文件 Ruff 诊断
  相对基准无新增。保留已有诊断及无关旧格式，没有安装工具。
- 第一阶段基准 HTTP 的 17 个顺序 case 已保存。按用户后续要求，未继续扩展
  双请求矩阵；中止的 `cohort` 数据排除。
- 最终 HTTP 使用 `http_smoke.py` 做有界单请求验证，结果见 `http_smoke.json`。
  4 个冷/热请求与对应旧版输出、logprob、top-logprob 完全一致，warm 512 命中
  256-token prefix。两个 96-token Python 流式请求的观测 decode 为
  65.31 / 64.07 tok/s，输出 token 与旧版一致。
  HTTP 流式吞吐只作前后观测，不冒充 ABBA 或 engine decode 口径。
- **已有问题**：512-token 整块 prefix reuse 的冷、热首 token logprob 分别为
  -0.1406364441 / -0.2854499817，token 相同。隔离原始 commit 的服务也复现了
  完全相同的值，见 `prefix512_original.json`；优化版相应请求与各自旧版严格对照。
  该已有问题尚未修复，不能声称所有 prefix 情形已实现冷热逐位一致。
- 无匹配的完整冷启动 A/B、8K 全模型质量评测或目标机之外的结果。未用微基准、
  AST/import 检查或其他机器的数字替代全模型验证。原始 swap 读数随各进程保存，
  不宣称系统全程没有其他内存活动。

## 数据与复现

`final_short/`、`dense1k/`、`selected2k/` 保存第一阶段逐 token 秒数、顺序、完整输出、
接受统计、offset、内存和源码 hash；`replay_final/` 对应最终状态机制源码。
提交的 `evidence.json` 汇集上述原始逐 token 数据，去除重复 cache-layout 描述；
完整逐层路由数组和大 trace 留在本机，提交中保留其 digest 与调用数。
`final_sources.json` 是投影阶段的历史标识，`final_combined_sources.json` 是最终
checkout 的文件 hash。未提供混入过期源码的 patch 包。

服务已使用原来的 LaunchAgent 配置恢复，端口 39007；没有推送远端。
以下命令需要先停服务，脚本拒绝同时加载第二份完整模型：

```sh
PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 \
 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200 \
 /Users/jay/workspace/prod/common/sglang-metal/bin/python \
 benchmark/mlx/kda_gates_20261107/replay_abba.py \
 --output benchmark/mlx/kda_gates_20261107/replay_new --tokens 128
```

`fullmodel_abba.py` 现在将当前 checkout 的整体实现与原始 commit 对照；
不能把未来运行的组合候选当作历史第一阶段源码。`--history dense1k` 或
`selected2k` 选择固定 history。`summarize.py` 核对各阶段记录及最终源码并汇总，
不混合不同阶段和不同口径的数据。
