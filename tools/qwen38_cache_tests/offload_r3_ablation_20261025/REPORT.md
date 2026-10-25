# Qwen3.8 r3 逐项优化验收与最终筛选

采用用户确认的标准：**只关闭“实现正确且确认性能回退”的项；实现有缺陷先修复；提升或持平的项保留。** 本轮没有确认应关闭的线上回退项。发现并修复了 U2、U3 两个实现缺陷；其余路径保留。表中的“持平”表示在本请求、重复基线波动内未确认回退，不表示严格零差异或所有硬件均无回退。

## 逐项结论

CUDA 算子时间为固定输入 CUDA Graph + GPU event 的 ABBA 中位数，单位 µs；U2/U4 单列主机时间。服务效果为启用相对关闭的估计；**正数表示更快**，斜线分隔正序、反序两轮。这里的算子改善不换算成整个服务同等百分比收益。

| 优化 | 独立证据 | 判断 | 最终处理 |
| --- | --- | --- | --- |
| U1 QSA gather | BF16 gather 8.414→1.492 µs；TTFT +0.12% / -0.11% | 算子提升；整请求 TTFT 持平 | 保留 |
| U2 eager direct | 普通参数 3.514→3.114 µs；VERIFY +2.84% / -1.10% | 主机开销提升；服务持平 | 修复容器 dispatch 后保留 |
| U3 logits cast | M1 1.655→0.959 µs；M3 2.167→1.553 µs；VERIFY +1.50% / -0.26% | cast 提升；服务持平 | 修复列 stride 回退后保留 |
| U4 Mamba planning | 真实规划函数 CPU 9.605→6.436 µs；VERIFY +4.58% / +1.81% | CPU 路径提升；服务正向趋势 | 保留 |
| A startup precompile | 独立冷缓存：停顿 1.789→0.247 s；启动 50.189→64.203 s；VERIFY +0.77% / +0.42% | 冷路径改善；稳态持平 | 保留（冷启动有编译成本） |
| B split fill | VERIFY -0.82% / -0.07%；吞吐 +4.52% / +7.98% | 服务持平，未确认回退 | 保留 |
| C1 HC gate overlap | 串行 M3 1.307→1.307 µs；VERIFY +1.93% / -2.37% | 服务持平，未确认回退 | 保留 |
| C2 shared gate overlap | 串行 M3 2.044→2.659 µs；fill 后 1.717 µs；VERIFY +1.21% / -1.40% | 串行工作增加；重叠服务持平 | 保留；不宣称减少总 GPU 工作 |
| U5 CUTLASS tune ceiling | TTFT +0.12% / +0.11% | 服务持平，未确认回退 | 保留 |
| F draft head formats | 未接入 serving，未采集真实 head 激活，未运行 GPU 格式收益验收 | 不属于本次线上性能集合 | 仅离线工具；不启用、不计作持平或收益 |

B 的每 VERIFY 差异约 -0.8% / -0.1%，远小于基线迭代波动，而且实际 token/s 两轮方向相反于该小差异，不能认定它退化。C1/C2 两轮每 VERIFY 差异方向相反，也不满足“确认回退”。因此按用户标准保留，不用单个慢轮次关掉正确实现。

### C2 为什么串行变慢仍判服务持平

C2 把一次融合算子拆成可提前执行的 FP32 gate 与 fill 后的消费者。M3 串行总 GPU 时间从 2.044 增到 2.659 µs；消费者含与基线相同的 routed reset copy，只有 1.717 µs。提前部分在独立 fill 窗口执行。总 GPU 工作增加是真实代价，不能宣称是融合减工作；两轮完整服务没有确认性能退化，所以本轮保留。C1 串行总量基本相等，gate 仅改变执行位置。

### A 的启动成本

A 把可达 bucket 的首次编译移到启动。独立冷缓存对照中，仅切换 A，其余优化全开：关闭时启动 50.189 s、首次生成最大流间隔 1.789 s；开启时启动 64.203 s、最大流间隔 0.247 s。冷启动增加 14.015 s，首次生成停顿减少 1.542 s。两臂分别用只增加注释、AST 完全相同的 phase 源生成独立 compiler cache key，避免另一臂预热；测后恢复原始源与部署。结果见 `cold-a-results.json`，控制脚本为 `bench_cold_start.py`。这是一对冷路径观察，不把不同输出的吞吐变化宣称为精确稳态收益。

修复还共享了 phase namespace 和已预编译键，48 个 adapter 共用编译结果，入口从 1440 降为 30，权重绑定、scratch、graph、reader event 保持各 adapter 独立。编译入口次数不等于 GPU 实际编译次数；原有 compiler cache 已能复用实际编译，没有把先前观察到的约 18.5 s 编译时间虚报为 namespace 共享节省。

## 两个实现缺陷与修复

- **U2**：旧快速路径只检查直接 Tensor 参数，Tensor 列表中的 CPU 数据会绕过 CUDA-only dispatcher，错误返回 CPU 结果。已真实复现（`u2-container-before.json`），列表/元组/字典参数现保留 dispatcher 的设备与 subclass 行为；增加 CPU 列表拒绝和 CUDA 列表结果测试。修复后普通参数和混合 scalar 参数的 host enqueue 都比注册 op 更短，见 `u2-fixed-timing.json`。
- **U3**：新 Triton cast 只支持列连续数据，集成处却仅检查 device/dtype；列 stride=2 的原合法 `copy_` 路径触发断言。`operators-before-layout-fix.json` 记录失败，修复后 `operators.json` 为 passed。集成处现在检查二维布局、非空、输入/输出列 stride 与输出行重叠；不符合的继续 `copy_`。新增覆盖 FP16/BF16、M1/M3、输入/输出列 stride 的回归，验证输出 alias 和 padding 不被覆盖。列连续的常用路径继续使用更快的 cast。

U1 的 CPU 序列长度遗漏、FC1 验收 fixture 的 CPU swizzle / 完整 mailbox plan、Mamba fixture allocator、eager 测试脚本重复注册问题沿用前两轮修复。本次 source diff 和最终散列均包含这些修复。

## 服务对照方法与限制

完整正反顺序矩阵 **25 轮**：7 个全开重复基线；U1、U2、U3、U4、A、B、C1、C2、U5 各关闭两次。每轮重新启动服务，仅变更该项；U1–U4 使用固定基线 `cd748677bb6bd87c3162b24c8a4d3591dce2c8ea` 对应生产文件，其余用独立环境变量。参考为全部 r3 路径及共享编译缓存修复开启。

输入为冻结 Hermes 的 25,560 input_ids。矩阵固定 greedy（temperature=0、top_k=1、top_p=1）、输出上限 4096、保留自然 EOS、服务 seed=42。25/25 均完成 4096，cached_tokens=0、retractions=0，专家总容量均为 **7161 slots**，无推理错误。容量和 chunk/MTP/KV 精度一致。最终另用原请求采样参数、5895 预算校验，不能把 greedy 的 token/s 与原采样直接比较。

即便 greedy，原有浮点累加与异步 cache / 路由会使输出不逐字一致：本轮 hash 均保留，接受长度也随内容变化。7 个正式参考吞吐范围 108.17–115.31 token/s，每 VERIFY 17.654–18.663 ms，TTFT 6.768–6.822 s。因此使用前后相邻参考插值作为描述性估计，同时报告原始吞吐/接受率、每 VERIFY 与算子固定输入测量；这些估计不是严格相同输出的因果加速比。U1/U5 按 prefill/TTFT 判断，不把随生成路径波动的 decode 变化记作其收益。

`pilot-results.json` 保留试跑，确认输出能达到4096。它不属于正反配对矩阵，未用来挑选最快基线；其同样显示 greedy 接受率会波动。

## 原始矩阵

| 轮次 | 单项状态 | token/s（128 后） | ms/VERIFY | TTFT s | accept length |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | ref | 112.59 | 18.118 | 6.815 | 2.031 |
| 01 | u1-off | 125.86 | 18.849 | 6.817 | 2.355 |
| 02 | u2-off | 113.54 | 18.393 | 6.777 | 2.082 |
| 03 | u3-off | 112.60 | 18.036 | 6.795 | 2.022 |
| 04 | ref | 115.31 | 17.654 | 6.789 | 2.033 |
| 05 | u4-off | 109.56 | 18.668 | 6.790 | 2.048 |
| 06 | a-off | 110.50 | 18.185 | 6.786 | 2.001 |
| 07 | b-off | 110.32 | 18.093 | 6.811 | 1.991 |
| 08 | ref | 115.31 | 18.440 | 6.768 | 2.118 |
| 09 | c1-off | 104.23 | 18.640 | 6.815 | 1.940 |
| 10 | c2-off | 111.95 | 18.356 | 6.802 | 2.051 |
| 11 | u5-off | 111.65 | 18.706 | 6.793 | 2.078 |
| 12 | ref | 112.05 | 17.833 | 6.791 | 1.999 |
| 13 | u5-off | 107.66 | 18.541 | 6.803 | 1.993 |
| 14 | c2-off | 104.64 | 17.993 | 6.800 | 1.884 |
| 15 | c1-off | 109.09 | 18.018 | 6.824 | 1.964 |
| 16 | ref | 110.35 | 18.663 | 6.810 | 2.053 |
| 17 | b-off | 101.69 | 18.525 | 6.803 | 1.886 |
| 18 | a-off | 109.38 | 18.492 | 6.785 | 2.017 |
| 19 | u4-off | 110.42 | 18.621 | 6.803 | 2.055 |
| 20 | ref | 108.17 | 18.166 | 6.822 | 1.962 |
| 21 | u3-off | 110.51 | 18.085 | 6.805 | 1.989 |
| 22 | u2-off | 106.90 | 17.900 | 6.802 | 1.908 |
| 23 | u1-off | 106.86 | 18.405 | 6.804 | 1.965 |
| 24 | ref | 109.73 | 18.033 | 6.807 | 1.978 |

## 算子方法

- U1 使用真实几何：BS1、25560 tokens、ratio4、KV head1、dim128，分别测 BF16/FP8；基线是原 index_select + cat 公式。
- U3 使用实际词表宽248320、M1/M3，含 row padding；CPU/IO 断言在计时外。独立测试验证 column stride 的集成回退。
- warm-L2 每 graph 包含100次调用并用 events 计时；cold-L2 在 graph 内先触碰128 MiB flush，再在外部 CUDA event 节点之间测算子。ABBA 四组，GPU event 包括实际 capture/replay，分配和编译在计时外。
- C1/C2 串行总量与 post-fill consumer 分开，不能把 consumer 变短直接宣称为端到端节省。
- U2 对比实际 eager wrapper 与注册的同一 CUDA op，包括普通 Tensor 与混合 scalar 参数；U4 执行真实旧/新 prefill 规划方法，使用 GPU ping-pong buffer，验证指向同一槽和 checkpoint。U4 计时是在 GPU 空闲时测得的 host 函数耗时，不是整服务节省。

## 可复现材料

- `serving-results.json`：全部25轮原始统计、逐流事件之外的摘要和输出hash。
- `serving-effects.json`：相邻参考插值与每轮启用效果，正数更快，未混淆收益方向。
- `operators.json`、`u2-fixed-timing.json`：修复后固定输入的独立测量与正确性状态。
- `components.json`、`reference-source-hashes.json`：每项文件范围和测试时的完整参考源散列。最终源散列另列，U2容器和U3布局修复在矩阵后加入，并定向重测、再做完整请求校验。
- 私有 request/response、每轮完整日志/source hashes/server-info 留在 `/home/user/workspace/prod/homelab-llm/models/profiles/offload-r3-ablation-20261025`，未提交正文。

复跑时使用新的输出目录（脚本拒绝覆盖旧证据），并使用同一部署的私有冻结请求：

```bash
python3 tools/qwen38_cache_tests/offload_r3_ablation_20261025/run_ablation.py \
  --tag matrix --output-dir /path/to/new-private-results --request /path/to/frozen-request.json \
  ref u1-off u2-off u3-off ref u4-off a-off b-off ref c1-off c2-off u5-off ref \
  u5-off c2-off c1-off ref b-off a-off u4-off ref u3-off u2-off u1-off ref
```

GPU operator 脚本位于 `test/manual/kernels/qwen38_offload_operator_ablation.py`，需在服务停止、GPU空闲时运行，使用原 FlashInfer0.7.2 / torch2.14.1+cu130 / SM120 环境。CI registry、package 注册、AST/format/isort/diff whitespace 检查和 GPU/CPU 回归日志位于 `validation/`。MLX 硬件路径未在 CUDA 主机运行，未宣称已硬件验收。

最终开关、Hermes 原采样验证和提交信息见下方最终结果。

## 最终验证与部署

- 原 Hermes 采样参数（temperature=1、top_p=.95、top_k=20、repetition_penalty=1.1、sampling_seed=42）完整回放：25,560 输入、5,895 输出；TTFT 6.860 s，128 tokens 后 **97.67 token/s**，最大流间隔 **0.245 s**。完成 1 次真实 rebalance，无推理错误、无 retraction，/ready=200。原采样结果不与 greedy 矩阵直接作加速比。
- 最终 A/B/C1/C2/U5 均为1，U1–U4自动生效；MTP2、BF16 draft KV、FP8 target KV、220k context 和8192 chunk 保留。observe=0，profiler关闭。
- `validation/` 中全量341项通过、1项跳过；U2容器修复后定向11项重测通过，其中新增2项；去重为 **343 passed / 1 skipped**。两组 memcheck 均为0 memory errors（关闭已记录的CUDA Python API查询报错报告，显存检测仍启用）。
- 物理部署脚本在非Git的 prod 目录，提交记录其精确 `deployment.patch`；源仓库提交实际代码、测试、逐项报告与脱敏统计。没有推送远端。
