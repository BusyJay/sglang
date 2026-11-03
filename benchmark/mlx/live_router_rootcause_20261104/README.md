# M5 live-router、权重 locality 与完整模型回退调查

2026-11-04。只在目标 Apple M5 Ultra 256 GiB（applegpu_g17d）执行。
macOS 27.2、Xcode 27.2、MLX 0.32.4。远端基准 HEAD
`7b7e28d950097c080d0e1f7fd68db9f31f3315fb`；原有 tracked 改动保持原样。
目标 revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`，草稿 revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a`。使用已组织的本地 checkpoint cache，
没有重新量化、修改数值、安装依赖、修改系统 wired-memory 或下载权重。

## 采用决定

这轮没有采用新的生产优化。`mx.depends(x, weights)` 确实恢复了单层 NVFP4
复用候选的收益，但完整模型 DFlash 仍比当前原生 NVFP4 路径慢。保留实验代码
作因果诊断，生产使用既有 shared FP8 优化和原生 NVFP4。不是接受率下降：
36 次 300-token 生成的 token 序列和接受计数在所有 arm 间一致。

| 同驻留权重、交替顺序、3 prompts × 2 rounds | AR tok/s 均值 | DFlash3 tok/s 均值 |
|---|---:|---:|
| fp8：当前 shared FP8 + 原生 NVFP4 | 30.24278 | 51.98693 |
| reuse1：NVFP4 专家工作表复用 | 30.24045 | 50.42361 |
| dependency：reuse1 + 输入等待归一化路由权重 | 30.24404 | 50.22708 |

这里 DFlash3 表示三个 proposals、四个 target 验证行。运行
`fullmodel_dependencies.py --tokens 300 --modes fp8,reuse1,dependency --rounds 2 --prompts 3`
（仓库 PYTHONPATH=python，MLX_ENABLE_TF32=0，buffer 1024MiB/200ops，decode/verify group2，prefill4）。
原始逐步耗时、token、接受深度和 host 建图计数在 `fullmodel_dependencies.json`。
host verify build 包含提交阻塞，不是纯 CPU 时间；同步诊断不能与异步吞吐相加。
峰值约172.54GiB。系统 swap 在实验前已有1521.88MiB，结束1595MiB；没有进程级
swap 归属证明，因此这一点是环境限制。只加载一个完整模型，服务在测量时停止。

## live-router 回退的因果证据

300 次交替调用，真实四层生产 FP32 padded8 router，独占设备，各 arm 逐位相同。

| layer | 原生 live ms | reuse1 live ms | reuse1 等待 weights ms |
|---|---:|---:|---:|
|26|0.653887|0.713672|0.647012|
|3|0.733348|0.797525|0.725579|
|35|0.663540|0.717546|0.650521|
|41|0.662237|0.715773|0.648494|

只等待 IDs 无效，等待 weights 或整个 router 有效。gate/up 顺序、提前构造工作表、
显式先执行 shared/experts 没有解决问题。BF16、生产 shared flags、GPU 新输入
producer 都保留相同现象。`mx.depends` 共享原有存储，没有额外计算或权重拷贝。
因此该现象涉及 lazy graph 调度和生产者依赖，不是固定路由热缓存的虚假收益。
MLX v0.32.4 源码还确认 BFS 调度和基于生产者标识的 buffer barrier；buffer commit
阈值用整个 buffer 大小。阈值调参此前已排除为单独修复方法。

42 层 bank macro 的126项数值检查精确；reuse1/dependency 均值20.864/19.102ms。
原生首个10-step block 约485ms，包含未热的表达式形状；后续20.34ms。
原生总均值35.837ms不能用于稳态收益结论。保留原始数据，正式比较需先热同一表达式。
即使稳态单层/bank 有收益，也不能据此推算完整模型吞吐。

## M5 的性能计数器可用

公开 MTLDevice.counterSets 只有 timestamp，并不代表 Xcode gpudebug 没有计数器。
实际能读取33组 timeline counters。第一次 profile run + load 在同一进程中得到
空统计；第二个 gpudebug 进程 profile load 能加载 embedded session。
v2 high/overlapping 的 Consistent state=no，仅保留为线索。v3 medium/overlapping
两次均 Consistent state=yes、80/80 cores，结果如下。

| 真实 layer26 reuse1 | 原始 live | weights dependency |
|---|---:|---:|
| GPU ms |0.797166|0.695375|
| occupancy % |32.16|38.15|
| occupancy target % |57.95|48.58|
| GPR pressure influence |0|0|
| memory stalls influence |2.44|1.97|
| GPU read GiB/s |409.90|474.00|
| ALU utilization % |38.52|44.05|

两者均30 commands/4 encoders。dependency 将路由 gather/reduce/divide 移至
experts 前，没有减少 dispatch。共享 FP8 的 cost fraction17.308→8.055；cost fraction
不是时间百分比。shader timeline 混合不同 replay 时钟，出现不合理约197ms frame，
不能把 shader-list 时间当单次耗时；encoder intervals 一致。每 command instruction
mix 位于 `/performance/commands/disp0/counters/compute_kernel`。
保存 v3 loaded/collect logs 和 summary JSON；庞大 gputrace 不提交、不分发。

## 用户提出的 LFU/locality 方案

`locality.py` 对完整专家 weight/scale/global-scale bank 作无损排列，GPU 逆映射
原 router ID；原始/重排 arm 使用相同 remap。路由计算与求和顺序不变，全部24项
检查逐位一致，所有存储字节可逆恢复。每 arm 300 次交替调用。
LFU 训练只用了一个归档 c4+c1 路由，不能当真实请求分布的 LFU。

| live 重排 | 原生收益 | reuse1 收益 | dependency 收益 |
|---|---:|---:|---:|
|小样本 LFU|-0.52%|+0.39%|+0.72%|
|当前共访问专家最紧密 oracle|+0.087%|-0.066%|-0.57%|

随机排列的 dependency 也有约+1.23%，没有 LFU 特异收益。首次重排0.1815s，后续
0.085–0.089s；两份单层 bank 活跃约7.62GiB、峰值8.75GiB（12GiB限额）。
该层的专家 block 距离不是主要瓶颈；未排除其他层、真实分布或 kernel 内访问组织。

`locality_sorted.py` 按物理 expert ID 执行，先逆恢复输出顺序再 weighted reduction，
保持精确。原始 reuse1 .7130→.6721ms，dependency .6437→.6522，原生 .657→.759；
LFU/oracle 重排仍无可靠帮助。排序改善部分调度回退，但慢于已有依赖修复，且伤害
原生与 dependency。没有采用，也没有把单层结果包装为全模型收益。

`row_locality.py` 用同一 custom kernel，仅改变专家 interleaved8/32/128 output-row
tile 的精确地址。6项输出和字节恢复检查精确；live/dependency 差异约1%内、无稳定收益。
原始→重排 ms：tile8 live .71715→.72299、dep .65250→.65587；
tile32 .71862→.71938/.64731→.65476；tile128 .71812→.71662/.65572→.65052。

## 为什么局部收益未进入真实模型

沿真实 layer26 的 mHC + FFN 残差图继续消融（控制输入由归档 x 构造，
不是完整 serving 的真实残差）。包含 mHC 多输出 producer 和残差 consumer 时，
原生/reuse1/dependency 1.1334/1.1951/1.1982ms，依赖修复失效。
仅保留 mHC producer、去掉残差 consumer：.7491/.7911/.7297，修复有效。
把 mHC 输出提前求值、保留残差 consumer：.6565/.7092/.6428，修复有效。
说明组合图改变了调度；不能单独归咎 mHC 算术、router、接受率或静态寄存器数。
额外约束 post/residual 等待 router/branch 没有修复。

降低 MLX BFS width 到1/4/8令候选相对原生改善，但绝对耗时约1.95/1.51/1.25ms，
仍慢于原始原生1.13ms。width64与原值相近。没有采用调度参数。
这是剩余机制的定位证据，尚未获得消除完整图回退的实现。后续转向真实 DFlash
长候选 verify 的投影、attention、快照和提交成本；不能宣布没有优化空间。

## 证据和复现边界

manifest 列出本次选择归档的脚本、raw JSON/log 及 SHA256。只 stage 这些路径，
不混入继承的生产代码、其他轮实验、权重/虚拟环境/庞大 captures/transport payload。
部分脚本通过 source replacement 派生，需要同时保留 production_route.py 和 wrapper。
测试为上述真实 M5 单层、原始 bank 和同驻留完整模型生成；本轮没有重新做 HTTP
服务准确率矩阵。失败的临时 producer kernel 使用 x.size()、HC SimpleNamespace
遮蔽问题已修复；失败运行不作为正确性或性能证据。
