# 机制追审修订与当前吞吐

2026-11-01。当前生产更改已同步到 M5 Ultra 远端；没有 git commit/push。本报告取代父目录文档作为当前源码说明。旧 HTTP18场景和早期五组ABBA仍是历史版本证据，不能移作这版收益。

M5 Ultra256GiB / MLX0.32.4 / TF32=0，GLM-5.3-Flash checkpoint `fa04cbc3622b16aa7475404060b05e506d81bd4e`，保留 checkpoint 原混合4/5/6bit及BF16权重。持久环境 `/Users/jay/workspace/prod/common/sglang-metal/bin/python`，远端 `/Users/jay/workspace/sglang`，HEAD `fb5d8fe5bad8c9086ea94bfd6341097fe1cd81a0`。按正常 runner 的进程资源默认值设置 wired limit，没有系统内存参数变更、安装或下载。

## 当前源码全模型 tok/s

基准为追审前已经实现图观察/5、6bit加载/KDA几何/prefill/服务契约的版本；本轮增量为4bit mask加载、head1/3行几何和mHC两套并行投影。2050个确定输入token，greedy、单请求，prefill chunk256。每arm输出48token，前5次调用单列，稳态吞吐=43/后43次耗时。两组ABBA，每模式每arm4样本，未进行大矩阵测试。

| 模式 | 追审前 tok/s | 当前 tok/s | 中位数变化 |
| --- | ---: | ---: | ---: |
| AR | 34.098 | 34.266 | +0.49% |
| MTP2 | 55.471 | 55.367 | -0.19% |

**这轮没有证明 MTP2 稳定吞吐提升。** AR小幅正向，MTP2结果跨组变化、总体中位数略低；不能把4bit/head算子收益相加换算全模型收益。保留全部原始顺序及离群值，见 `remote_evidence/final_compare.json`。首5次调用与全程吞吐均保留，不把热路径结果当首次成本。

同模式全部生成token哈希一致；MTP2每样本draft proposed40、reuse28，depth1/2 accepted19/9，与基准相同。记录每budget/depth原始统计。这个输出序列不是通用质量评估。峰值为进程累计峰值：AR约190.252GiB、MTP2约190.942GiB；活跃内存约190.19/190.47GiB。未发现该短程对照有大型权重副本，不能代替长context/请求释放资源验收。

分项全模型对照（此前每arm64token、排前5次、两组交替，每arm4样本；没有包含新mHC，亦不替代最终合并对照）：

| 模式 | 基准 | 仅4bit gate | 仅head1/3 | 两者 |
| --- | ---: | ---: | ---: | ---: |
| AR | 34.423 | 34.452 | 34.451 | 34.532 |
| MTP2 | 54.727 | 54.967 | 54.760 | 54.825 |

独立mHC全模型MTP2（其他本轮更改已开启）：54.852→54.870 tok/s。幅度极小，不能宣称稳定收益。

## 采用的生产机制及边界

- **4bit gate/up**：M5上多行grouped路径，将原逐字节thread数组替换为两个对齐uint32 word生成uint4 native ushort masks；保留原四项求和、affine rounding、clamp/SwiGLU边界、scatter pick顺序和任意packed stride。实际checkpoint三行算子约0.3939→0.3861ms，逐位一致；单行仍原native路径。5/6bit采用的原宽加载继续保留，不能再冒充4bit已优化。
- **head1/3行**：K4096/N154880采用BM4/RPS4，覆盖普通MTP2 target3与draft1，包含singleton greedy函数和rows函数。各lane K切片及shuffle次序不变。真实checkpoint head rows1/3约1.3508→1.3143 /1.3569→1.3169ms；未改变checkpoint/量化。9行以上原BM8/RPS8保留。早期actual_probe标注vector的实现未实际替换source，该标签不作vector收益证据；refine中修正并检查替换，但vector实验未合入。
- **mHC**：M5/count≥3/K16384使用512线程组，两套原256投影并行、每套保持4个accumulator，仅第一套生成RMS inverse，组内共享。没有device inverse中间数组或额外dispatch，原reduction次序不变。单行、其他K、其他设备仍走原融合机制。通过6项现有设备回归和全模型生成一致性；operator独立审核小收益保留，最终全模型没有签稳定收益。

## Down加载、融合及metadata复用：已推进实验，生产热点仍未改善

新增完整融合device qdot/byte/word/threadgroup packed tile、保留per-pick并行的threadgroup版本、compact gate任务表，以及gate/up→down同一事务复用 `_group_picks`。这些是workspace内独立候选；没有临时改变生产共享模块状态。synthetic 4/5/6bit×spread/mixed/repeat exact，实际checkpoint5bit operand和4层完整gate+down事务exact。

真实5bit down baseline0.2714ms，device0.2951、byte0.3249、threadgroup0.4172。完整事务包含metadata：四层baseline0.5142/0.5183/0.5190/0.5178ms；compact0.5240/0.5198/0.5194/0.5215；共享metadata+device down0.5202/0.5241/0.5190/0.5227。原始顺序见 `actual_probe.json`/`transaction_probe.json`；事务baseline已经包含本轮4bit mask加载。

去掉partial、额外combine，并复用真实metadata后仍未胜出。保留原生产down/combine，排除本轮已验证回退候选；**不能把原因仅归于额外dispatch，不能称down方向已优化完成，也不能称Metal限制。** synthetic小专家工作集与真实288专家工作集区别明确。该方向缺最终shader加载/寄存器计数和能在实际路由下胜出的跨行复用机制，仍是最大未改善热点之一。

## Stream归因及保留group的原型

A=原固定group；B=相同拆分图/观察/边界但shared也在默认stream；C=同B仅shared侧stream。两轮ABC/CBA、32decode/arm，前期cold处理仍可能有布局扰动，数据仅代表该短程工作负载。

| 模式 | A tok/s | B tok/s | C tok/s |
| --- | ---: | ---: | ---: |
| AR | 34.101 | 33.863 | 27.680 |
| MTP2 | 61.608 | 59.900 | 52.587 |

随后实现显式stream传入固定group/MLA finish编译图；stream身份入图缓存键、调用闭包持有stream，不修改MoE共享状态，去掉逐层 `async_eval(shared)`，保留原group提交边界。MLX Stream不能直接与None做缓存键比较的接口错误已修复，不将其当设备限制。

| 保留group模式 | 单stream tok/s | side stream tok/s |
| --- | ---: | ---: |
| AR | 34.288 | 25.850 |
| MTP2 | 55.265 | 44.404 |

每模式全部token和最终cache数组字节哈希一致。保留group仍回退约20–25%；该候选不合入生产。两类原型和host pstats文本均保存，不因拆图候选回退直接结束机制研究。

设备短capture成功（PID45529，3142条encoder/device interval，总union522.423ms）。原始trace留远端 `group_stream.trace`，导出为 `group_device/device.trace.json`；它是encoder/device区间，**不是shader持续时间、DRAM流量或occupancy证据**。arm窗口包含首次图重建及末尾cache hash，第一arm开始早于trace start约152.5ms；因此不拿窗口绝对耗时比较净GPU overlap收益。host profile亦含配置失效与cache哈希，不能把其总时间当稳态submit成本。这次证据只支持“保留group的该实现仍回退”，最终带宽/寄存器/依赖归因未完成。

## 最少量最终检查

本机没有可调用的ruff/black，未安装；静态diff检查通过。最终4bitmask1项、head6项、缓存契约3项通过；mHC6项在最终同一kernel source上通过。含token-only在消费边界切换、保留logits/cache契约、替换layer container刷新图。既有相关kernel/服务测试未大规模重复。`git diff --check`和AST检查仅为静态检查，不冒充设备验证。

HTTP最终源码单轮：2050prompt/64输出，AR和MTP2、并发1/2；chunk256/context32768/pool65536，端口30022。没有本轮HTTP baseline重跑，以下为绝对吞吐，不能称HTTP增益：

| 模式 | 并发 | Aggregate tok/s（含prefill） | 平均request decode tok/s |
| --- | ---: | ---: | ---: |
| AR | 1 | 14.833 | 32.884 |
| AR | 2 | 18.923 | 27.946 |
| MTP2 | 1 | 17.217 | 48.231 |
| MTP2 | 2 | 18.575 | 28.483 |

服务cold/prefix logprob路径检查：AR与MTP2的cold/prefix token及logprob panel哈希一致，warm cached_tokens均为256。原始请求结果保存在JSON，没有长32K/MTP4/rejection/cancel的本轮完整覆盖。

## 源码身份、复现与遗留

`final_compare_source.json`和`final_remote_source.json`记录最终测量源码SHA256；后者包括远端继承pyproject改动，不应纳入本轮patch。`baseline/`冻结追审前实现；局部实验source/修订脚本/原始JSON均保存。生产与registered diff保持未提交，没有自动推送。

复现最终checks：远端repo中按 `final_checks.sh` 使用持久环境执行；本地通过tmux helper的command-file传输。不得并行启动两个完整模型以免内存压力。短程capture与吞吐独立，raw全部保留。

**未完成的性能机制**：down真实路由跨行权重复用；shared/routed净并发收益；稳定MTP2提升；长context/首次布局与完整HTTP资源矩阵。已保留并默认采用通过数值且没有确认实际回退的4bit/head/mHC候选，未用诊断关闭开关包装完成。当前不能验收为“所有热点已经优化完毕”。
