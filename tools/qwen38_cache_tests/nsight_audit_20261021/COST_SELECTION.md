# 共算改为可选扩展，先准入再选择 CPU 子集

2026-10-21。本文件描述审计后的源码修改；README、原 trace 和审计统计
仍对应工作区起始版本 `76406be864`，不能作为修改后的性能结果。

## 为什么 GPU 会等 CPU

历史策略没有持续执行“只有完整 CPU 路径更快才参与”的判断。
`9e4c66a657` 引入 hybrid 时 distinct miss 门限为 33，而 M≤3、top-k=10
最多只有 30 条路由；`1c44f94100` 改为 3，仍有 ownership 条件。
`f65e26e5ab` 为异步 admission 删除 GPU crossover 分支，decode/verify
所有不可读专家改由 CPU 计算，包括新填充尚未发布的专家。

LFU 的 admission=1.9 是频率准入条件，不是时间成本。GPU resident
计算结束后，下一层仍依赖 CPU 输出。原审计中 3022/4848 次 CPU 尾部、
0.852703 秒 join/native/GPU idle 交集与这条依赖一致，但不能据此直接
推算可以回收多少吞吐。

## 默认行为与可选共算

`SGLANG_ENABLE_QWEN38_CPU_HYBRID` 默认 False。默认使用已有 GPU cache
实现，不启动 hybrid 管理器、CPU worker 或 CPU kernel 构建。
显式开启但未提供成本标定时，也使用已有 GPU 实现。

开启扩展需要同时设置：

```sh
export SGLANG_ENABLE_QWEN38_CPU_HYBRID=1
export SGLANG_QWEN38_CPU_COST_CALIBRATION=/absolute/calibration.json
```

标定可传 JSON 文件路径或 JSON 字符串。开启后，decode/verify 的步骤是：

1. GPU planner 更新 LFU 频率、预判准入。免费 slot 与全局较冷 victim
   都可用于准入；只有 rejected 专家可以成为 CPU 候选。
2. 启动时将完整配对标定中满足 margin 的样本上传 GPU，设备选择精确
   匹配且耗时最低的 CPU 候选前缀。未知样本或无收益时全部留 GPU。
3. rejected 且仍交给 GPU 的专家，先用空闲 slot，否则强制替换全局
   最冷且本次未使用的 resident。保护本次 active 专家，清除 victim 原层
   lookup 并迁移 ownership；接受新缓存状态，不恢复旧专家。
4. planner 产生 GPU pool、CPU 两组互斥路由和一张注册内存填充计划。
   所有 GPU miss 都直接填入共享 pool，仅一次搬运、一次 GPU 专家计算。
   planner、fill、GPU reader 同一有序 stream，不需要搬运完成 flag。
5. 仅 CPU 路由异步传到 worker。worker 选中 CPU 时搬运激活/路由权重、
   计算并上传 FP32 输出；主线程不读回原始路由做决策。选中 CPU 才等待
   完成信号，GPU/CPU 以 FP32 合并后转换为 BF16。

| 示例 | CPU | LFU 准入 | GPU 强制放入 pool | GPU 合计 |
| --- | --- | --- | --- | --- |
| 10 miss 全准入 | 0 | 10 | 0 | 10 |
| 10 miss，5 准入，标定选 1 CPU | 1 | 5 | 4 | 9 |
| 10 miss，5 准入，未知/无收益 | 0 | 5 | 5 | 10 |

这里的 LFU 拒绝指初始保留意愿，不代表禁止 GPU 放入。强制替换可能
淘汰原本更热的专家，影响后续命中率；这是取消临时空间并坚持一次
GPU 计算的缓存代价，需要用真实请求负载测量。

标定表没有 CPU 优胜样本时，不发起路由 D2H、CPU 任务或完成等待。
有 CPU 优胜样本时，全 GPU 调用仍会发起小路由 mailbox，但不会等待
worker。GPU 路径只有一张计划、一次填充和一次专家计算提交。

## pool 容量与 prefill 借用

decode 没有独立专家 arena，也不永久预留 pool 尾部专家槽。B12X
已有 zero padding slot 0 保留。启动交接和 resize 验证可用 slot 数至少
为 `min(num_experts, 3 * top_k)`，保证未知标定也能全部在 GPU 计算。
预算不足明确报错，不强迫未经标定的 CPU 子集参与；直接 kernel 调用
违反容量前提时也会 fail closed，不输出缺失专家的计算结果。

prefill 仍使用旧的整层临时 view。若它借用 pool 尾部，只在 prefill 期间
退休旧读者、失效对应 resident、暂时禁止准入；完成后归还全部借用槽，
decode 恢复使用整个 cache（B12X padding slot 0 除外）。反复 prefill 可能
导致这些 resident 被淘汰，需要实际负载验证，不能宣称无缓存代价。

设备状态在首次交接、prefill/resize/close 边界与主机 GlobalLfu 同步。
频率采用 FP64，保留原主机算术与判定；设备已写入的 mapping 只由同一
stream 上、填充之后的计算消费，主机边界等待读者后才读取 metadata。
当前仅支持 M=1..3、top_k<=10、expert<=512、有注册 pack 的 mailbox。
开启共算的 decode 必须在 graph capture 外、同一 compute stream 执行。

## 全局 victim 选择

设备 decode planner 扫描整个共享 pool 的 resident，以 `(frequency, layer,
expert)` 顺序选 victim；slot ownership 只标识当前 payload，不限制淘汰
范围。当前调用的 active experts、padding/借用保留槽不作为 victim。
跨层替换同时清除原层 lookup、迁移 ownership，再将 candidate 填入 slot。

所有 decode planner、填充和读者在同一个 compute stream 上按序执行，
后续覆盖自然发生在先前读者之后，无需额外 host 等待或逐层 reader gate。
这不表示 host 返回时 GPU 已执行完，而是表示同一 stream 已保证正确顺序。
prefill 的旧后台填充则只跨层使用已完成 reader 且无维护任务的层；主机
参考策略通过 safe_layers 接收这个边界，不能把它的旧默认范围当成设备
全局淘汰范围。

## 为什么 CPU 完成信号仍存在

GPU planner、填充和 GPU 计算的顺序依靠 stream，不使用搬运 flag。
CPU 完成信号负责另一条依赖：worker 线程计算并上传结果后发布 sequence，
GPU 合并 CPU 输出前等待。worker 内部同步完成 H2D 不能替代 compute
stream 的结果依赖。设备等待 kernel 对没有 CPU 路由的调用立即返回。
失败/超时沿用 mailbox watchdog 的 poisoned 状态和 mapped failure flag，
防止选中 CPU 的等待无限挂住。该等待 kernel 会占用一个 CTA；必须在
GPU 验收时检查实际影响，不能把移除主机 Future 等同于消除 CPU 长尾。

## 标定口径

启动日志输出 `Qwen38 hybrid cost calibration metadata`。将这些字段
复制进标定表，再加 `schema_version: 1`、`margin`、`samples`。
metadata 包含 backend、hidden、width、top_k、物理 cores、gpu、cpu、
transport、runtime_version，必须精确匹配。版本指纹包含当前 hybrid、
策略、GPU adapter 源码、CPU 动态库、Torch 和 FlashInfer 版本。

每个 sample 需要：

| 字段 | 含义 |
| --- | --- |
| tokens | M=1..3 |
| hit_routes | 留在 GPU 的全部路由数，包括本次需要搬运的专家 |
| miss_routes | 候选 CPU 子集的路由数 |
| miss_experts | 候选 CPU 子集不同专家数 |
| miss_tokens | 升序排列的候选 CPU 专家路由次数 |
| rejected_tokens | 升序排列的全部 LFU rejected 专家路由次数，包含 CPU 候选 |
| cpu_us | 该子集在 CPU、其余在 GPU 的完整配对耗时 |
| gpu_us | 同一工作量全部在 GPU 的完整配对耗时 |

`rejected_tokens` 区分“其余 GPU 路由全部驻留”与“其余仍需搬运 GPU miss”，
避免复用搬运成本不同的样本。CPU 子集必须是 rejected 的子集。
计时包含主机派工、激活/权重 D2H、GPU 工作、CPU worker、join、结果
H2D、专家向共享 pool 的搬运及 FP32 合并，用 elapsed time，不累加重叠时间。
配对实验需相同实际路由、缓存状态、预算、传输方式，并覆盖后台竞争。
仅 CPU/GPU kernel 微测不能直接填写这些时间。

目前没有生产硬件配对样本；不提供虚构 CPU 优胜标定。空 samples 表可
加载 metadata 并保持扩展内 GPU 选择。实现改变后应重新标定；固定表
不能保证负载变化后仍准确，需多轮无 profiler 的服务 A/B 验收。

## 验证与边界

benchmark_performance.py 和 profile_nsight.py 的 default 保留调用方显式
共算配置，legacy_gpu 强制关闭扩展；未设置配置时，两者都走 GPU 默认
实现。启用扩展也会切换到已有 GlobalLfu 缓存组织，因此整体 A/B 同时
包含缓存组织差异，不能把所有差异归因于 CPU 计算。

Nsight scopes 已补设备 planner、worker 与 device wait 的 Python 入口。
原 audit.py 用于复核原 trace；新增 planner、fill 和 CPU merge 的时间
需要按新路径解释，不能把原统计当作优化后的性能结果。

本机本地 Docker 有 CUDA 版 Torch，但 CUDA 不可用。策略、主机协议、
CPU Torch 数值参考、loader 配置可验证；实际 CUDA 并发、B12X 精度、
graph replay、吞吐与显存预算需要 GPU 验证。本次策略 44 项、dispatch 20 项、resize 协议 15 项、loader 11 项、
environ 11 项及原工具 CPU 测试 351 项全部通过。相关 Python 文件
的 Ruff format、E9/F/I 检查、测试注册检查及 diff 空白检查通过。
未生成优化后的 Nsight 或性能报告。

CPU 多专家 batch 候选仍保存在原 candidate.patch，未套用生产 kernel。

本次将实际 planner kernel 主体以单线程主机方式编译执行，1000 个随机
窗口的 mapping、ownership、热度、填充/淘汰数量与独立 GlobalLfu 加
强制全局放入参考实现完全一致，LFU 每个窗口只观察一次。另验证
5 admitted、1 CPU、4 强制跨层替换，共 9 个专家用同一 GPU pool。这是算法验证，
不验证 CUDA 线程、内存可见性或并发。新 planner/wait JIT 可编译为 SM120f；
GPU 对照测试已注册，但本机没有可用 GPU，未运行。

真实 route trace 的非共算逐层 LFU 对照见 [TRANSFER_REPLAY.md](TRANSFER_REPLAY.md)。
固定相同 slot 预算，预热后整体专家 H2D 减少 11.01%–12.77%；
此结果不包含实际 CUDA planner 时间与 CPU 参与，不代表已通过吞吐验收。
