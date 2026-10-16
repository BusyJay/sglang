# Decode shared expert / fill 并发

用户要求将可独立的 GEMV 与 fill 并发。本次选择同层 shared expert 分支：
它只读取 MLP 输入，不依赖 miss 权重。不能让整段 5.32 ms 的所有 GEMV 都并发，
因为注意力投影等计算在当前 router 之前，或依赖当前 MoE 输出。
prefill、verify 及其 arena、dense/sparse、缓存复用和预热策略保持原调用路径。

## 执行与依赖

```text
compute: router/topk -> planner/alpha -> ready event
                                     -> shared expert -> wait filled -> routed MoE
                                                                        -> gate/add
fill:                                  wait ready -> fill -> filled event
```

运行条件为单 token CUDA decode、分离的 shared expert、非 inplace routed runner、
cached NVFP4 的 threaded direct-fill runtime、TP=1 且 A2A none。
shared/routed 使用同一输入但各自输出，保留原 dispatcher、router 权重、CUTLASS
runner、最终 fused gate/add 和 reduction。未拆分 routed GEMM，也没有额外 expert 复制。
冷的非 sampling decode 如仍需旧 transport，shared callback 仅串行执行一次。

独立 fill stream 等待 planner/alpha 的 ready event；compute 在 shared 分支后
等待 filled event，再执行 routed GEMM。现有 reader-completion event 因此覆盖
shared、fill 和 routed reader。异常路径同样 join 已提交的 fill，防止缓存提早复用。
两组同步 event 跟随原 completion event pool 复用，初始化时创建/record，
不会在每层服务调用中创建/销毁 event。

模型层给 cached MoE 增加专用 eager 边界，包住 router、shared 和 routed。
原 breakable graph 会在结束 segment 时自动 join side stream；只在旧调用处
加 stream 无法绕过这个汇合。新的 capture stub 仅创建输出 bridge；replay
恢复 decode phase 并计算新 route / 激活。shared callback 使用 ContextVar 作用域
传给原 quant 方法，正常返回或异常时均恢复，不把 Tensor 闭包保存进 captured graph。

## 开销与可观测性

没有新增权重/staging 缓冲，新增一条 stream 与两组 event，均在 KV 预算前准备。
它们仍有驱动元数据成本，不能保证 context 容量绝对不变。

router/shared 从原 graph segment 移到 eager 边界，会增加主机提交开销。
同时两个内存访问密集 kernel 可能争用 GPU 资源。因此目前实现的是正确的并发调度，
尚未证明吞吐获益，不能按 max(5.32, 5.27) 估计整步时间。

`decode_shared_fill_overlap_invocations` 应随符合条件的 decode 增长；
它不是硬件并发时间。profile 中应看到 fill 在独立 stream，shared GEMV 位于
其执行区间附近，routed GEMM 在 fill event 后运行。需计算 kernel 区间交集并
比较整步耗时，同时固定输入/route、上下文、cache 布局和 sampling 阶段。

## 验证与交接

CPU 回归包含 event 提交/汇合顺序、两类异常汇合、decode-only/inplace 保护、
真实 breakable decorator 的 capture stub 和新输入 replay，以及 callback 作用域恢复。
真实 SGLang CPU 容器检查还验证 quant 方法把 callback 传给 runtime 且只执行一次。
定向 6 项和完整 248 项 CPU 测试均通过，SM120 routing JIT 扩展编译、加载通过
（未执行 kernel）。容器无 GPU、无网络，限制为 2 CPU / 4 GiB 内存。

新增 GPU 回归覆盖 shared GEMV + direct fill 的跨 stream 输出、sampling、
breakable graph replay 的新 route/激活及 shared 分支异常后的下一次 cache reader。
这些 GPU 用例尚未执行。没有发送线上请求、重启服务或调整部署参数。

最终日志位于 `/home/jay/workspace/sglang/results/mailbox-shared-overlap-20261016/`。
GPU 正确性、真实模型数值与在线吞吐验证由用户执行。
