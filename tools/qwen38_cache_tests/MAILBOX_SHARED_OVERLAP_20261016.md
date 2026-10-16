# Decode shared expert / fill 并发

仅优化单 token CUDA decode 的独立 shared expert 分支。prefill、verify、
arena、dense/sparse、缓存复用和预热策略继续使用原路径。

## Profile 结论与本轮修正

926c764182 的整个 MoE eager 边界已在线采集：shared GEMV 与 fill 确实重叠，
但整步从上一份的 17.890 ms 增至 22.698 ms。shared GEMV 只有约 0.771 ms/step，
其中约 0.487 ms 与 fill 重叠；GPU 空闲反而从 1.194 增至 8.223 ms/step。
不同输入/route/cache 条件不是严格 A/B，然而 launch 时间戳明确显示多处 GPU
在等主机提交。原始数据与完整诊断在
`results/mailbox-overlap-profile-20261016/ANALYSIS.md`。

本轮去掉整个 MoE 的 eager 边界：router/topk、最终 gate/add 和后续算子回到
原 breakable graph segments。仅保留 quant 方法原有 expert transport 边界。
它的 capture stub 在前一个 segment 结束后录制 shared 小 graph；每次 runtime
提交 fill 时只 replay 该 graph，不再由 Python 逐个提交 shared 算子。

## 执行依赖与生命周期

```text
main graph: router/topk
compute:    planner/alpha -> ready -> shared graph replay -> wait filled -> routed MoE
main graph: gate/add -> 后续层
fill:                       wait ready -> fill -> filled
```

保留原 router、dispatcher、CUTLASS runner、fused gate/add 和 reduction。
运行条件仍是单 token decode、shared/routed 分离、非 inplace、TP=1、A2A none、
threaded direct fill。冷的非 sampling decode 如仍走旧 transport，shared 仅串行
计算一次。未启用 graph 的 eager decode 也直接计算 shared，不在服务循环捕获图。

ContextVar 只在模型 Python forward/capture 作用域内传递 `(call, original_input)`，
正常或异常均 reset。quant 边界将 original_input 作为单独 Tensor 参数传递，
由原 breakable wrapper 转成 weak view；不把激活藏在持久回放闭包中。
shared graph 固定读取这个地址，因此上一段 replay 每轮写入的新激活都会生效。
shared 输出由 call 强引用持有，供下一段 gate/add 使用；没有额外 shared 输出拷贝。

辅助 shared graph 复用当前 breakable capture 的 allocator pool，避免每层各自
保留独立 allocator slab。新增公开 helper `get_current_capture_pool()` 只返回
当前 capture 的 pool；非 capture 返回 None。使用共享 pool 的辅助 graph 必须
完成后才能开始下一段。现有 runtime computed event 在 shared replay、fill join、
routed reader 之后记录，满足该顺序。pool 随 segment/shared graph 对象存活。

graph capture stub 不读取尚未 replay 的 route，不进行 cache transport，也不执行
shared graph replay。图的录制及准备只发生在模型 graph capture 阶段。
ready/filled、reader event 的复用与异常 join 协议沿用上一轮，不增加 host ready 等待。

## 开销与验证

仍有每层一次 shared graph launch 的主机开销，以及 graph/输出/中间结果的存储成本。
共享 allocator pool 避免每层独立 slab，但不能保证 KV/context 容量绝对不变。
可隐藏的 shared 计算量级约 0.77 ms/step，不能用整段 5.32 ms 普通 GEMV 估计收益。

本轮完整 252 项 CPU 测试通过，包括真实 eager decorator 的 quant capture/replay、
新输入/route 和 phase 参数、shared callback 转发、capture pool 作用域、失败恢复、
事件提交/join 和 decode-only 保护。真实 SGLang 导入与参数无关 runner wiring 通过。
容器无 GPU/网络，限制 2 CPU / 4 GiB；日志在
`results/mailbox-shared-graph-20261016/cpu-container.log`。
CUDA kernel 没有修改，本轮未重复编译。

GPU 用例已更新为小 graph 路径，覆盖跨 stream、sampling、fresh route/activation、
共享 capture pool，并检查 replay 不再调用 Python shared 分支；尚未执行。
没有线上请求、重启或部署参数修改。

下一份 profile 应检查额外 shared graph launch、shared/fill 区间交集、GPU 空闲和
整步时间，特别验证此前 routed prefix-sum、gate/add 前的提交空隙是否收敛。
`decode_shared_fill_overlap_invocations` 仍只是路径计数，不等于硬件重叠。
