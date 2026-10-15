# Decode sampling 与 GPU 直接填充

范围按用户最新要求收敛为 decode；未实现 prefill 搬运、arena、dense/sparse、
cache 复用或预热的改造。没有重启服务、发送推理/采样请求或执行 GPU 测试。

## 行为

原先 sampling 会触发逐层 plan D2H、CPU staging 和 H2D 的旧路径。
现在满足原单 token / top-k <= 32 / 无 arena 条件的 sampling 也进入 direct fill，
从冷 cache 的第一个采样 step 开始即可使用。非 sampling 的首次 warm-up 判定保留。
planner 顺便写入 count + 原始 expert IDs（排序、去重、排除 padding），
不导出 remapped slots，不复制权重到采样缓冲，也不改变 MoE 的路由或计算。

`DeviceRouteSampler` 预分配两份 `[layers, window, top_k+1]` int32 GPU/host 缓冲。
每层满窗口后，由前台在独立 stream 提交 pinned D2H，GPU fill 不等待此 stream。
collector 只等待 D2H 完成并提交批量 Python 样本，不调度权重搬运。
CPU 分析器展开批次后仍使用原先逐层、逐 cycle、逐 index 的序列和拒绝缺口规则。

切回 prefill/verify/旧 transport 前，先排空 reader 并交接部分采样记录；
旧 verify worker 在发布 sample 后通知前台，避免下一条观察超前进入分析器。
真正的 cache rebalance 和迁移保护不变。

bank 在 D2H、collector 读取和分析队列消费完成前不复用，这同时约束 Python
批量样本积压。观察线程太慢时仍采用原有的显式丢样计数/不完整窗口拒绝语义，
不会等待 CPU 采样确认来放行 GPU 权重填充。异常采样数据会使 runtime poisoned。

48 层、512 step、top-k 10 的逻辑 GPU 缓冲为 2162688 字节，即 2.0625 MiB；
pinned host 同样大小，另有事件和 Python 样本对象。启用 adaptive + fused planner
时才创建 sampler，buffer 纳入现有 device/pinned memory budget 统计并在 KV 预算前分配。
实际 context 容量还受 allocator 预留和对齐影响，不能仅按逻辑字节保证不变。

## 验证

- 242 项 CPU 容器回归全部通过，无跳过；真实 SGLang 导入、配置与 runner wiring 检查通过。
- 11 项相关 CPU 定向测试通过，覆盖完整/部分批次、原始 index、padding 数量、双 bank
  的 CUDA/CPU 所有权、队列拥塞、异常数据、混合新旧观察协议、缺口拒绝和 bank credit 释放。
- SM120 JIT 扩展在无 GPU、无网络的独立容器编译并加载通过；没有执行 kernel。
- 新增 GPU 回归覆盖从冷 cache 进入 sampling、跨 stream、重复/padding expert、
  原始 route 序列一致性、完整/部分窗口及 bank 复用，并禁止调用旧 `_host_plan`/staging job。
  已有 LFU/SLRU sampling 回归更新为所有 128 步均使用 GPU gate。GPU 回归尚未运行。

日志：`/home/user/workspace/sglang/results/mailbox-decode-sampling-20261016/`。
`cpu-container.log`、`targeted-host.log` 和 `compile.log` 分别记录上述最终结果。

## 用户后续验证

GPU 正确性和线上吞吐尚待用户验证，本次没有已测得的性能提升。
profile 应区分 prefill 与其后的前 512 个 sampling decode step。采样期间应看到
mailbox fill，而不是旧的 packed scatter/逐层 CPU plan 往返；window/transport
交接时仍有小规模 route D2H。检查 `adaptive_sample_drops`，正常负载应为 0，
并确认完整窗口之后仍能执行 adaptive evaluation / 原有条件下的 rebalance。
