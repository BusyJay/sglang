# 取消 mailbox ready 等待

原方案 1 的 expert 地址已固定，但 miss 仍需 GPU 发布 plan、CPU 更新统计并
写 ready 后，GPU 才开始搬运。本次移除这项不承载权重数据的 CPU 确认。

GPU planner -> alpha scatter -> fill -> 原专家计算顺序不变。
planner 不再向 host 发布 miss plan；fill 不再轮询 ready。
统计改为每层四个 int64 GPU 累计计数，snapshot/synchronize 时同步汇总增量。
保留跨 stream/cache slot 的 reader event、迁移保护、错误检查和 completion watchdog。
普通 prefill/sampling 的 transport worker 不受影响。

CPU 测试覆盖延迟统计、多层混合 hit/miss、多次同层 plan 的累计、重复快照、
resident alpha 字节计数、失败传播和 watchdog 超时。
GPU 回归已更新，覆盖跨 stream 排队与字节一致性、统计不丢失/不重复、
无 ready 下的 failure flag。由于线上正在推理，本次未执行 GPU 回归或压测。
SM120 CUDA 扩展已在无 GPU 容器内完成编译和模块加载，没有执行 kernel。
237 项 CPU 容器回归全部通过、无跳过；真实 SGLang 导入/配置检查已通过。

原始日志：`/home/jay/workspace/sglang/results/mailbox-no-ready-20261016/`。
host 直接跑 CPU 套件的尝试因 host 未安装 torch 有失败/跳过；最终验证使用
现有部署镜像的独立 CPU 容器，见 `cpu-container.log`。编译见 `compile.log`。

没有重启服务、发送推理请求、执行 GPU 测试或修改启动配置。
部署、GPU 回归与线上吞吐验证由用户执行，不能将离线编译视为 GPU 正确性或性能验证。
此修改没有取消整个源 pack 的 CUDA 注册，context 容量问题不在此次修复范围内。
