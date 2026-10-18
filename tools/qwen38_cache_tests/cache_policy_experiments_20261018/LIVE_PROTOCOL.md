# 当前 FP4-Local 固定路由实验

预先定义：两个独立录制 worker，代码与 Raft 设计两类合成请求，各录 1024 个完整 target verify sweep（每 sweep 48 层）。使用正常 MTP=2、prepared=1、adaptive CTA，以及当前 pack。输入 token 与请求参数一同存档。录制读回 CUDA route 会同步，录制 worker 吞吐不作为性能结果。

录制只在服务器就绪、指定请求发送前开启，避免建图合成路由混入。仅一个请求流；不拼接请求。生产启动脚本及 YAML 不修改，实验环境值放在独立 shell 脚本。GPU 使用 5090 D，ViT 所在 GPU 不重启。

CPU 阶段：五种策略 LFU、LRU、SLRU(0.5/0.7/0.8)，预算 6390。按前半段训练布局、后半段验证；128 sweep 预热。分别按 miss 和已测 fill 成本选择布局。成本仅为历史孤立 kernel 测量的代理量，不推断业务吞吐。

GPU 正确性：每个策略新进程，前 128 sweep 预热，随后逐次逐张量核验实际 pack payload。首个代码 SLRU(0.7) 已开始的校验保留 128 个计分 sweep；发现 CPU payload 读回开销较大后，在首个正确性进程结束、正式计时开始前，将后续校验预算设为 16 sweep（每层全张量，不抽选单独 tensor）。该调整仅减少正确性检查的重复数量，不改变或筛除任何性能样本。正确性运行与计时进程分离。

GPU 计时：每个策略、每轮新进程，从空 cache 回放相同 128 sweep 预热，计时其后相同 896 sweep。固定近均匀容量，总计 6390；真实 planner、策略、miss 搬运及 stream/event 都执行。每轮随机排列五个策略，种子 20261018，共 5 轮。主指标为整个区间 CPU 墙钟时间（含末尾 GPU 完成）；同时看 native gate miss、copy bytes 和 all-hit 计数。每个策略五次重复还用于估计同策略运行噪声，不把同进程内的 kernel 当作独立重复。

源权重使用与当前服务一致的 shared THP 缓存，避免回放工具默认 anonymous THP 导致额外 63 GiB 源副本以及不同内存路径。计时不与 CPU 布局搜索或线上服务并行。每个子进程最长 900 秒，失败保留日志并停止矩阵。

报告每个 workload 的原始重复值、同轮相对 LFU 比值、范围及描述性区间。两个 workload 不能建立跨业务普遍优势；无法分辨的小效应标为不确定。布局选择的额外 GPU 验证如需运行，将明确记录其独立比较范围。

这是 planner/transport 实验，不含 B12X、attention、GDN、shared expert、draft 计算，不据此声称端到端收益。不自动改变线上算法；恢复原语言服务并核验健康与 ViT 启动时间。

解释性补采：首轮看到耗时差大于 miss 数差后，计划在 GPU 正式计时的间隙，分别采 LFU/LRU 的短 transport trace。使用同一个 128 sweep 预热前缀和随后的 8 sweep；profile 结果与正式计时完全分开，不进入重复测量汇总。只用于检查 native planner/fill 的时间分布。
