# 真实 route trace 搬运量对照（2026-10-21）

目标是追平并超过非共算版本。此次基线明确为非全局、逐层独立 GPU LFU，
不是旧共算版本。新路径使用全局 GPU planner，空成本标定表、CPU 不参与。
这先检查取消 decode arena 后是否减少专家 H2D，并不代表已通过吞吐验收。

此固定均分对照已补动态基线，见 [ADAPTIVE_TRANSFER_REPLAY.md](ADAPTIVE_TRANSFER_REPLAY.md)。
动态基线下优势明显缩小，7,006 slot 全局 H2D 回退 1.95%；
下文固定分配结果不能代表非共算最好状态。

## 方法与数据

重放 `verify_reuse_diagnosis_20261018/service/verify-routes.jsonl.gz` 中的
short、raft、long-0、long-1 四个真实请求：共 1,708 个完整 48 层 sweep，
81,984 次层调用，每次 3 token × top-10。不同请求分别从空 cache 起步，
不拼接请求，也不将无标签 graph setup 或独立 warmup 请求计入结果。

`replay_transfer.py` 直接提取生产源码中的 `apply_plan` 和
`hybrid_plan_kernel`，编译为主机函数并逐调用执行。基线保留 resident
FP32 热度、LRU tie break、按专家 ID 排序的逐层准入；新路径保留 FP64
全专家热度、LFU 准入判断及 rejected 的强制全局放入。每次新路径 GPU
miss 只计一次 fill，没有恢复拷贝。主机模拟不验证 CUDA 并发或性能。

相同总专家 slot 预算，基线均匀分配给 48 层；新路径共享全部 slot，
另保留已有 B12X zero slot 0。6,390 是 route 采集命令中的预算；
6,388、6,643、7,006 对应已有 staged KV 验证预算。预算固定重放，
不伪造 trace 没有记录的 resize 或 prefill 状态。

按已有 pack-manifest，单专家搬运量 **2,764,800 bytes**：

- w13_weight：1,638,400 bytes。
- w2_weight：819,200 bytes。
- w13_blockscale_swizzled：204,800 bytes。
- w2_blockscale_swizzled：102,400 bytes。

专家在一次调用内按 distinct 集合填充，共享 token 路由不重复计拷贝。
下面的 GiB 为逻辑 H2D payload / 2^30，不是物理 PCIe 总线计数。

## 预热后结果

各请求先重放自己的前 128 个 sweep，再统计后续，共 57,408 次层调用。

| 总 slot | 非共算逐层 fills | 新全局 fills | 非共算 H2D GiB | 新全局 H2D GiB | 搬运变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6,388 | 175,596 | 153,201 | 452.15 | 394.48 | -12.75% |
| 6,390 | 175,495 | 153,108 | 451.89 | 394.24 | -12.76% |
| 6,643 | 165,504 | 144,366 | 426.16 | 371.73 | -12.77% |
| 7,006 | 151,928 | 135,194 | 391.20 | 348.11 | -11.01% |

6,388 slot 下逐请求结果：

| 请求 | 非共算 fills | 新全局 fills | 非共算 H2D GiB | 新全局 H2D GiB | 搬运变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 40,185 | 36,255 | 103.47 | 93.35 | -9.78% |
| raft | 43,224 | 36,662 | 111.30 | 94.40 | -15.18% |
| long-0 | 40,006 | 33,667 | 103.01 | 86.69 | -15.85% |
| long-1 | 52,181 | 46,617 | 134.36 | 120.04 | -10.66% |

6,388 slot 下 distinct 专家命中率由 **85.86% 提高到 87.66%**。
所有四预算 × 四请求组合的搬运量都下降，下降范围 8.17%–15.85%。

`lfu_admitted` 与 `forced_placements` 分开统计：6,388 slot 下预热后
95.11% 的 GPU fill 来自 LFU rejected 后强制放入。这说明初始 LFU
准入意愿并不能代替最终 GPU 放置；这些专家在当前路由序列下全部
搬到 GPU 后，整体搬运量仍小于逐层基线。它也说明 rejected 候选范围
很大，不能凭“被拒绝”推断 CPU 计算一定有益。

## 冷启动敏感性

不丢弃任何请求内 prefix，统计完整 81,984 次层调用：

| 总 slot | 非共算 H2D GiB | 新全局 H2D GiB | 搬运变化 |
| --- | ---: | ---: | ---: |
| 6,388 | 674.84 | 599.14 | -11.22% |
| 6,390 | 674.49 | 598.83 | -11.22% |
| 6,643 | 639.63 | 570.31 | -10.84% |
| 7,006 | 592.69 | 541.04 | -8.72% |

包含冷启动后，各请求组合仍无搬运回退，下降范围 7.87%–13.08%。

## 性能目标与证据边界

这次结果支持继续采用无 decode arena 的共享 pool：已观察到的强制
替换代价没有抵消整体命中率收益。但 **搬运减少 12.75% 不等于吞吐
提高 12.75%**，也不能据此宣布超过非共算版本。

基线是固定均匀逐层预算，没有模拟非共算版本的 adaptive rebalance；
trace 未记录初始生产 cache、prefill、draft、接受 token 状态或 KV resize。
因此本次验证覆盖固定预算 target verify 子路径，不是完整部署 replay。
相同专家 slot 预算也没有扣除新路径的小型 metadata 显存。

下一步端到端验收必须以关闭共算的真实服务配置为基线，保持模型、
请求、KV 阶段和显存预算一致。先测新全局路径全 GPU 时的吞吐与 planner
耗时，再用完整配对标定开启 CPU，确认其新增调度和结果合并成本被覆盖。
当前环境没有可用 GPU，也没有新路径完整配对标定；本报告不提供虚构
CPU 节省字节数、PCIe 带宽或吞吐预测。

## 复现与验证

```sh
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_transfer.py \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay.json
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_transfer.py \
  --warmup 0 \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay-cold.json
```

两个 JSON 保留每个请求/预算的结果、逐层计数、全局 occupancy 和
trace/manifest/planner 源码 SHA256。主机 g++ 编译使用
`-ffp-contract=off`。提取的基线函数额外通过 1,000 个随机窗口与独立
LayerExpertCachePolicy 对照，FP32 resident 热度、miss 和 slot 映射一致；
新 planner 之前已通过 1,000 个随机窗口与独立全局放入参考对照。
