# 动态非共算基线对照

2026-10-21。此前固定均分逐层 cache 的 11%–13% 优势不足以回答
“超过非共算最好状态”。本报告用已有动态分配器重新对照。

## 重放方法

逐层 GPU planner 仍提取实际 `apply_plan` 主体，FP32 resident LFU 和
LRU tie break 不变。动态分析直接调用生产
`qwen38_cache_analysis_worker.py`，使用其候选 shadow cache、前缀预热、
3% min_gain 和 8-window payback 门限。准入顺序、热度和 age 恢复与
生产一致。候选上限测试默认 256 与扩大到 512 两组。

测试观察窗口 32、64、128、256、512 sweep；短窗口间隔 0，默认
512-window 的间隔 4096。每个请求仅 402–442 sweep，默认配置未形成
完整窗口，无法产生调整，所以另检查短窗口调优结果。

每次决定只看刚完成的历史观察窗口，在该 sweep 之后立即应用，
不输入未来路由，不估计真实分析线程/进程的调度延迟。立即生效偏向
动态基线。窗口和候选上限在重放结束后按请求择优，形成“已测试配置
中最好”的比较包络，不能称为所有可能算法或真实吞吐的最优。

调整保留每层最热 resident，以 age 降序打破并列，按新分区重新排槽；
使用生产 `relocate_slots`，路径和环都计算实际 slot copy。环的 scratch
save/restore 也计入。被丢弃专家的后续重新加载由正常 miss 统计，
不把丢弃本身重复算一次 H2D。H2D 和 D2D 分开报告，不用总字节直接
推断总耗时。

## 与最佳已测动态基线比较

同样四请求各重放 128-sweep prefix 后统计，共 57,408 次层调用。
下表按每个请求 H2D 最少的动态配置择优，再按预算汇总。

| slot | 动态非共算 fills | 全局纯 GPU fills | 全局 H2D 变化 | 动态迁移 slot copies |
| --- | ---: | ---: | ---: | ---: |
| 6,388 | 156,974 | 153,201 | -2.40% | 24,167 |
| 6,390 | 156,780 | 153,108 | -2.34% | 30,131 |
| 6,643 | 146,458 | 144,366 | -1.43% | 37,396 |
| 7,006 | 132,611 | 135,194 | **+1.95%** | 33,271 |

6,388 slot 的全局 H2D 优势由固定均分对照的 12.75% 收窄至 2.40%。
7,006 slot 的全局 H2D 已发生回退；各请求回退 short +2.93%、
raft +3.62%、long-0 +1.24%、long-1 +0.38%。6,643 slot 的 short
也回退 +0.14%。因此不能再将固定均分对照中的“全部请求有优势”
用于动态非共算基线。

动态基线的迁移量单专家按 2,764,800 bytes 计，四预算分别约
62.23、77.59、96.29、85.67 GiB D2D。D2D 与 H2D 带宽/重叠不同，
不能说这些 D2D 字节一定抵消了 H2D 优势。原 JSON 还保留按
H2D+D2D 总 payload 择优的配置作为补充，非延迟成本模型。

第一轮 128-sweep 边界的迁移作为预热前缀结束后维护，不计入后续
评分的 D2D，保留在 events 中；后续边界的迁移均计入。各配置的
预热 resident、动态布局不同，这是完整独立重放产生的真实模拟状态。

## 对当前优化的结论

共享全局 pool 的搬运收益并非在最好逐层配置下普遍成立。目前不能
宣称已达到追平/超过非共算性能的目标。全局 LFU 使用 per-expert
历史热度、不同并列规则和实时全局容量竞争，动态逐层 LFU 的优势
不只是是否共算；GPU 全局扫描和 CPU 依赖仍要实际测量。

同一全局缓存内 CPU 子集的诊断对照见 GLOBAL_HYBRID_REPLAY.md。
它可减少 GPU expert H2D，但没有完整配对标定，不能用来证明 CPU
弥补了上述端到端性能差距。

## 复现

```sh
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_adaptive_transfer.py \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay-adaptive.json
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_adaptive_transfer.py \
  --windows 32 \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay-adaptive-fast.json
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_adaptive_transfer.py \
  --windows 64 128 --candidate-max 512 \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay-adaptive-wide.json
```

原 JSON 保留每次 evaluation、target、迁移量、最终布局和输入哈希。
`transfer-replay-adaptive-best.json` 为上述三组结果按请求择优汇总。
随机迁移检查确认 resident payload 身份、热度、age、clock 和 eviction
计数保持一致。此报告仍是 target verify 固定预算主机重放，没有初始
生产 cache、prefill、draft、resize 或 CUDA 并发/吞吐数据。
