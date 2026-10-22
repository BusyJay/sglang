# 全局 LFU hit 快路径（2026-10-22）

当前 indexed planner 在路由验证和 FP64 热度更新之后，使用 CTA 统一判断：所有有效路由都驻留，且当前层无未访问驻留专家衰减到 `<0.5`。满足条件直接走 hit 路径；全 masked 也可满足，冷回收条件仍需检查。scan 参考实现继续走完整流程。

快路径保留 histogram 去重计数、全层频率衰减、底向上并行专家堆修复、当前层全局最小值刷新、路由 slot 映射和 hit-call 统计。它跳过 requested-route 压缩、CPU 成本表读取、全局 top-index 的 shared-memory 预取和整表回写、准入/淘汰、lookup 回写和 CPU mailbox staging。当前层 top 修复直接操作持久索引；当前层 heat/heap 通过 shared views 读取，其他层在同一个有序 compute stream 上保持稳定。未新增 launch、CPU route readback 或开关。

冷回收、任何 miss、非法路由都保留原有完整路径/错误处理。命中增加热度后仍会改变 heap/global minimum，因此没有跳过这些维护工作或使用近似热度。跨层全局 LFU 策略与固定成本表 gate 不变。只优化 indexed 实现；不声称达到旧逐层缓存相同策略/性能。

验证：

- SM120f 当前与优化前 CUDA 模块都编译、加载成功，见 `compile-sm120.log`。indexed 两版均为 64 registers、32096 bytes shared memory、0 stack/spill；快路径没有增加该 kernel 的资源预算。
- CPU policy / dispatch / resize：88 tests、279 subtests 通过，见 `cpu-tests.log`。
- 已扩展现有 GPU 测试：重复 hit/去重计数、全 masked 衰减、冷回收阈值与 reserved slot、hit 后全局 minimum 变化及跨层 miss、2/48/65 层 heap/position/top 一致性、全 hit 不触碰 mailbox。
- 本机无 GPU，上述 GPU 用例尚未执行，性能收益和 miss 路径额外判断成本待目标机实测。CPU 测试和编译不能替代 GPU 正确性/性能验证。

`planner_before.cuh` 是本轮修改前的准确快照，包含先前 fill-key / binary-search 改动，用于隔离本轮 hit 优化。前后都包含相同 heap helper。微基准新增 `--baseline-source`，交错比较 scan、优化前 indexed、当前 indexed，打印 P50/P95，并比较居民集合、FP64 heat、stats、CPU routes 和 fill expert 集合。

目标机运行：

```bash
PYTHONPATH=python python3 test/registered/kernels/ops/moe/test_qwen38_hybrid_plan.py -f
PYTHONPATH=python python3 test/manual/kernels/moe/bench_qwen38_hybrid_planner.py \
  --tokens 3 --top-k 10 --samples 200 --warmup 30 \
  --baseline-source tools/qwen38_cache_tests/bottleneck_review_20261022/hit_fast_path/planner_before.cuh \
  --calibration /target/new/calibration.json
```

另跑 tokens=1/2；保留真实非空校准表，以及空表诊断。输入/状态 reset 在 event 之外，synthetic all-hit/full-miss/free-miss 微基准只能隔离 planner 成本，不能替代 48 层服务 trace 和无 profiler 吞吐。需要检查 all-hit P50/P95 降低、miss 额外判断无明显退化，最后重校准并重测服务 MTP accept、ms/sweep、fill bytes 和 tok/s。
