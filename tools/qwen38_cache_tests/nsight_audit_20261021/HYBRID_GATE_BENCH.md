# 在目标 GPU 上标定 CPU 协同成本门禁

门禁比较同一层、同一路由和同一初始专家驻留状态下两条分支，从路由
决策之后到最终 BF16 结果可用的墙钟时间：

- 全 GPU：注册内存中的缺失专家填入 GPU pool，B12X FC2 直接写 BF16；即使存在 CPU 标定样本，也不再提交单独的 merge kernel。
- CPU 协同：GPU 只在选中 CPU 专家时写 mmap 请求；CPU worker 从 mmap
  读取激活、路由和权重，调用 native operator，写回结果；GPU 同时填充
  其余专家并计算，B12X FC2 写回时等待已有完成标志，从 mmap 读取 CPU 结果并在 FP32 累加值上合并后写 BF16。

两条分支的 GPU fill 和 CPU 计算可以重叠。标定使用各分支最终结果
可用时的 elapsed time，不把各阶段时间简单相加。旧的
`bench_qwen38_cpu_batch.py` 仅测 native CPU，不可用于成本表。

`hybrid-gate-cases-6388.json` 来自真实 route trace 的回放，只含真实出现的
路由与 LFU rejected 候选，**不含时间门限**。默认保留最常见的 100 个
代表配置，并按完整 workload key 分组；组未完整覆盖时不能发布该 key。
候选提取命令：

```sh
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_global_hybrid.py \
  --budgets 6388 --cpu-limits 0 1 2 3 --max-cases 100 \
  --cases-output tools/qwen38_cache_tests/nsight_audit_20261021/hybrid-gate-cases-6388.json \
  --output /tmp/qwen38-gate-route-replay.json
```

在实际 SM120 GPU、对应的 Qwen3.8 FP4 pack 和目标 CPU 物理核上运行：

```sh
PYTHONPATH=python python3 test/manual/kernels/moe/bench_qwen38_hybrid_gate.py \
  --manifest /absolute/path/to/manifest.json \
  --library /absolute/path/to/qwen38_cpu_hybrid.so \
  --cases tools/qwen38_cache_tests/nsight_audit_20261021/hybrid-gate-cases-6388.json \
  --cores 0,2,4,6 \
  --samples 9 --warmup 3 --margin 0.1 \
  --output /absolute/path/to/hybrid-gate-measurements.json
```

选择的 CPU 核须在进程 affinity 内且互为不同物理核；模型 pack、GPU
架构和源码 SHA256 均会校验。请用与生产一致的专家 slot 预算、CPU
核心、CPU 动态库、B12X 后端和运行环境测量。`SGLANG_QWEN38_FILL_CTA`
也沿用目标服务的配置（默认 8）；要求注册 DMA。脚本随机化每对 CPU/GPU
分支的运行顺序，检查最终数值，并保存所有样本。仅当同一个精确
workload key 的全部代表都稳定且 CPU 中位数比 GPU 至少快 `margin`
时，才把它写进与测量报告同目录的 `.calibration.json`。表在写出前
还通过运行时 `HybridCostModel` 的 schema 验证，并在 worker 退出和资源
清理成功后发布；已有标定表不会被覆盖。没有覆盖或没有收益的
key 不会进入表，在服务中仍走全 GPU。

使用表前还应在真实服务请求上做无 profiler 的多轮吞吐和延迟 A/B；
此微基准使用真实专家 pack 与路由形状，但激活和路由权重是基准输入，
从**设备路由决策完成**后开始计时，因此不含 planner kernel 及生产
请求的调度开销；预制计划和路由只用于避免 Python 构建开销污染
分支计时。它也不覆盖 prefill、draft、动态 resize 或生产环境中的
后台竞争。
本工作区没有本地 GPU，未产生任何实测 `cpu_us`/`gpu_us` 或成本门限。

单独量化全 GPU 层的固定输出开销，可在同一设备运行：

```sh
PYTHONPATH=python python3 test/manual/kernels/moe/bench_qwen38_hybrid_epilogue.py \
  --manifest /absolute/path/to/manifest.json --capacity 6388
```

它用已驻留专家比较旧 FP32 B12X + 独立 merge 和新 FC2 融合写回，
并先要求输出逐元素相等。上述结果尚需在目标 GPU 上实测。
