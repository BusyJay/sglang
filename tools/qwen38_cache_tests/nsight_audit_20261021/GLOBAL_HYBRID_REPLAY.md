# 同一全局缓存下，共算与非共算的搬运对照

2026-10-21。固定相同全局 pool、LFU 和预算，隔离 CPU 参与的影响。
基线使用全局 planner、空成本表，所有专家由 GPU 计算。

当前没有符合新路径版本指纹和完整配对口径的 CPU/GPU 标定样本。
已有 cpu-table-benchmark.json 是旧 native table 微测，不能代替完整路径
标定。因此真实选择器用空表时不选择任何 CPU 专家，两边专家放置和
搬运相同；这里不是两个实际服务实例的吞吐 A/B。

为判断 CPU 参与有没有搬运空间，另重放每调用最多选择 1、2、3 个
LFU rejected 专家的假设配置。选择仍按实际 planner 的 rejected 前缀
顺序执行，命中和 LFU 已准入专家不会交给 CPU；每个模式独立重放
整个请求及其 prefix，CPU 专家保持在 RAM，后续缓存状态可以不同。

## 6,388 slot 的结果

四请求各先重放 128 个 sweep，然后统计共 57,408 次层调用。
H2D 包含专家 payload 和 CPU FP32 结果；D2H 包含路由、BF16 激活和
FP32 路由权重。CPU native 只读 host pack，专家权重没有 PCIe D2H。

| CPU rejected 上限 | GPU fills | H2D GiB | 相对全局纯 GPU | D2H MiB | CPU 调用数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0，全局纯 GPU | 153,201 | 394.48 | — | 0 | 0 |
| 1，假设共算 | 124,051 | 320.50 | -18.75% | 565.61 | 37,868 |
| 2，假设共算 | 107,853 | 278.67 | -29.36% | 502.99 | 33,626 |
| 3，假设共算 | 98,430 | 254.33 | -35.53% | 460.51 | 30,749 |

最多 1 CPU 时，GPU expert H2D 减少 29,150 次填充，约 75.06 GiB；
CPU 结果 H2D 约 1.08 GiB，扣除后净减少约 73.98 GiB。
同时 37,868 次 CPU 调用覆盖 65.96% 的层调用，不能忽略 native
调度和 GPU 合并等待。CPU 专家逻辑权重读取量约 97.51 GiB，
这不是硬件 DRAM 计数，也不能用来推断实际 CPU 用时。

最多 1 CPU 时逐请求 GPU fills：

| 请求 | 全局纯 GPU | 假设共算 | CPU 调用数 |
| --- | ---: | ---: | ---: |
| short | 36,255 | 29,870 | 8,959 |
| raft | 36,662 | 29,996 | 8,891 |
| long-0 | 33,667 | 27,108 | 8,062 |
| long-1 | 46,617 | 37,077 | 11,956 |

## 各预算汇总

| 总 slot | 全局纯 GPU H2D GiB | 最多 1 CPU H2D GiB | 变化 | 最多 2 CPU 变化 | 最多 3 CPU 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6,388 | 394.48 | 320.50 | -18.75% | -29.36% | -35.53% |
| 6,390 | 394.24 | 320.46 | -18.71% | -29.32% | -35.48% |
| 6,643 | 371.73 | 313.61 | -15.64% | -24.15% | -28.64% |
| 7,006 | 348.11 | 309.93 | -10.97% | -15.92% | -18.09% |

CPU 上限越大，CPU 调用数反而下降，是各模式缓存演化不同的重放
结果，不能把它解释为多专家 CPU 本身更快。留在 CPU 的专家不挤占
pool，使其他 resident 更容易保留；CPU 候选和未来 miss 数也会变化。
因此这里没有简单从全 GPU miss 数中减去固定专家数。

## 对性能目标的含义

CPU 共算有额外减少专家 H2D 的空间，但这不是“CPU 共算更快”的证据。
必须用完整路径配对 elapsed time 选择子集，包含 worker 调度、激活
D2H、native 计算、结果 H2D 和 GPU 结果依赖。不能根据上表把 CPU
上限设为 1、2 或 3，也不能把 CPU kernel 微测填入完整成本表。

`replay_global_hybrid.py` 为了驱动实际选择器，在诊断进程内生成精确
匹配的 rejected profile 和排序哨兵。哨兵不是耗时，不作为标定文件
输出，不用于生产。获取 rejected profile 的 probe 只运行在状态副本
上，原状态每次实际调用只更新一次 LFU。

## 复现与产物

```sh
python3 tools/qwen38_cache_tests/nsight_audit_20261021/replay_global_hybrid.py \
  --output tools/qwen38_cache_tests/nsight_audit_20261021/transfer-replay-global-hybrid.json
```

JSON 保留四预算 × 四请求 × 四 CPU 上限的所有计数及输入源码哈希。
CPU 上限 0 的全部 16 组均与之前全局纯 GPU 重放 fills 完全一致。
所有模式检查 CPU/GPU 专家互斥；专家 H2D 单次 payload 为 2,764,800 bytes。
数据范围与 trace 缺失项同 TRANSFER_REPLAY.md；没有运行 CUDA 或 CPU
native，没有新增性能测量。
