# verify-experiments 实机验证与默认启用（2026-10-18）

已默认启用容量扩展、K=1 fused projection、warp planner、compact 日志。
reuse 保留实验入口，默认关闭。启用标准为存在可测量的单项或单样本收益，
不要求整机吞吐稳定提升；下表保留汇总中的负向结果。

## 可测收益与修复

- **容量扩展**：固定 6390 slots，热点层实际达到 256 slots。单个 27323-token
  请求达到 99.43 token/s；两轮合计均值为 93.20，未高于基线 95.13。
  首次影子分析预测减少约 11% misses，后台累计分析约 60s，基线约 36s。
  该项依据热点分配及正向样本启用，不宣称平均吞吐提升。
- **fused**：2/3-token verify 的 scratch 从 343040/514560 bytes 降为
  240640/360960 bytes，减少约 30%。短输入汇总 96.12 对 94.45 token/s；
  长输入和 Raft 汇总更低。图内内核耗时没有稳定改善。
- **warp planner**：24 组 miss 场景的图内耗时下降 0.30–4.18%，中位下降
  1.86%；全命中基本持平。LFU/SLRU、32/80/144/192/256/512 slots 的状态、
  eviction 顺序、plan 和采样边界与参考实现一致。
- **日志压缩**：5 条历史完整服务统计严格按 JSON 类型和值还原，体积减少
  68.90%。修复了 Python equality 合并 bool/int/float 和正负零的缺陷，
  递归比较类型与浮点 hex 值，补充回归。stats_snapshot 本身保持原结构。
- **reuse**：有效路由图内耗时约为 baseline 的 1.3–3.3 倍，重叠路由也更慢，
  因此没有默认开启。

## 验证结果

- 最终 **321 项 CPU、67 项 GPU 回归全部通过，GPU 无跳过**。
- 初次 112 组 projection 配置通过独立 dense reference、非法/重复路由和改变
  输入后的 graph replay。最大绝对误差为 0.00048828125。
- 默认启用后再次验证全部 112 组，并额外检查 unset mode 的 K=1 输出与显式
  fused 逐位一致。初次测速为 100 iterations × 3 rounds；最终复查采用 5 × 1，
  后者只用于正确性，不用于收益结论。
- 8 个全新对照 worker 的 32 个 2048-output 请求均保留；32/32 已知答案检查通过。
  最终默认组合另有 4/4 通过。它们是 smoke 检查，不是完整质量评测。
- 所有对照及最终默认 worker 的 KV 容量均为 **221248 tokens**。
- 最终组合实际完成 **218015-token prompt + 128-token generation**，
  cached_tokens=0、num_retractions=0，结束后 health=200；TTFT 为
  31.800s。
- 最终 worker 没有显式设置四个候选开关，使用源码默认值；其源码 hash 与
  [final-runtime-sha256.json](validation/final-runtime-sha256.json) 全部一致。

## 完整服务对照

RTX 5090 D / SM120，既有生产镜像，checkpoint revision
`b9ce18a6887e18c089fdea8509a33e63429a3300`。6390 slots、static fraction=0.90、
8192 chunk、MTP=2、draft W4A4、target W4A16、FP8 KV、context=220000 固定。

顺序：baseline / fused / planner / cache256 / cache256 / planner / fused / baseline。
每轮全新进程、相同 128-token warmup、固定 seed=103、temperature=0，
每请求先 flush prefix cache，全部 cached_tokens=0、无请求回撤。
短输入分别为 CPU cache 和中文 Raft；长输入为固定的 27323-token 实际仓库代码
审查输入，每轮重复两次。吞吐以 stream token 总数除以 stream 区间总时间，
TTFT 为算术均值。全部文本、stream 时刻和接受长度已保存。

| 模式 | 短输入 tok/s（2 样本） | Raft tok/s（2 样本） | 27K tok/s（4 样本） | 27K TTFT s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 94.45 | 100.46 | 95.13 | 6.400 |
| cache256 | 89.07 | 89.25 | 93.20 | 6.305 |
| fused | 96.12 | 84.87 | 89.90 | 6.383 |
| planner_warp | 88.00 | 89.55 | 92.11 | 6.439 |

最终默认组合另起一个 worker：短输入 88.67、
Raft 78.50 token/s；两个 27K 样本分别为
94.57 / 87.97。
该次主要验证组合、容量和实际请求，未因波动追加稳定性复测。

## 工件与复现

- [README.md](README.md)：默认值、回退开关和复现命令。
- [validation.json](validation/validation.json)：最终状态与默认配置。
- [sustained/comparison.json](validation/sustained/comparison.json)：全部正式对照及
  每轮 KV、采样峰值显存、allocator mapping retry、adaptive logs/counters。
- [defaults/](validation/defaults/)：最终源码默认组合、完整结果、218K 请求及日志。
- [graph-bench.json](validation/graph-bench.json)、[verify-kernels.json](validation/verify-kernels.json)：
  图内耗时与初次 projection 正确性/测速原始数据。内核测试不包含 expert fill 或服务开销。
- [package.json](validation/package.json)：原始提交包的校验和及 manifest，保留其启用前状态。
- service/ 与 service-ab.log.gz 是初始 1024-output 诊断。该长度不足 512 轮采样窗口，
  基线完成后中止下一 worker 的启动，未纳入正式 2048-output 对照。正式样本无剔除。
- 日志和 GPU memory CSV 使用 gzip 归档；采样显存峰值不是瞬时 allocator 峰值。
  两种日志格式的 canary/benchmark 消费路径均通过 CPU 回归。
