# Qwen38 adaptive KV / remote ViT 实测（2026-10-19）

三个补丁已完整应用并在生产 launcher 中启用。最终实测有小幅吞吐收益，
80K 阶段可执行专家槽位从 6390 增至 7001（+9.56%），历史图片可跳过远程
编码。没有把后台分析耗时、迁移提交时间或 profiler 下的吞吐当成性能收益。

输入包为 `qwen38-adaptive-kv-vit-20261018-r2-slim.tar.gz`，SHA256：
`9e5b8a4c3aa9ce001f5022de65f07571e1e6ed2b047a2b0e5ce768383c791bb7`。
基线 `18fdf4a7e403d74d8c7a3ee49e2e8f96c23ec77f` 与包要求一致；先在独立
checkout 用 apply.py 校验并应用，再逐文件验证 before/after 哈希复制到工作区。
包的原始快照见 `bundle-manifest.json`；本提交还包含下述实跑修复和格式检查。
其他原有未跟踪文件、补丁包没有加入提交。

环境：RTX 5090 D 32 GiB 运行 LLM，RTX 4080 SUPER 运行远程 ViT；驱动
615.78.08；镜像 `swallow-sglang:nightly-dev-20261018-b234b5c3`；
PyTorch 2.14.1+cu130。模型为 Qwen3.8-Flash-Next-FP4-Local，checkpoint
`b9ce18a6887e18c089fdea8509a33e63429a3300`，W4A16/B12X，MTP=2，
TP1，max-running-requests=1，prefill chunk=8192，mem-fraction-static=0.90。
完整参数随每轮 JSON 保存。

## 性能

原始源码通过 `git archive HEAD python` 固定在独立 source 目录运行。
原版和最终实现均从新进程启动，隔离到容器 8001 端口，避免 Hermes 的
约 107K-token 实际请求干扰。每组依次运行相同的三个提示词三轮，每次固定
输出 192 token，temperature=0、ignore_eos=true，记录流式 TTFT 和 decode。
比较 `original-round-{1,2,3}.json` 与 `final-round-{1,2,3}.json`，共各 9 个请求。
分析、测试和 profile 不与最终计时请求同时运行。

| 指标（9 请求中位数） | 原版 | 最终实现 | 变化 |
| --- | ---: | ---: | ---: |
| decode token/s | 101.05 | 103.29 | +2.22% |
| 端到端 token/s | 88.14 | 90.11 | +2.24% |
| TTFT ms | 288.04 | 279.32 | -3.03% |

这是当前三个短提示词的小样本结果，不是所有业务负载的稳定加速承诺。
MTP 接受长度在请求间变化，结果文件保存了接受统计。profile 内核时间和
decode H2D 字节数接近，不能据此声称 CUDA 内核本身获得显著提速。
`results-summary.json` 可通过 `python3 summarize_results.py` 重算。

`baseline-*`、`patched-round-*`、`pre-transport-fix-round-*`、
`patched-isolated-bench.json`、`contended-patched-bench.json` 和
`post-adaptive-bench.json` 是初测、存在其他请求或修复前的诊断，均没有混入最终表格。

## 实跑发现与根因

1. **启用 CPU replay 导致 KV 容量检查失败。** 原部署将 VLM cache 设为 0。
   改为 100 MiB 后，同一个设置还会启用旧 GPU embedding cache，并从 KV
   预算扣除 100 MiB。对 unpooled `language_only + zmq_to_scheduler` 路径，
   现在只保留 CPU replay，GPU embedding cache 和对应预算扣减均为 0；
   显式 GPU receive pool 和其他后端保留原来的 GPU cache 行为。
   注册测试验证了预算以及缓存拒绝驻留，真实图片请求验证了 replay。

2. **VMM 对齐侵蚀了最终容量的紧余量。** 排除上述重复预留后，6390 槽位
   在真实 VMM 分配下只能 profile 出 219136 token，而最终阶段要求 220032。
   将基础槽位调为 6382，保留 0.90 静态比例和 8192 chunk，实际可启动。
   三档真实预算为 7001 / 6635 / 6382。原版 KV profile 为 220608；最终
   220032 是请求的 220000 按 page64 向上取整。证据见 `startup-failure.log`。

3. **长请求后的回借会让下一块 prefill 无法分配。** 74K 请求后，decode
   把 reserve 从 8192 降到 3，回借至 80K；随后复用 74K 前缀的请求无法分配
   8192-token chunk，即使逻辑 available_size 还有 146048 token。
   现在保留已见的最大 batch reserve，避免在下个结果边界之前回借掉所需空间。
   `prefill-reserve-regression.txt` 保存旧逻辑 reborrow=true 与修复后的 false。
   `stage-failure.log` 保存真实失败，`stage-cycle-fixed.*` 保存修复后的全流程。

4. **新增槽位使短 prefill 误切换到慢传输路径。** 第三个提示词为 14 token、
   top-k=10，140 的路由上界大于原版每层约 133 槽位，却小于借入后的约 147。
   因而原版使用 arena 批量传输，未修复候选使用逐专家路径。
   三轮未修复 TTFT 为 407/386/331 ms，最终为约 281/268/271 ms。
   现在以原始逐层槽位作为 arena 选择上限，缩小时仍按实际容量保证安全；
   新增槽位继续参与 arena 的 resident reuse。禁用 arena 的配置保持原行为。

   | 第三个提示词的独立 prefill trace | CPU span ms | GPU window ms |
   | --- | ---: | ---: |
   | 原版 | 196.57 | 198.94 |
   | 修复前 | 334.63 | 335.31 |
   | 修复后 | 220.98 | 222.24 |

   profiler 会改变绝对延迟；这里只用于验证分支和成本。传输明细及完整 trace
   都已保留，见 `results-summary.json` 的 `prefill_diagnostics`。

5. **旧维护接口残留。** 完整 CPU 套件及固定路由 replay 工具仍调用已经移除的
   `_maybe_rebalance`。已迁移到 snapshot / result-boundary 接口；离线工具
   显式完成自己的 snapshot，服务维护路径不新增全局 CUDA 同步。

## 验证

- 完整 `tools/qwen38_cache_tests/run_cpu.py`：347 项通过，见 `cpu-suite.log`。
  使用镜像依赖；宿主没有 torch，不把宿主缺依赖结果当成代码失败。
- 注册的 stage allocator 测试：9 项通过；合并执行边界和 encoder 用例共
  21 项通过，见 `targeted-tests.log`。包含真实 paged allocator 和 GPU cache
  预算回归；另有 CPU regression 覆盖 prefill 路径选择。
- 真实 LLM：74K → 170K，跨过 80K/160K 两次扩容；flush KV 后短请求回借
  至 80K，再次 74K 正常扩容。未发生先前的 allocator OOM。
- 独立 RTX 4080 SUPER CUDA VMM 测试：300 次 0→1→2→0 映射循环，稳定虚拟
  地址、保留前缀数据、物理字节和映射数；三档映射数为 18/12/6，见
  `vmm-cycles.json`。这是小型合成 tensor 的底层映射测试，不等同于完整模型
  运行 300 次长上下文。
- 默认自适应策略：2048-token decode 后产生后台建议，下一结果边界迁移
  6190 个专家、淘汰 811 槽位，继续正常完成请求。`adaptive-migration.log`
  的 enqueue=0.240s 是 CPU 处理和 CUDA 提交时间，不是 GPU 迁移完成时间。
- 图片：红图首次编码；携带历史的后续轮次命中 CPU replay；改 cache_salt、
  改成蓝图均重新编码。回答 Red/Red/Red/Blue，原版回答相同；见
  `image-replay-isolated.json`、`image-original.json`。首次隔离图片测试曾与
  warmup benchmark 交错，故不比较这些 JSON 的端到端耗时。
- isort 7.0.0、ruff 0.15.1（F401/F811/F821/F823/UP037 与 format）、
  `git diff --check` 已检查。生产脚本的最终版本见 `launcher-common.sh`
  与 `launcher-fp4.sh`；测试结束恢复原端口和当前工作区源码。

当前实际部署限制为 TP1、并发 1。本次没有声称验证 TP2、取消/错误的跨 rank
集成，或并发多请求的所有 allocator 交错；这些限制不能由 CPU mocks 替代。
长 prefill 中还观察到 PyTorch expandable-segment allocation retry 警告，
请求均完成；日志保留了这些内存压力，未把它们隐藏或称作已消除。

## Profile

完整 CPU/GPU Chrome trace 以分片形式提交，以遵守仓库每文件 1500 KiB 的
检查。每片不超过 1400000 bytes，`profiles.json` 记录原文件 SHA256 和顺序。
这只是二进制拆分，未裁剪事件。恢复命令：

```bash
python3 tools/qwen38_cache_tests/adaptive_kv_vit_20261019/restore_profiles.py
```

恢复后的 `final-fixed.trace.json.gz` 是最终实现的 profile，
`original-isolated.trace.json.gz` 是原版隔离对照；另外三份 case2 trace
保存原版、回退和修复后的 prefill。`final-fixed-triage.txt` 和
`original-isolated-triage.txt` 包含 kernel / overlap / fuse 三表，
`*-profile-summary.json` 保存 decode 统计。单 trace 的 overlap 结果是有限的，
没有把它解释成已证实的新 overlap 优化。
