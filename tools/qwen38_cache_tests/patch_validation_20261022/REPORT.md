# optimization_patch_20261022 应用与目标机验收

日期：2026-10-22（Asia/Shanghai）。基线：`9321a3f4504534811676c4d1f39d62341089e610`。补丁已应用，未创建提交、未部署到正式推理服务。

## 结论

**两种缓存都能实际运行，但本次没有同时通过用户要求的性能验收。**

- 正确性与集成：107 项 CPU 测试 / 297 个子用例，20 项 GPU 测试 / 28 个子用例全部通过。普通 scale / CSF 原生服务、独立 scalar 参考、GPU schema 3 查表、B12X mask、FP32 merge、真实 MTP serving、KV/专家缓存短→长→短切换均有执行证据。
- 全局缓存：两轮短 decode 中位数约 -0.99%，自然聊天约 -1.34%；170k TTFT 持平。差异较小且运行间存在波动，**不能签署严格“无性能回退”**。
- 分层缓存：最终代码和最终成本表对同统计配置的旧分层缓存，短 decode -1.68%，自然聊天 -3.28%；**“对比分层缓存自身有改进”未通过**。较早轮次曾有正收益，最终确认并不稳定，不能选取最好的一轮作为验收结论。

`performance.json` 的 `status=passed` 仅表示请求完成且输出 token 数符合要求，不表示性能达标。校准分支获利也不等于端到端获利。

## 环境与对照方法

- RTX 5090 D 32GB，Ryzen 9 9950X，CPU 物理核 2–5。已有本地镜像 `swallow-sglang:nightly-dev-20261018-b234b5c3`；完整 image ID 见 [image-id.txt](image-id.txt)。Python 3.12.15 / PyTorch 2.14.1+cu130 / CUDA runtime 13.0 / FlashInfer 0.7.2。
- 本地 `nvidia/Qwen3.8-Flash-Next-FP4-Local`，revision `b9ce18a6887e18c089fdea8509a33e63429a3300`，真实专家 pack；容器 `--network none`，无权重下载。
- threaded runtime，adaptive LFU，B12X W4A16 decode / NVFP4 prefill，MTP 2、top-k 1、draft 3，名义 6388 专家槽位；KV loan 后实际可达 7006 槽位。KV stages 80k / 160k / 220k，chunk 8192，context 220k，mem fraction 0.90，分段 decode graph。
- 1024+64 salted 输入 / 512 输出，3 个短请求取中位数；自然聊天 / 512 输出；170064 输入 / 8 输出；随后 1088 输入 / 512 输出。固定输入、seed 42，每个 variant 重启服务且先预热。长输入只有 8 个输出，**不将其 decode 速率作为持续吞吐指标**。
- baseline 是独立原提交 worktree。全局 baseline 显式启用原全局 CPU/GPU 模式及原 schema 1 成本表；当前全局显式 `cpu_hybrid_mode=global`，加载重新实测的 schema 2 表，未退回原分层 GPU 模式。
- 第一轮关闭周期统计；第二轮前后都设置 `stats_log_interval=19200`。全局以 before→after、after→before 的顺序复测。未锁 GPU 时钟，轮次和 MTP 接受率均有变动；有限样本不足以证明约 1% 差异的统计显著性。

## 全局缓存：两轮合并

短请求合并 6 个请求取中位数；其他指标取两次同 workload 的中位数。吞吐越高越好，TTFT 越低越好。

| 指标 | 原全局缓存 | 当前全局缓存 | 变化 |
|---|---:|---:|---:|
| 短请求 decode 中位数 | 102.843 tok/s | 101.820 tok/s | -0.99% |
| 自然聊天 decode | 103.508 tok/s | 102.125 tok/s | -1.34% |
| 170k TTFT | 30.691 s | 30.689 s | -0.01% |
| 长输入后短请求 decode | 97.794 tok/s | 96.614 tok/s | -1.21% |
| 长输入后短请求 TTFT | 0.902 s | 1.453 s | +61.13% |

首轮长输入后短请求 TTFT 为旧 0.898s / 新 2.019s；第二轮为旧 0.905s / 新 0.887s。保留首轮异常，不把它从合并结果删除；未采集足够样本估计该恢复过程的尾延迟。

最新全局快照：57600 次 runtime 调用，56256 次 device planner 调用，7850 次 CPU 调用、18327 个实际 CPU 专家；5 次 resize、24916 个累计保留专家。快照最终 KV stage=0，global slots=7006。计数满足 `misses = admitted + rejected`、`rejected = GPU miss experts + CPU experts`。

原始记录：[before-global](before-global/performance.json)、[after-global](after-global/performance.json)、[before-global-r2](before-global-r2/performance.json)、[after-global-r2](after-global-r2/performance.json)。

## 分层缓存：最终 A/B/C

A 为原分层 GPU；B 为最终代码强制零分流；C 为最终代码真实校准分流。三者均使用低频统计，B 仅执行短请求组。

| 指标 | A 原分层 GPU | B 新框架零分流 | C 真实分流 |
|---|---:|---:|---:|
| 短 decode 中位数，tok/s ↑ | 104.474 | 105.257 | 102.715 |
| 自然聊天，tok/s ↑ | 103.549 | 102.320 | 100.151 |
| 170k TTFT，s ↓ | 29.985 | — | 29.852 |
| 长输入后短 decode，tok/s ↑ | 95.702 | — | 93.520 |
| 长输入后短 TTFT，s ↓ | 0.843 | — | 0.793 |

最终 A→C 的变化：

| 指标 | A 原分层 GPU | C 真实分流 | 变化 |
|---|---:|---:|---:|
| 短请求 decode 中位数 | 104.474 tok/s | 102.715 tok/s | -1.68% |
| 自然聊天 decode | 103.549 tok/s | 100.151 tok/s | -3.28% |
| 170k TTFT | 29.985 s | 29.852 s | -0.44% |
| 长输入后短请求 decode | 95.702 tok/s | 93.520 tok/s | -2.28% |
| 长输入后短请求 TTFT | 0.843 s | 0.793 s | -5.98% |

最新 C 快照：57600 次 runtime 调用中，10 次实际 CPU 分流、20 个 CPU 专家；启用混合路径的分支累计原始 miss=20723、GPU fill=20703，满足 `original misses = CPU experts + GPU fills`。GPU 回退分支也已计入总搬运量；最终 decode copy bytes=454643712000。这证明真实执行，不是空表/零分流冒充通过；同时也显示 CPU 覆盖率很低。

初轮 C 及中间版本包含不同的表和代码指纹，仅保留作历史诊断，不能与最终结果混用。中间轮 `after-layered-r2` 曾测到短 108.19、自然聊天 111.57 tok/s；最终确认没有稳定复现。该中间轮的 GPU 回退搬运量统计不完整，**不得用于传输量对比**。

原始记录：[A](before-layered-r2/performance.json)、[B](after-layered-zero-confirmed/performance.json)、[C](after-layered-confirmed/performance.json)。每个目录保留启动命令、全部流式事件、输出文本、MTP 接受率及 server log。[summary.json](summary.json) 含跳过前 64 token 后的吞吐、平均 sweep 时间和流式 P95；流式事件含网络与调度因素，并非 kernel profiler 时间戳。

## 校准和 GPU 微基准

全局表：[calibration.json](calibration.json)，schema 2。180 个 M3 + 72 个 M1/2 分支代表，11 对样本、4 次预热、5% margin；接受 45 个 key，M1/2 没有通过的 key 则回退 GPU。使用新 CPU 库和当前 runtime fingerprint；没有伪造或修改旧表的 metadata 来绕过校验。

分层表：[layered-calibration-confirmed.json](layered-calibration-confirmed.json)，schema 3，最终 8 个获准 key。对真实服务中的 500 个代表，在原层缓存状态和权重保存/恢复后，随机交错执行 C=0 与 CPU 前缀 1–3；保留 shared-expert overlap、真实 mapped native CPU service、GPU fill/B12X/merge。每个候选 4 次预热和 15 对样本，共 18555 对测量；所有配对均校验输出和实际 CPU/fill 专家集合，最大输出绝对误差 0.0004882812。同一个精确 key 至少 2 个代表均通过中位数、P95、胜率和 margin 才授权。

这些诊断重放会同步并增加 GIL/调度扰动；校准运行的服务吞吐不参与性能结论。强制前缀跳过成本表查找，实际端到端测试仍必须单独通过；精确键覆盖率、未知键回退及 MTP 路由变化均会影响真实收益。

[planner-benchmark.log](planner-benchmark.log) 为 M3/top-k10、45 行实表、100 个样本的 synthetic CUDA-event 微基准。优化前快照为包内 `planner_before.cuh`，它已包含 fill-key 改动，**不是原 Git 基线**。全命中 P50 16.032→12.320µs；满缓存 miss 58.816→59.424µs；空闲槽 miss 22.240→22.608µs。对应 P95 见日志，miss 尾延迟未改善；不得据此声称完整服务加速。

## 应用之外的补充改动

- 暴露分层校准 metadata 供真实服务采集；指纹补入实际 quant dispatch 源文件。
- 无获准 `(layer, tokens, slots, shared overlap)` 的静态配置直接走原分层 GPU 路径，避免无用 FP32/merge。
- 混合静态回退后，GPU 搬运量以总原始 misses 减实际 CPU 专家统计；copy-call 计数遵从实际 GPU fill。新增 CPU/GPU 回归覆盖同层回退、CPU-only miss、未知静态配置和 schema 3 查表/B12X mask。
- 修正一项新增测试中的 E741 命名，格式化新增改动。其余 5 项 Ruff E741 在原基线已存在，见 [ruff-final.log](ruff-final.log)。未声称全仓库 lint 全绿。

最终可应用补丁：[final-applied.patch](final-applied.patch)。含原包的 41 个源/测试/设计文件及本次补充；不包含本证据目录。独立 index 未修改用户真实 index；重新从原基线应用并逐文件比对 SHA256，见 [final-patch-check.log](final-patch-check.log)、[final-source-manifest.json](final-source-manifest.json)。

## 验证证据

- [cpu-final.log](cpu-final.log)：107 passed，297 subtests；[gpu-final.log](gpu-final.log)：20 passed，28 subtests。
- [cpu-scalar.log](cpu-scalar.log)：30 个独立 scalar / partition / error 检查，max_abs_error=0.00024414062。
- [cpu-csf.log](cpu-csf.log)：299 个 native/CSF、partition、fallback、routing、malformed-input 检查。
- [cpu-layered-service.log](cpu-layered-service.log)：普通 scale 和 CSF 的连续页复用、M=1/2/3、重复 route、非法请求检查。
- 两种真实服务均使用 decode graph 且完成 80k→160k→220k→80k 的 KV/专家池生命周期。长输入中出现的 allocator 分配重试在旧/新版本都有发生，请求成功完成。
- 未执行全量模型 accuracy benchmark、新 Nsight trace 或多小时压力测试；本报告不声称这些项目已验收。

## 复现

本目录提供已测旧/新 CPU 库、校准表、实际执行脚本和原始结果。已有 5090 空闲时使用新输出名，脚本拒绝覆盖保存的测量：

```bash
# 本机基线 worktree；若已清理，可重新创建并设置 QWEN38_BASELINE_ROOT。
git worktree add --detach /tmp/optimization-validation-20261022/baseline 9321a3f4504534811676c4d1f39d62341089e610

# 在本目录运行；baseline worktree 已存在则不必重复上一步。
bash replay.sh replay-before-global --baseline --mode global --calibration /results/baseline/tools/qwen38_cache_tests/plan_20261022/calibration.json --stats-interval 19200
bash replay.sh replay-after-global --mode global --calibration /results/calibration.json --stats-interval 19200
bash replay.sh replay-before-layered --baseline --mode off --stats-interval 19200
bash replay.sh replay-layered-zero --mode layered_zero --stats-interval 19200 --short
bash replay.sh replay-layered --mode layered --layered-table /results/layered-calibration-confirmed.json --stats-interval 19200
```

`run_variant.executed.py` 为本次实际执行的驱动；`run_variant.py` 只将旧/新 CPU 库改为本目录保存的同内容二进制，便于重放。表与当前源码、二进制和硬件绑定；后续修改实现应重新校准。真实 calibration hook 见 [calibrate_site/sitecustomize.py](calibrate_site/sitecustomize.py)，global gate 参数见 [run_gate.py](run_gate.py)。重新校准应使用新的结果目录，保留本次原始数据。

建议后续工作先解决分层精确键覆盖率和原 GPU 路径相比的完整成本门槛，再采集足够的交错服务重复；全局还需定位小幅吞吐差异与 miss P95。当前不建议把这份结果作为两项性能目标均达标的发布依据。
