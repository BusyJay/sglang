# 跨请求 DFlash prefetch 合批

2026-11-06。本机 M5 Ultra 256 GiB，MLX 0.32.4。基准引擎 `00f5b6c890`；
候选文件 hash 见 `prefetch_measurements.json`，没有改变量化或模型方程。
checkpoint、D3、BF16 KV/A16、TF32=0、eval groups 2/4/2、cache 1 GiB 和进程
资源预算均与 `RESULTS.md` 相同。

## 找到的实现缺口

`decode_batch` 原来在遍历每个输出时立即调用 `_prefetch_round`。两个请求即使
在同一步准备好了下一轮，仍分别 draft、分别执行单请求 target verification，
没有利用已有的多请求共享投影接口。它不是硬件不能并行，也不是接受率低。

现在先收集本步可预取的私有轮次，按 proposal count 合批 draft，并用一次
`verify_mtp_requests` 共享 target 投影。copy proposal 仍按原请求构造，允许
不同剩余预算。链、draft state、target cache clone、offset、特征和 position
都保持请求所有权；只有确认 sampler anchor 后才安装预取状态。
没有让多个请求共享可写 cache，也不等待尚未准备好的请求。单个 ready 请求
继续直接使用原单请求执行机制。新的合批行为默认使用，没有增加诊断开关。

## 完整模型 ABBA

同一份 resident 权重，分别测不同 prompt 双请求、相同 prompt 双请求和单请求。
每臂预热，再测 192 输出 token；每请求排除 prefill/首 token，计时 191 个 decode
区间。双请求数字是总共 382 个 token 除以同一段墙钟时间。

| 场景 | 原引擎 tok/s | 合批 tok/s | 变化 |
| --- | ---: | ---: | ---: |
| 两个不同 prompt，总吞吐 | 72.0309 | 76.0408 | +5.57% |
| 两个相同 prompt、验证边界对齐，总吞吐 | 71.8449 | 93.2543 | +29.80% |
| 单请求 | 70.8977 | 70.8445 | -0.075%，基本持平 |

不同 prompt 的原始两次 A 为 72.36/71.70，B 为 76.43/75.66；对齐双请求
A 为 71.93/71.76，B 为 93.25/93.26。每次输出 token、全部非计时 speculative
统计和最终 offsets 一致。对齐请求能更频繁合批，不能将 93.25 当作普通
双请求保证，更不能当作单请求达到 90。

峰值分配从约 173.647 增至 173.722 GiB，增加约 77 MiB 的并行中间状态。
单请求约 172.534 GiB。两台 engine 对象共享同一 drafter/target 权重。
系统已有 swap；该整个 ABBA 前后从 1406.81 增至 1558.06 MiB，最终 HTTP
测试期间也有变化，均在原始数据中记录。它们是系统级观测，不能由一次前后差值
归因于某个实验臂；不声称系统内存行为完全不变。没有调整系统 wired 参数，
没有同时加载两份完整模型，也不继续增加并发来施加内存压力。

## 数值、缓存与服务验证

- 额外 64-token 完整模型 A/B，分别使用 BF16 和 FP8 KV；三个场景的最终 target
  cache 逻辑内容逐位相同，token、接受计数和 offsets 一致。包含所有层和缓存槽，
  不只比较 cache 长度。此组仅作正确性验证，冷态耗时不混入上表。
- 97 个 registered MLX 设备测试通过。新增测试覆盖不同请求 offset、剩余预算、
  copy/draft 混合、拒绝、sampler edit、移除/恢复、full-logit 和 token-only 模式切换。
- 最终真实 HTTP 17 个顺序 case 和两个并发请求，输出 token、logprob、top-logprob
  及 finish signature 与上一版服务完全一致；terminal abort 和后续 prefix/槽位复用
  也通过。覆盖 chunked prefill 和 2563-token prompt 的 prefix hit。
- HTTP 双请求端到端总吞吐 58.09 → 61.49 tok/s，包含请求和 prefill 成本，
  与 engine decode 的口径不同。这是前后服务各一组观测，不冒充 HTTP ABBA。
- 两个生产/测试文件格式检查通过；DFlash 引擎 Ruff 无诊断，测试文件仅保留
  基准已有的一条诊断。整理后的 benchmark 脚本 Ruff 与静态编译通过。

逐 token 原始秒数、顺序、内存、完整输出和统计在 `prefetch_measurements.json`；
BF16/FP8 cache 对照结果也在该文件。最终 HTTP 摘要是
`http_prefetch_summary.json`。完整本机日志分别位于 `prefetch_abba.log`、
`prefetch_cache_check.log`、`prefetch_fp8_check.log` 和 `http_prefetch_final.json`。

## 复现与边界

服务仍使用 `../decode_gain_20261105/launch_default.sh`，监听 `0.0.0.0:39007`。
执行以下全模型脚本前需另行安排停机；脚本检查端口，拒绝同时加载第二份模型：

```sh
PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 \
 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200 \
 /Users/jay/workspace/prod/common/sglang-metal/bin/python \
 benchmark/mlx/engine_optimization_20261106/prefetch_abba.py \
 --output benchmark/mlx/engine_optimization_20261106/prefetch_replay --tokens 192
```

`--validation-only --tokens 64` 执行缓存逐位对照；加 `--kv-cache-dtype fp8` 验证
packed KV。基准代码从固定 Git commit 提取，候选使用当前 checkout。
完整设备测试在 `RESULTS.md` 的三个模块基础上，再加
`test.registered.unit.hardware_backend.mlx.test_dflash_serving`。

未验证大于两个并发请求、长上下文并发吞吐和独立质量评测。目标验证的多请求路径
仍有 eager 构图成本；没有完整的 GPU 接受环或固定地址执行器。继续提高单请求
吞吐仍需要改进实际 MoE/投影执行，不能从并发数字推导单请求已达到 90。
