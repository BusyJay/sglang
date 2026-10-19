# 动态路径选择（auto）复验

基准仍是 `d3eb24c40031999b3e62e4e9351c5b6118dfaa24`、同一张 RTX 5090 D、同一镜像 `swallow-sglang:nightly-dev-20261018-b234b5c3` 与同一 FP4-Local checkpoint。本轮只增加选择策略：新增 `cache_overlap_policy`（`auto` / `always`，默认 `auto`），当时三个计算开关仍默认关闭。

本轮结论落地后做了默认值调整：`prefill_direct_cache` 在 threaded + `prefill_reuse_cache` 下改为默认开启（下面的 prefill 档位表就是它开启并由 auto 选路的结果，且 170K 长请求依赖同一开关上的 workspace 上界）；`decode_early_cache`、`preserve_cache_on_resize` 继续默认关闭。当前默认值汇总见 [上层报告](../REPORT.md#当前默认值)。

## 方法

- 每次计时前恢复 device policy 状态、cache 权重和 arena 驻留集合，保证计时对象是真实缺失，而不是被前一次调用预热成的命中。脚本用 `host_missing` 增量断言注入的缺失数（`sweep_decode_auto.py`、`sweep_prefill_auto.py`、`sweep_prefill_hot.py`、`sweep_prefill_1024.py`）。
- 单 token decode 每档 30 次热重复调用，prefill 每档 12 次，去掉前 3 次后端特化预热，取 CUDA event 中位数（包含主机提交空隙，不是纯 GPU kernel 时间）。三种模式 `off` / `always` / `auto` 的输出互查，`rtol=atol=0.02` 未放宽。
- kernel 与拷贝数量来自 chrome trace，只统计 `cat=kernel`，汇总在 `profile-summary.json`，原始 trace 在 `profiles/`。
- 计时是单层合成权重与固定路由（hidden=2560、intermediate=640、32 experts、16 cache slots、top-k=10），不是并发吞吐。

## Decode：没有可靠盈利区间，auto 绕过 early 分支

B12X 后端，每档 30 次中位数（ms）：

| 每层缺失专家 | 关闭 | `always` | `auto` |
| --- | ---: | ---: | ---: |
| 0 | 0.277 | 0.492 | 0.283 |
| 1 | 0.282 | 0.478 | 0.288 |
| 3 | 0.293 | 0.486 | 0.297 |
| 5 | 0.399 | 0.495 | 0.399 |
| 7 | 0.535 | 0.550 | 0.534 |
| 10 | 0.714 | 0.727 | 0.713 |

`auto` 与关闭一致，`always` 在低缺失区间慢 60%~70%，即使 7 个以上缺失仍不优于关闭。Triton 后端（`decode-sweep-triton-adaptive.json`）在缺失 >=3 时最好也只有 -2%~0%，属于噪声量级；B12X 全程更慢，只有缺失=10 出现 -5% 的单点（`decode-sweep.json`）。这些都不足以支撑一个阈值，因此 auto 模式下 decode 不进入提前计算分支，包括跳过空 kernel、事件和输出合并；`always` 保留给强制对照和后续调优。

trace 印证：`decode-off-0` 与 `decode-auto-0` 均为 7 个 kernel（36.2 / 32.4 us），`decode-always-0` 为 12 个 kernel（41.0 us）。不是把权重置零后再算两遍。

## Prefill：三个可复现区间

`auto` 与关闭（`off`）、强制开启（`always`）的每档中位数（ms），单层 2048/128/3 token：

| tokens | 缺失 | 关闭 | `always` | `auto` | auto 选择 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 0 | 0.359 | 0.364 | 0.352 | cache 单次 |
| 3 | 1 | 0.449 | 0.632 | 0.452 | arena 单次 |
| 3 | 5 | 0.568 | 0.686 | 0.580 | arena 单次 |
| 128 | 0 | 0.321 | 0.287 | 0.291 | cache 单次 |
| 128 | 5 | 0.535 | 0.498 | 0.555 | arena 单次 |
| 2048 | 0 | 0.669 | 0.596 | 0.595 | cache 单次 |
| 2048 | 2 | 0.709 | 0.747 | 0.751 | arena 单次 |
| 2048 | 3 | 0.748 | 0.727 | 0.650 | split |
| 2048 | 5 | 0.836 | 0.677 | 0.667 | split |
| 2048 | 8 | 1.064 | 0.969 | 0.969 | split |

1024-token 独立复跑（`prefill-auto-1024.json`）同样在缺失 3~9 选择 split，并快于关闭（缺失 7：0.738 vs 0.785；缺失 8：0.802 vs 0.857）。

三条规则对应三类收益来源：

1. 全部所需专家都在 cache 且没有缺失时直接读 cache，单次 dispatch。省掉 cache→arena 权重复制，kernel 数从 16 降到 11（`prefill-auto-3-0` vs `prefill-off-3-0`）。
2. arena 已经包含全部所需专家时短路回原有单次 arena 调用，不构建命中列表，也不提交第二次 dispatch。kernel 数与关闭完全相同（`prefill-hot-auto-3-5` 与 `prefill-hot-off-3-5` 均 10 个，`always` 为 21 个），热缓存计时 0.301/0.603 ms 对 0.310/0.603 ms。
3. 只在混合命中、批量足够大且确有主机搬运时拆成 cache + arena 两次计算。2048-token 5 个真实缺失的对照中 auto 0.667 ms、关闭 0.836 ms，约快 20%；三者的 kernel 累计时间同为 367~369 us，收益来自主机搬运与计算重叠，而不是 GPU 计算变少。

小批量混合命中（3/128 token、1~2 个缺失）拆成两次计算更慢：3 token / 1 缺失时 `always` 0.632 ms 对关闭 0.449 ms，`auto` 选择 arena 后回到 0.452 ms。arena 缺失已全部驻留时不构成搬运机会，同样退回 arena 单次（`test_qwen38_cache_overlap_policy.py` 覆盖）。

## 阈值与边界

`select_prefill_cache_path()` 只使用 CPU 侧已有 plan（路由计数、cache keys、arena 驻留集合），不增加逐 token 的 GPU 状态读回：

- 缺失专家已全部在 arena → arena 单次；
- 无缺失 → cache 单次（任意后端与几何，仍是一次 dispatch）；
- 混合命中 → 仅当 W4A4、几何为已标定的 (hidden=2560, intermediate=640, top-k=10)、总路由 >=10240、命中路由 >=1024、且至少 3 个专家真正需要主机搬运时 split；
- 未标定的后端或几何 → arena 单次。

阈值来自上面的档位对照，属于合成数据上的经验下界，不是普适结论。

## 与原有 OOM 防护共存

完整模型 canary（`run_model_auto.sh` / `model-auto/`）：FP4-Local checkpoint、MTP=2、B12X、breakable graph、6388 基础 slots、KV stages 80000/160000/220000、`mem_fraction_static=0.90`、外层 chunk=8192，三个计算开关全部显式开启且 `cache_overlap_policy=auto`，**未设置** `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`。1024 / 8192 / 74000 / 170000 token 请求全部完成（各 8 tokens，其中 8192 在 4 tokens 处命中停止词），算术探针返回 `42`，`report.json` 状态 `passed`。

同一份 `server.log` 的运行时计数确认动态选择真的生效而不是恒等于某一条路径：`auto_split_runs` 在多数层为 2（真实模型走过拆分），`auto_arena_runs` 出现在短上下文层，`decode_early_cache_invocations=0`（decode 未进入提前分支）。

该配置在 16:55:11~16:55:13 有 3 条 PyTorch allocator 的 `memory allocation failed with OOM` 重试告警（Triton kernel 在开始服务后才 device-load），请求仍完成；没有 policy、仅保留分块 MoE 的 `../optimization/model-chunked-moe/server.log` 同样有 3 条同类告警，因此不是本轮引入，也不能据此宣称长上下文峰值已经彻底解决。

## 限制

- 单层合成权重、单请求、12/30 次重复调用，事件计时包含主机提交空隙；不能换算成完整模型吞吐或生产 p99。
- auto 自身有 CPU 成本：2048-token / 缺失 1~2 档 auto 0.719 / 0.751 ms 慢于关闭 0.659 / 0.709 ms，12 次样本尚在波动范围内，需要更多轮才能确认，故未据此再调阈值。
- 1024-token 全命中档本次为 auto 0.477 vs 关闭 0.445，与 3/128/2048 token 同档的正向结果不一致，说明该分支收益不稳定；保留它只因为省掉一次权重复制且不增加 kernel 数。
- 本轮没有重跑 GPU 回归（GPU 被在线服务占用，5090 D 上另有 `homelab-llm` 在 31.9 GB / 99% 运行）。GPU 数值与性能证据来自同一工作树在 00:0x~00:5x 的运行（`final-regression.log`、`ab-b12x.json`、`resize-fixed.log`、`vmm-resize.log`、`profiles/`、`model-auto/`）；本轮在 pinned 镜像里重跑的是 CPU 回归 348 项、Qwen38 cache 相关注册单测 17 项（选路策略、路由拆分与搬迁时序、loader 配置边界），以及 ruff / isort / CI 注册审计。

## 复现

```bash
# 计时扫描（需要空闲 GPU；results 目录必须已存在且可写）
repo_root=$(git rev-parse --show-toplevel)
results="$repo_root/tools/qwen38_cache_tests/cache_overlap_20261019"
docker run --rm --pull never --network none --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$repo_root:/workspace/sglang:ro" -v "$results:/results" \
  -w /workspace/sglang -e PYTHONPATH=/workspace/sglang/python \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 swallow-sglang:nightly-dev-20261018-b234b5c3 \
  tools/qwen38_cache_tests/cache_overlap_20261019/adaptive/sweep_prefill_auto.py

# 完整模型 canary（三个开关 + auto，脚本内自带 checkpoint/manifest 路径）
bash "$repo_root/tools/qwen38_cache_tests/cache_overlap_20261019/adaptive/run_model_auto.sh"
```
