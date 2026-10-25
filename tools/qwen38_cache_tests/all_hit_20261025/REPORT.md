# 全命中路由与 alpha 准备融合（2026-10-25）

已在主工作区完成优化与完整模型 ABBA，正式后端已恢复并确认 ready=200。生产保持原配置，`decode_early_cache` 仍未开启；本次改动在启用 native 命中早算时使用。

## 结果

| 范围 | 基线 | 优化后 | 收益 |
|---|---:|---:|---:|
| 完整模型短请求（每轮三次中位数，再取两轮均值） | 113.899 token/s | 115.080 token/s | +1.04% |
| 完整模型自然文本（两轮均值） | 115.008 token/s | 116.308 token/s | +1.13% |
| M=1 全命中 mailbox，无共享专家 | 41.199 µs | 37.187 µs | 吞吐 +10.79% |
| M=1 全命中 mailbox，有共享专家 | 51.664 µs | 47.468 µs | 吞吐 +8.84% |
| M=3 全命中 mailbox，无共享专家 | 59.894 µs | 57.551 µs | 吞吐 +4.07% |
| M=3 全命中 mailbox，有共享专家 | 73.970 µs | 69.698 µs | 吞吐 +6.13% |

吞吐改善按基线耗时/优化耗时计算；这里的微测是完整 mailbox 回调，不是单个 FC1 kernel。全模型收益小于微测，不应将微测百分比直接用于整体服务。

## 实现

1. GPU-only native FC1 路径让现有 planner 在修改 lookup 前直接保存命中 slots，省去独立 `cache_partition_routes` kernel。保留原 full-MoE early planner 的全命中/全 miss 处理语义，native 变体则保留全命中的 early 路由；没有读回 CPU 命中计数。
2. resident alpha 模式将 miss alpha scatter、互补路由生成和全部 routed globals 的刷新合并进一个 CTA。该模式的 fill layout 排除了 alpha 行，因此它可以与权重搬运并行；CTA 内 barrier 保证 scatter 对后续读取可见。非 resident 模式保留原来只读 protected hit、fill 后再刷新的路径。
3. 省去 finish graph 的重复 alpha refresh。实测全命中 GPU kernel 数从 9 降到 6：partition、独立 alpha scatter、后置 alpha refresh 消失。算术顺序、FC2 原生中间格式和路由累加顺序保持不变。
4. stage 缓存 key 包含 resident plan/table 的地址，避免绑定切换时复用旧指针。首次预热用私有空 plan，避免旧 plan 写入新 cache；resident kernel 在 capture 前完成首次设备加载。
5. 私有 graph 使用 thread-local capture mode，保留当前线程的 capture 检查，同时允许后台线程在无关 stream 上处理采样。原先 global capture mode 在真实 serving 冷绑定时失败，改后两轮优化组冷启动、动态容量切换及测试通过。后台线程具体哪一次 CUDA 调用触发 global 检查仍属于推断，没有 API 级完整重放。

全命中稳态没有新增 `.cpu()`、`.item()`、CPU ack 或 CUDA host synchronize。现有 GPU stream/event 依赖和 mailbox 背压保留。空 fill 及空 late-FC1 仍存在；本次通过融合降低总开销，没有为了强行删除它们而保留更慢的条件图方案。

## 验证

- 真实权重 mailbox 连续回放 60 次（M=1/2/3），包含 30 次全命中，所有 FP32 输出逐位一致。`mailbox-fused-real.json` 和 `mailbox-fused-real.log`。
- 最终 9 项 GPU 测试通过：路由/策略状态、两类 early planner、resident scatter、非 resident 保护、重复路由、CPU placeholder、新建图无整卡同步、旧 plan 不得在新绑定预热时写 cache。native 数值矩阵同时覆盖 resident=False/True、M=1/2/3、miss=0/1/3/10。
- GPU-free namespace/registry 测试 70 通过；Ruff、注册测试检查、diff whitespace 检查通过。
- 完整模型按 baseline/fused/fused/baseline 顺序串行运行，两组均启用 `decode_early_cache=true`，其它模型配置相同。三条短请求及自然文本各生成 512 tokens；四轮文本 SHA256 全部一致，MTP 接受长度全部一致。
- 四轮实际 expert budgets 均为 `[7161, 6798, 6543]`，memory fraction=0.92，220K context，Mamba=10。未通过减少槽位/上下文换取吞吐。
- 优化组额外完成 219K 输入、之后 53K 前缀恢复及追加验证。
- 完整模型对照固定 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`，与当前服务使用的 Python TreeCore 一致。预检遇到测试容器 Rust 缓存缺失/路径自动选择差异，未混入有效 ABBA 数据。

## 放弃的条件图方案

设备端条件节点可以跳过空 fill 和 late-FC1，数值验证也通过，但实测更慢。即使把两个 handle setter 合进路由 kernel，M=1 全命中仍约 38.9µs，对照约 33.0µs，miss 也有退化，因此已从运行代码撤回。实验源码/数据留在本目录及 `conditional-experiment-source`，不属于最终补丁。

## 产物与限制

- `all-hit-fusion.patch`：本轮修改，保留此前早算和 OOM 修复。
- `source-manifest.json`：最终实现/测试源码校验和。
- `serving-summary.json`、各 `r*-*/cases.json` / `server.log` / `environment.json`：ABBA 数据、配置、文本与接受长度。
- `baseline-all-hit.trace.json` / `fused-all-hit.trace.json`：9→6 kernel 的 GPU 执行证据。
- `final-gpu-tests.log`、`namespace-tests.log`：最终测试日志。

完整模型改善约 1%，是当前机器和冻结工作负载下的结果；不能据此保证所有请求的收益。219K 与前缀切换通过不能替代整夜运行稳定性验证。
