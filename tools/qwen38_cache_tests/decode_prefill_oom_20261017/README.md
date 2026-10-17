# Decode 后长 prefill OOM 修复（2026-10-17）

原日志在持续 decode、两次 adaptive migration 后进入新的 8192-token prefill，
于 MTP draft extend 的 `fp4_mtp_moe` 最终输出处申请 40 MiB 失败。
`intermediate` 和 FP32 `projected` 原来按整个 tokens×topk 分配，8192 tokens、
topk=10、hidden=2560、width=640 的测试峰值为 1020 MiB，足以挤掉后续输出空间。

修复保留 A16、FP32 sum 和 BF16 rounding，将 MTP MoE 内部 scratch 限制为
128-token tile，小 decode/verify 走原路径。GPU 全尺寸测试峰值 55.9375 MiB，
减少 964.0625 MiB，输出与单个 8192-token tile 的原算法逐字节一致；129-token
尾块和更换输入后的 CUDA Graph 回放也一致。独立 dense 参考的既有 FP4 数值
检查通过（1/2/3/4/17 tokens）；W4A16 独立参考和 graph 检查也通过。

另将 cached W4A16 两个 projection 内核的 cache capacity E 改成不做 specialization
的运行期参数。容量 2/3/4/8 的数值回归通过，始终只有两个 split-K 内核变体，
避免在 adaptive migration 后按容量加载新内核。

真实服务已重建。第一次回归在 23410-token 上下文中持续生成 10240 tokens，
生成期间排入 46809-token 请求；新请求以 5×8192+5849 分块完成，输出 64 tokens。
第二次使用不同前缀，持续生成 2048 tokens，再完成 46814-token prefill 和 64-token
输出，两轮均 PASS、cached_tokens=0、无请求回撤。整个服务日志确认经过两次
adaptive migration、六次借用 arena 恢复（2785 个专家），无 OOM、无运行期
split-K device-load 告警。最终 /health=200、容器重启次数 0。

KV 容量仍为 425088、外部 chunk=8192、static fraction=0.90、MTP=2、上下文
220000、cache slots=5340，没有依靠缩小这些参数绕过问题。
本机合成回归覆盖本次已知触发路径，不代表所有输入都不可能耗尽显存。

关键证据：`mtp-memory-tests.log`、`capacity-tests.log`、`fp4-numerics.log`、
`service-regression.json`、`service-repeat.json`、`final-health.json`。
复现脚本是 `../repro_decode_then_prefill.py`；GPU 回归是
`../test_gpu_mtp_prefill_memory.py`。完整原始故障/服务日志留在工作区，未纳入提交。

分发包同步修复：memory-savings 的 05 patch 包含 MTP 分块和回归；W4A16 包
包含动态容量内核及编译脚本调整，两包通过组合 apply/reverse 检查。
