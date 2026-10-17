# 保持 MTP 开启的 verify 性能修复

用户要求开启 MTP；关闭 MTP 仅是此前的诊断对照，不作为修复方案。
当前共享入口默认值已恢复 `MTP_MODE=2`。

## 改动

- 3 token × top-k 10 的 verify 共 30 条路由，接入现有最多 32 条路由的
  GPU fused planner；以 view 展平给 planner，保留原始 token/route 输出形状。
- 原生 histogram 计算跨 token 的专家并集，保护全部活跃专家，避免错误驱逐，
  每个 miss 专家只加载一次。fill grid 上限改为全部路由数，避免只填前 10 个 miss。
- verify 使用 device fill，与 shared expert 跨 stream 重叠；不再逐专家 CPU staging。
- 采样 row 扩为 count + 32 IDs，防止多 token 并集超过原 top-k 缓冲造成越界。
- 当前优化限定路由总数不超过 32 且能放进该层 slots；更宽输入仍使用原通用路径。
  prefill 精度及 verify A16 精度保持原有设置。

冻结源码：`models/sources/fp4-local-verify-mailbox-bba4aaf182a3`。
父版本 `fp4-local-pin-mtp-edf6633e5c76` 保留，镜像、权重和 pack 不变。
两个 FP4 preset 均挂载新快照；备份位于 `model-backups/20261017-verify-mailbox`。

## 正确性证据

- GPU：30-route planner 对比通用 histogram/plan/remap，覆盖 LFU、SLRU、
  冷 miss、命中、驱逐、重复专家、-1 padding 和采样边界；通过。
- GPU：5 项 shared fill 测试通过，包含 verify 重复路由、采样和 breakable replay。
- GPU：原有 mailbox byte 检查和 device sampling 回归通过。
- CPU：cache/shared/sampling/graph/analysis 共 34 项，33 通过、1 环境相关跳过。
- W4A16 独立参考：真实 2560/640/top-k 10 形状、32 个合成专家，
  3-token verify + graph replay 最大绝对误差 0.000244140625；
  单 token prefill A4 与原 resident A4 路径一致。详见 `verify-mailbox-numerical.json`。

完整旧数值脚本一次连续运行多个 graph 用例时遇到 CUDA allocator capture
invalidated；单独运行新增 `--verify-mailbox-only` 用例通过。此项保留在验证记录中，
不能把整套旧脚本声称为全通过。真实服务的 graph 初始化与请求另行验收。

## 真实服务结果

服务保持 MTP=2、3 draft tokens、verify A16、pinned PLE、5340 expert slots。
内部 server_info 将 NEXTN 规范化为 EAGLE，`speculative_num_steps=2`，
`speculative_num_draft_tokens=3`，GDN state 为 BF16。

| 同样的短 prompt，每条生成 192 token | 旧 MTP=2 | MTP off 对照 | 修复后 MTP=2 |
| --- | ---: | ---: | ---: |
| CPU cache 英文解释 token/s | 40.25 | 55.57 | 60.32 |
| 中文索引解释 token/s | 38.93 | 52.69 | 60.09 |
| Python LRU 实现 token/s | 39.39 | 50.80 | 59.87 |
| 合计 token / 合计时间 | **39.51** | **52.95** | **60.10** |

修复后相对旧 MTP 提高 52.1%，相对 off 对照提高 13.5%。均为不开 profiler
的独立请求。prompt 均无前缀 token 命中。两次进程的 cache/adaptive 历史不同，
旧 MTP 的观察日志启用、修复版关闭；不能将其视为严格隔离 transport 的性能实验，
也不能把短请求收益推广为所有长上下文/低接受率负载的保证。

profile（修复后 19 步）确认 912 次 mailbox fill、912 次 fused planner，即每步
48 层均走新路径；verify 窗口内没有专家 H2D memcpy 和 packed scatter。
GPU verify 窗口均值约 39.59 ms，旧版约 80.52 ms，含 profiler 扰动，不能当作
正常请求速度。target `_cache_project` 每步约 11.33 ms，旧版约 11.25 ms，
计算精度/计算工作量保持，收益来自 transport/planner/overlap。
新的 mailbox fill 仍从 mapped host memory 通过 PCIe 取权重，并未消除 PCIe 流量。

summary 工具现按 External id 关联 GPU annotation，统计异步提交后的 GPU 工作；
CPU step span 与 GPU step span 分开列出，避免 CPU 返回时截掉 GPU 尾部工作。
4 项汇总测试通过，包含异步 kernel 超出 CPU span 的回归。

功能 canary：17+25 回答 42，远端 ViT 红图回答红色；10093-token 图文输入跨
8192-token prefill chunk，成功识别红色并输出 1 至 20。1024-token 连续 decode
15.155 秒完成（约 67.6 token/s），接受长度 2.69。

详见 `bench-mtp2-fixed.json`、`fixed-comparison.json`、`fixed-profile-summary.json`、
`fixed-canaries.json`、`fixed-long-decode.json`、`server-fixed.json`。

最后的服务累计快照：device gate / fused planner 均 37823 次，shared-fill
overlap 37775 次，采样丢弃 0；最终算术 canary 再次返回 42，health 200。
512-token 追加生成 8.622 秒完成，MTP 接受长度 2.32。
