# 2026-10-17 启动、显存和 W4A16 验证

当前语言服务已重建并健康，实际源码挂载为 `/home/user/workspace/sglang:/workspace/sglang:ro`。
`fp4-local`、`fp4-local-a4`、`nvfp4` 的 Compose 配置和 catalog 已统一该目录；
启动脚本在源码缺失时报错，不再静默回退到镜像内代码。ViT 配置也已更新，
本次按约定没有重建 ViT；其正在运行的进程仍使用原来的代码。

此前运行的是 `fp4-local-gdn-lifetime-788a53cee45c` 快照，缺少 token embedding、
借用 arena、PLE packed cache 和 split-K 实现。`down/up` 只会重新挂载配置中的
同一路径，脚本中 `SPLIT_K=1` 对没有对应内核的旧源码无效。

| 指标 | 原服务 | 新服务，split=0 | 新服务，split=1（当前） |
|---|---:|---:|---:|
| 主模型权重加载秒数 | 196.59 | 59.56 | 124.75 |
| target+draft 权重加载秒数 | 198.41 | 62.34 | 126.07 |
| tokenizer_e2e 秒数 | 220.38 | 90.23 | 151.54 |
| max_total_num_tokens | 235456 | 425088 | 425088 |
| 相同三条短请求 decode token/s | 未测 | 64.76 | 75.79 |

两项显存优化都已在实际模型生效：token embedding 为 1212.5 MiB；
独立 prefill arena 分配为 0，借用 decode pool 尾部 512 个 slot，大小为
1350.00 MiB；decode cache 总量仍为 14080.12 MiB。合计节省 2562.5 MiB。
KV 自动扩容，所以启动后的 available_gpu_mem 仍是 2.09 GB；容量增加 80.5%。
5340 cache slots、mem_fraction_static=0.90、chunk=8192、MTP=2、上下文 220000 保持原配置。

启动优化：实际 checkpoint 的 26.82 GiB PLE 最终行布局缓存已预生成并验证 shard
边界；启动确认 packed cache hit。剩余瓶颈是 63.28 GiB 专家源重新加载到匿名
THP 内存以及主机回收压力，实测 source preparation 为 55.91 / 123.88 秒。
保留文件页缓存只允许 OS 缓存，不保证匿名源副本和 PLE/HiCache 分配后还能
保留整份文件。两次启动的缓存冷热不同，不能把最快的 59.56 秒当成固定保证，
也不能把 split=0/1 的启动耗时差异归因于 split-K。增加了常规启动阶段和显存日志，
无需打开运行期 profiler 即可定位。完整日志和 summary.json 与本文同目录。

W4A16：原 patch 已应用到开发仓库。两次空闲 SM120 微基准均通过全部数值和
CUDA Graph 校验；split=1 最差场景加速分别为 1.566、1.564 倍。服务三条相同
prompt、每条 192 output tokens 的持续 decode 对照提高 17.03%，service-compare
为 SERVICE_SPEED_PASS。它是小样本合成请求结果，不是所有生产负载的收益保证。
当前启动脚本 split 默认 1，允许环境变量覆盖为 0。

额外检查：12012-token 新 prompt（cached_tokens=0）跨 8192-token chunk，输出 32
个 tokens，耗时约 3.21 秒，无错误/回撤。arena 跨 stream/恢复和 adaptive migration
GPU 回归通过；相关 CPU、部署配置单元测试通过。最终健康检查、挂载和重启次数
见 final-service.json。

直接注册共享文件映射的实验在小尺寸测试通过，但完整 63 GiB 映射注册被 CUDA
拒绝，已经撤回，当前仍使用 anonymous_thp；失败证据在 startup-baseline.log，
不作为最终可用实现。没有修改 checkpoint 或 expert pack 内容。

后续修订：持续 decode 后再次长 prefill 暴露 MTP 未分块 scratch 峰值。修复和更完整回归见 ../decode_prefill_oom_20261017/README.md；425088 KV 容量保持不变。
