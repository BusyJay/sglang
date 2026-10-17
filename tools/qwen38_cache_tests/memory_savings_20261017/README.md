基于 4ecfcb7e91 的两项显存优化，保留此 HEAD 的 PLE cache 实现。

* `SGLANG_QWEN38_BORROW_PREFILL_ARENA=1` 默认开启：threaded adaptive pool 的尾部 `num_experts` 行作为 arena，不减少 decode cache 的总 slot 数。连续 arena 调用共享一次租借；首次非 arena 调用、adaptive migration 前恢复被覆盖的原专家。被借用层禁用 prefill cache scatter 和 warm/seed，避免重叠写；其他层保留原行为。恢复包含全部 tensor 和 alpha，不修改原 lookup/LFU/SLRU 元数据。pool 不足、非 adaptive 或其他 runtime 继续使用独立 arena。
* `SGLANG_QWEN38_HOST_TOKEN_EMBEDDING=1` 默认开启：qwen38 cache loader、TP=1/attention TP=1、untied BF16 checkpoint。构造时 meta 参数，加载时直接映射原 safetensors tensor，CUDA register 后 GPU gather；不增加 cache 目录，不复制 token 权重到 GPU。输出和 LM head 在 GPU，qwen38_mtp 共享完整 embedding 模块。其他 TP/tied head 保留原路径。

预期当前模型独立 arena 显存减少约 1350 MiB，token embedding 减少约 1212.5 MiB，合计约 2562.5 MiB。KV pool 可能自动吃掉释放空间，应比较启动 allocator 探针、`runtime.memory_budget()`，不要仅看 nvidia-smi。专家借用计数见 `borrowed_prefill_arena`、`borrowed_restore_runs`、`borrowed_restore_experts`。

mmap 本身不保证 pin 快：5090 上 CUDA register 可触发 COW、锁页，消耗 host RAM；GPU 每次 gather 经 PCIe。连续 prefill 之后恢复最多 num_experts 个旧缓存专家，可能影响首个 decode/TTFT；interleaved decode 或混合容量触发额外恢复。当前策略优先保持原专家命中行为。此次不修改已有 PLE/expert pin 实现。

包中的 run.py 是复用的 manifest Docker runner，支持 --image、重复 --mount、-v、--env，不校验 checksum，不指定新的 cache 目录。默认实验写临时 safetensors，用满尺寸 248320×2560 BF16 embedding，比较 byte parity、CUDA graph replay、GPU 分配和 1/3/128/4096 token 延迟。CPU/GPU expert tests 包括跨 stream prefill/decode、adaptive migration；没有 GPU 时 assessment 输出 GPU_REQUIRED，退出 2。

```bash
python3 /path/to/bundle/run.py all --bundle /path/to/bundle \
  --repo /path/to/sglang --image YOUR_IMAGE --gpus device=0 \
  --mount 'type=bind,src=/your/models,dst=/your/models,readonly' \
  -v '/your/existing/cache:/your/existing/cache' \
  --env TRITON_CACHE_DIR=/your/existing/cache \
  --output /path/to/results
```

assessment PASS 只表示微实验门槛通过。上线前用原模型、原启动脚本、相同 KV/static-memory 设置跑两组（两个开关都设 0 vs 都设 1），记录 startup register 秒数、allocator bytes、TTFT、TPOT、吞吐、MTP 接受率。相同请求重复至少 3 轮，包括首次长 prefill、连续 chunk、decode 后新 prefill，以及 prefill→MTP verify。要求质量一致，VRAM 权重分配减少约上述数值，吞吐不下降超过 5%，TTFT 无不可接受回退。达标后原启动命令无需加开启参数；更新源码挂载或镜像再重启原服务。脚本不替你部署/重启。

临时关闭分别设置上述开关为 0 并重启。代码回滚用 `git revert <对应commit>`，允许工作区有不相交的修改；有重叠先保存那部分修改。未 commit 的外部 patch 用 `run.py rollback --bundle ... --repo ...`，forward/reverse apply 检查保留无关修改。不要用 reset --hard 清理工作区。

2026-10-17 修订：修复被重复方法覆盖的 arena 显存统计，补齐独立/借用 arena 的 GPU 测试预期；66 个 CPU、38 个 GPU cache 测试通过。满尺寸 embedding 输出和 CUDA Graph 校验通过，但 PCIe gather 延迟未达到原定 prefill 5% 门槛，assessment 仍为 FAIL，不能据此认定可上线。详见 VALIDATION.md 和 results/fixed。

2026-10-17 整模型复测：两项优化均已部署。embedding 节省 1212.5 MiB，arena 借用 1350 MiB，独立 arena 为 0；token 容量 235456→425088。原服务挂载旧快照，现已改为始终挂载 /home/user/workspace/sglang。两次主模型加载为 59.56/124.75 秒（原 196.59 秒）；缓存冷热影响明显。完整记录见 tools/qwen38_cache_tests/startup_memory_20261017/README.md。

2026-10-17 OOM 修订：增加 05-bounded-mtp-prefill.patch。MTP draft extend 的 MoE 临时量按 128 tokens 分块，8192-token GPU 峰值从 1020 MiB 降到 55.94 MiB，输出逐字节一致；外部 chunk=8192 和 KV 容量不变。GPU manifest 增加长 MTP 分块/graph 回归。详情见 tools/qwen38_cache_tests/decode_prefill_oom_20261017。
