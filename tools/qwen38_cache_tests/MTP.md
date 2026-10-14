# 同卡 MTP eager 实验

默认关闭，SM120、纯文本、单请求、threaded 固定专家缓存。draft 只读 mtp.*，专家保留 block-FP8，其他参数 BF16，完整共享 target embedding/head。

在原 target 命令增加：
```bash
--speculative-algorithm NEXTN --speculative-draft-load-format qwen38_mtp \
--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
--cuda-graph-backend-prefill=disabled --cuda-graph-backend-decode=disabled
```
steps 1/2/3 对应 verify 宽度 2/3/4。先保持固定缓存比较不同宽度；graph 和 adaptive MTP 要应用对应兼容 patch。draft 默认 BF16 KV，GDN BF16，权重不整层转 BF16。MTP 原始独立权重约 2.513 GiB，KV/状态/工作区/graph/后处理峰值另算。

```bash
python3 -m unittest tools.qwen38_cache_tests.test_cpu_mtp
python3 -m unittest tools.qwen38_cache_tests.test_gpu_mtp_fp8
```
GPU parity、真实模型状态提交、显存与速度尚待 5090 验收；不承诺提升。
