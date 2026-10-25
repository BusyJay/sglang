# C2 consumer tuning (2026-10-26)

Follow-up to `06cbc28918`. The original C2 uses a 16-warp reduction kernel both to compute the FP32 sigmoid gate and to apply the gate. Only the second launch is changed to 8 warps. The gate reduction, FP32 FMA/BF16 store, PDL ordering, per-call buffers and service flags are preserved.

## Screening

`layout-sweep.json` and `layout-sweep-extended.json` compare row and tiled consumers. Tiling reduces post-fill consumer time but increases the full serial chain, so it is not adopted. `pdl-sweep.json` shows disabling PDL also increases the serial chain, so PDL stays enabled. `early-load-sweep.json` shows moving the gate load earlier increases the serial chain; that source experiment was restored and is not adopted. These are screening sweeps, not paired acceptance results. In `early-load-sweep.json`, warps=0 labels the unchanged 16-warp reference kernel copied from the parent commit.

## Paired acceptance

`paired.json` records four ABBA cycles per shape and phase, 200 operations per captured graph, seven CUDA event samples per graph. Both arms include the same routed reset copy. The serial phase additionally computes the same gate into the same preallocated FP32 buffer. Measurements ran with the service stopped, on GPU0, using the same CUDA/PyTorch/FlashInfer image as production. Each M=1/2/3 case also checks 20 random inputs against the stock fused BF16 output with zero tolerance (60 cases).

| M | consumer old→new, µs | serial old→new, µs |
| --- | --- | --- |
| 1 | 1.6524→1.6362 | 2.3376→2.3278 |
| 2 | 1.6815→1.6798 | 2.3624→2.3498 |
| 3 | 1.7262→1.7042 | 2.3909→2.3803 |

The consumer change is small (0.1–1.3% including reset), and the serial chain changes by about 0.4–0.5%. It does not eliminate C2's extra launch or make the full C2 path faster than the original unsplit fused operator. No equivalent whole-service speedup is claimed.

Reproduce with the inference service stopped:

```bash
python test/manual/kernels/qwen38_shared_gate_tuning.py --output /new-private-dir/paired.json
python test/manual/kernels/qwen38_shared_gate_tuning.py --mode sweep --output /new-private-dir/sweep.json
```

The serving controller compares 16/8/8/16 warps using the unchanged private Hermes payload, restarting each time and fixing server seed=42. It preserves natural EOS (5895 is the maximum, not a forced completion length), records output length and acceptance statistics, and restores the final source and original launcher in `finally`. Complete private request/response and logs remain outside Git. The initial attempt incorrectly asserted exactly 5895 output tokens; it was abandoned before producing a comparison and is excluded from estimates. The repeated controller saves results before checking validity.

## 原始 Hermes 请求服务对照

| 顺序 | consumer warps | 输出 tokens | TTFT s | 稳定 token/s | ms/VERIFY | 接受长度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 5346 | 6.854 | 97.77 | 18.292 | 1.787 |
| 1 | 8 | 5895 | 6.865 | 95.03 | 18.024 | 1.711 |
| 2 | 8 | 5895 | 7.757 | 93.79 | 18.017 | 1.687 |
| 3 | 16 | 5895 | 6.865 | 95.36 | 17.690 | 1.685 |

四轮均无 retraction、cached_tokens=0、无推理错误；保持原请求采样参数与自然 EOS，第一轮正常提前结束。16/8-warp 两组平均每 VERIFY 为 17.991 / 18.020 ms，差约 0.16%，属于此前重复基线的波动范围，服务层面判持平。稳定吞吐平均为 96.57 / 94.41 token/s，后者低约 2.23%；接受长度平均为 1.7363 / 1.6987，也下降约 2.17%。输出和接受率不同，不能把 token/s 差异单独归因于线程配置。第二轮 8-warp 的 TTFT 发生单次增高，原始值完整保留；两轮不能建立该项影响 prefill 的结论。

最终保留 **8-warp consumer**：固定输入下消费者 M1/M3 和串行链路小幅改善，M2 消费者接近持平；完整服务没有确认回退。未启用任何筛选中串行链路回退的候选，C2 开关仍开启。不会把这次微小调参描述成端到端加速或消除 C2 的额外工作。

## 正确性和部署

- `implementation-proof.json` 用 AST 对比确认生产实现仅改变 consumer 的 `num_warps=16→8`；gate reduction、计算表达式与 PDL 均未改变。
- `handoff.txt`：现有 5 项 handoff / breakable graph 集成测试通过。
- `memcheck.txt`：同样 5 项在 compute-sanitizer memcheck 下通过，0 memory errors。沿用前轮对 CUDA Python API 查询报告的抑制，未关闭显存访问检测。
- `static.json`：format、isort、定向 lint、CI 注册与 package 检查通过。
- 当前改动只针对已启用 C2 的 consumer，A/B/C1/C2/U5 仍全开，MTP、KV 精度、context 与 chunk 不变。机器为同一 RTX 5090 D / SM120；这些结论未扩展到其它硬件。

最终恢复原部署启动参数后的原始 Hermes 回放：25,560 输入、5,895 输出，TTFT **6.835 s**，稳定生成 **96.27 token/s**，最大流间隔 **0.247 s**，cached_tokens=0、retractions=0、无推理错误、/ready=200。完整私有正文和日志留在 `/home/user/workspace/prod/homelab-llm/models/profiles/c2-tuning-20261026/production-final`。
