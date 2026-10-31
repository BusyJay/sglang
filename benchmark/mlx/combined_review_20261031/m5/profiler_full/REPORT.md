# GLM MLX performance evidence

Host spans measure Python build/enqueue/wait wall time, not GPU kernels. Sync attribution adds explicit drains and changes graph lifetime/overlap. Inclusive rows overlap; self rows subtract nested recorded spans.

Workload: `verify`; synthetic: `False`; steps: `1`; checkpoint revision: `fa04cbc3622b16aa7475404060b05e506d81bd4e`.

## Alternating measured trials

| Mode | First ms | Median steady ms | Relative to baseline | Token hash equal |
|---|---:|---:|---:|---|
| baseline | 60.550 | 45.127 | 1.000x | True |
| host | 46.624 | 46.978 | 1.041x | True |
| sync | 163.209 | 184.572 | 4.090x | True |

## host recorded host spans

| Operation | Module | Calls | Inclusive ms | Self ms |
|---|---|---:|---:|---:|
| mx.async_eval |  | 24 | 48.922 | 48.922 |
| mx.eval |  | 2 | 4.757 | 4.757 |
| Model.verify_mtp_rows |  | 2 | 87.733 | 4.285 |
| mlx_hc_collapse_norm |  | 360 | 3.565 | 3.565 |
| mlx_affine_rows |  | 438 | 3.425 | 3.425 |
| mlx_kda_decode |  | 204 | 2.724 | 2.724 |
| mlx_glm_router |  | 84 | 0.825 | 0.825 |
| forced_verify |  | 2 | 93.213 | 0.723 |
| MLP.__call__ | model.layers.31.mlp.shared_experts | 2 | 0.611 | 0.572 |
| mlx_glm_down_combine |  | 84 | 0.528 | 0.528 |
| DecoderLayer.verify_ffn_rows | model.layers.1 | 2 | 0.762 | 0.389 |
| mlx_dense_rows |  | 70 | 0.372 | 0.372 |
| mx.quantized_matmul |  | 264 | 0.316 | 0.316 |
| MLA.__call__ | model.layers.19.self_attn | 6 | 0.353 | 0.297 |
| DecoderLayer.verify_ffn_rows | model.layers.8 | 2 | 0.871 | 0.251 |
| MLA.__call__ | model.layers.7.self_attn | 6 | 0.284 | 0.237 |
| MLA.__call__ | model.layers.39.self_attn | 6 | 0.281 | 0.236 |
| MLA.__call__ | model.layers.43.self_attn | 6 | 0.283 | 0.235 |
| DecoderLayer.verify_ffn_rows | model.layers.20 | 2 | 0.815 | 0.233 |
| MLA.__call__ | model.layers.3.self_attn | 6 | 0.289 | 0.232 |
| MLA.__call__ | model.layers.11.self_attn | 6 | 0.288 | 0.231 |
| MLA.__call__ | model.layers.35.self_attn | 6 | 0.276 | 0.227 |
| MLA.__call__ | model.layers.27.self_attn | 6 | 0.280 | 0.226 |
| DecoderLayer.verify_ffn_rows | model.layers.16 | 2 | 1.410 | 0.221 |
| MLA.__call__ | model.layers.23.self_attn | 6 | 0.277 | 0.219 |
| MLA.__call__ | model.layers.15.self_attn | 6 | 0.263 | 0.217 |
| DecoderLayer.verify_ffn_rows | model.layers.21 | 2 | 0.722 | 0.215 |
| DecoderLayer.verify_ffn_rows | model.layers.18 | 2 | 0.740 | 0.215 |
| DecoderLayer.verify_ffn_rows | model.layers.41 | 2 | 0.749 | 0.214 |
| MLA.__call__ | model.layers.31.self_attn | 6 | 0.262 | 0.213 |
| DecoderLayer.verify_ffn_rows | model.layers.38 | 2 | 0.688 | 0.210 |
| DecoderLayer.verify_ffn_rows | model.layers.12 | 2 | 0.767 | 0.209 |
| DecoderLayer.verify_ffn_rows | model.layers.40 | 2 | 0.806 | 0.205 |
| DecoderLayer.verify_ffn_rows | model.layers.44 | 2 | 0.733 | 0.204 |
| DecoderLayer.verify_ffn_rows | model.layers.4 | 2 | 0.743 | 0.203 |
| DecoderLayer.verify_ffn_rows | model.layers.36 | 2 | 0.764 | 0.202 |
| DecoderLayer.verify_ffn_rows | model.layers.32 | 2 | 0.746 | 0.201 |
| DecoderLayer.verify_ffn_rows | model.layers.14 | 2 | 0.738 | 0.201 |
| DecoderLayer.verify_ffn_rows | model.layers.28 | 2 | 0.746 | 0.200 |
| DecoderLayer.verify_ffn_rows | model.layers.26 | 2 | 0.763 | 0.198 |

## sync recorded host spans

| Operation | Module | Calls | Inclusive ms | Self ms |
|---|---|---:|---:|---:|
| mx.eval |  | 1210 | 237.042 | 237.042 |
| MoE.__call__.drain_after |  | 84 | 83.807 | 42.717 |
| mx.synchronize |  | 1208 | 28.329 | 28.329 |
| Model.verify_mtp_rows |  | 2 | 367.397 | 5.166 |
| mlx_hc_collapse_norm |  | 360 | 3.622 | 3.622 |
| mlx_kda_decode |  | 204 | 3.061 | 3.061 |
| mlx_affine_rows |  | 438 | 2.934 | 2.934 |
| KDA.__call__.drain_before |  | 204 | 12.294 | 2.508 |
| KDA.__call__.drain_after |  | 204 | 51.221 | 2.297 |
| DecoderLayer.verify_ffn_rows.drain_after |  | 90 | 20.605 | 1.516 |
| mlx_glm_router |  | 84 | 0.963 | 0.963 |
| forced_verify |  | 2 | 368.636 | 0.935 |
| MLA.__call__.drain_after |  | 66 | 30.096 | 0.846 |
| DecoderLayer.verify_ffn_rows.drain_before |  | 90 | 3.069 | 0.760 |
| MLP.__call__.drain_after |  | 90 | 22.692 | 0.699 |
| mlx_glm_down_combine |  | 84 | 0.677 | 0.677 |
| MLA.__call__.drain_before |  | 66 | 17.382 | 0.614 |
| KDA.project_verify_rows.drain_after |  | 68 | 26.630 | 0.606 |
| MoE.__call__.drain_before |  | 84 | 25.548 | 0.569 |
| KDA.__call__ | model.layers.30.self_attn | 6 | 2.115 | 0.533 |
| mlx_dense_rows |  | 70 | 0.524 | 0.524 |
| MLP.__call__.drain_before |  | 90 | 4.135 | 0.505 |
| KDA.project_verify_rows.drain_before |  | 68 | 19.185 | 0.458 |
| KDA.__call__ | model.layers.18.self_attn | 6 | 2.113 | 0.455 |
| KDA.__call__ | model.layers.10.self_attn | 6 | 2.014 | 0.446 |
| KDA.__call__ | model.layers.8.self_attn | 6 | 2.018 | 0.444 |
| MoE.__call__ | model.layers.37.mlp | 2 | 2.098 | 0.375 |
| mx.quantized_matmul |  | 264 | 0.372 | 0.372 |
| DecoderLayer.verify_ffn_rows | model.layers.6 | 2 | 6.767 | 0.327 |
| DecoderLayer.verify_ffn_rows | model.layers.21 | 2 | 6.984 | 0.324 |
| DecoderLayer.verify_ffn_rows | model.layers.2 | 2 | 5.871 | 0.320 |
| DecoderLayer.verify_ffn_rows | model.layers.17 | 2 | 7.057 | 0.305 |
| MLA.__call__ | model.layers.23.self_attn | 6 | 3.138 | 0.301 |
| MLA.__call__ | model.layers.15.self_attn | 6 | 3.080 | 0.300 |
| MLA.__call__ | model.layers.35.self_attn | 6 | 3.063 | 0.297 |
| MLA.__call__ | model.layers.3.self_attn | 6 | 3.057 | 0.289 |
| MLA.__call__ | model.layers.39.self_attn | 6 | 3.180 | 0.289 |
| MLA.__call__ | model.layers.19.self_attn | 6 | 3.102 | 0.285 |
| MLA.__call__ | model.layers.27.self_attn | 6 | 3.120 | 0.283 |
| MLA.__call__ | model.layers.11.self_attn | 6 | 3.049 | 0.280 |

## Interpretation

- Use baseline decode_ms / generated_tokens for committed-token cost. Verify rows are speculative work, not committed throughput.
- Inspect dispatch.json for shape/dtype changes and returned_none fusion misses. Descriptors identify operands; they do not prove full-array casts or DRAM traffic.
- graph.dot contains only the still-lazy graph at export time. Evaluated or compiled regions can be opaque. Cast nodes prove graph conversion, not its measured cost.
- Inspect window.gputrace in Xcode for dispatched kernel names, resources and casts. Use Metal System Trace for device execution and CPU/GPU gaps; host timeline alone cannot distinguish bandwidth from compute saturation.
- Compare identical checkpoint, source hashes, prompt hash, steps and context. This replay excludes HTTP scheduling and network overhead.
