# GLM MLX performance evidence

Host spans measure Python build/enqueue/wait wall time, not GPU kernels. Sync attribution adds explicit drains and changes graph lifetime/overlap. Inclusive rows overlap; self rows subtract nested recorded spans.

Workload: `verify`; synthetic: `True`; steps: `1`; checkpoint revision: `unrecorded`.

## Alternating measured trials

| Mode | First ms | Median steady ms | Relative to baseline | Token hash equal |
|---|---:|---:|---:|---|
| baseline | 140.416 | 2.151 | 1.000x | True |
| host | 4.517 | 2.970 | 1.380x | True |
| sync | 21.819 | 10.694 | 4.971x | True |

## host recorded host spans

| Operation | Module | Calls | Inclusive ms | Self ms |
|---|---|---:|---:|---:|
| mx.async_eval |  | 2 | 2.442 | 2.442 |
| DecoderLayer.verify_ffn_rows | model.layers.0 | 2 | 0.785 | 0.384 |
| mx.eval |  | 2 | 0.370 | 0.370 |
| mlx_hc_collapse_norm |  | 32 | 0.318 | 0.318 |
| MLA.__call__ | model.layers.3.self_attn | 6 | 0.297 | 0.269 |
| mlx_kda_decode |  | 18 | 0.207 | 0.207 |
| DecoderLayer.verify_ffn_rows | model.layers.1 | 2 | 0.496 | 0.163 |
| Model.verify_mtp_rows |  | 2 | 5.176 | 0.162 |
| DecoderLayer.verify_ffn_rows | model.layers.2 | 2 | 0.530 | 0.158 |
| DecoderLayer.verify_ffn_rows | model.layers.3 | 2 | 0.747 | 0.149 |
| KDA.__call__ | model.layers.0.self_attn | 6 | 0.193 | 0.121 |
| KDA.__call__ | model.layers.1.self_attn | 6 | 0.154 | 0.087 |
| mlx_affine_rows |  | 22 | 0.083 | 0.083 |
| KDA.__call__ | model.layers.2.self_attn | 6 | 0.150 | 0.082 |
| forced_verify |  | 2 | 5.621 | 0.075 |
| MoE.__call__ | model.layers.3.mlp | 2 | 0.223 | 0.066 |
| _compiled_quantized_ffn |  | 8 | 0.065 | 0.065 |
| mlx_dense_rows |  | 8 | 0.047 | 0.047 |
| KDA.project_verify_rows | model.layers.0.self_attn | 2 | 0.059 | 0.045 |
| MLP.__call__ | model.layers.3.mlp.shared_experts | 2 | 0.091 | 0.038 |
| MLP.__call__ | model.layers.0.mlp | 2 | 0.060 | 0.038 |
| MLP.__call__ | model.layers.1.mlp | 2 | 0.054 | 0.034 |
| MLP.__call__ | model.layers.2.mlp | 2 | 0.051 | 0.033 |
| KDA.project_verify_rows | model.layers.1.self_attn | 2 | 0.042 | 0.033 |
| KDA.project_verify_rows | model.layers.2.self_attn | 2 | 0.040 | 0.031 |
| MLA._latent_matrices | model.layers.3.self_attn | 6 | 0.028 | 0.028 |
| mlx_glm_router |  | 2 | 0.018 | 0.018 |
| DecoderLayer._branch_output | model.layers.0 | 8 | 0.017 | 0.017 |
| DecoderLayer._branch_output | model.layers.1 | 8 | 0.017 | 0.017 |
| DecoderLayer._branch_output | model.layers.2 | 8 | 0.015 | 0.015 |
| DecoderLayer._branch_output | model.layers.3 | 8 | 0.014 | 0.014 |
| mlx_glm_down_combine |  | 2 | 0.012 | 0.012 |

## sync recorded host spans

| Operation | Module | Calls | Inclusive ms | Self ms |
|---|---|---:|---:|---:|
| mx.eval |  | 102 | 14.262 | 14.262 |
| mx.synchronize |  | 100 | 2.409 | 2.409 |
| MLA.__call__ | model.layers.3.self_attn | 6 | 2.640 | 0.357 |
| mlx_hc_collapse_norm |  | 32 | 0.296 | 0.296 |
| DecoderLayer.verify_ffn_rows | model.layers.1 | 2 | 4.342 | 0.279 |
| DecoderLayer.verify_ffn_rows | model.layers.0 | 2 | 4.577 | 0.274 |
| DecoderLayer.verify_ffn_rows | model.layers.2 | 2 | 4.460 | 0.250 |
| mlx_kda_decode |  | 18 | 0.244 | 0.244 |
| DecoderLayer.verify_ffn_rows | model.layers.3 | 2 | 6.352 | 0.226 |
| KDA.__call__ | model.layers.0.self_attn | 6 | 1.456 | 0.198 |
| KDA.__call__.drain_before |  | 18 | 1.063 | 0.194 |
| KDA.__call__.drain_after |  | 18 | 3.480 | 0.180 |
| Model.verify_mtp_rows |  | 2 | 20.902 | 0.176 |
| DecoderLayer.verify_ffn_rows.drain_after |  | 8 | 1.685 | 0.126 |
| KDA.__call__ | model.layers.2.self_attn | 6 | 1.365 | 0.117 |
| KDA.__call__ | model.layers.1.self_attn | 6 | 1.325 | 0.107 |
| forced_verify |  | 2 | 21.091 | 0.107 |
| MoE.__call__ | model.layers.3.mlp | 2 | 1.174 | 0.097 |
| KDA.project_verify_rows | model.layers.2.self_attn | 2 | 0.529 | 0.080 |
| MLA.__call__.drain_before |  | 6 | 1.314 | 0.078 |
| mlx_affine_rows |  | 22 | 0.076 | 0.076 |
| DecoderLayer.verify_ffn_rows.drain_before |  | 8 | 0.537 | 0.061 |
| MLA.__call__.drain_after |  | 6 | 2.248 | 0.060 |
| MLP.__call__.drain_after |  | 8 | 1.560 | 0.059 |
| _compiled_quantized_ffn |  | 8 | 0.057 | 0.057 |
| KDA.project_verify_rows.drain_after |  | 6 | 1.272 | 0.056 |
| MLP.__call__.drain_before |  | 8 | 1.546 | 0.055 |
| mlx_dense_rows |  | 8 | 0.052 | 0.052 |
| MLP.__call__ | model.layers.2.mlp | 2 | 0.479 | 0.049 |
| KDA.project_verify_rows | model.layers.0.self_attn | 2 | 0.487 | 0.045 |
| KDA.project_verify_rows | model.layers.1.self_attn | 2 | 0.463 | 0.040 |
| KDA.project_verify_rows.drain_before |  | 6 | 1.480 | 0.039 |
| MLP.__call__ | model.layers.3.mlp.shared_experts | 2 | 0.484 | 0.039 |
| MLP.__call__ | model.layers.0.mlp | 2 | 0.446 | 0.037 |
| Model.verify_mtp_rows.drain_after |  | 2 | 0.443 | 0.037 |
| MLP.__call__ | model.layers.1.mlp | 2 | 0.417 | 0.035 |
| MLA._latent_matrices | model.layers.3.self_attn | 6 | 0.035 | 0.035 |
| DecoderLayer._branch_output | model.layers.0 | 8 | 0.034 | 0.034 |
| Model.verify_mtp_rows.drain_before |  | 2 | 0.077 | 0.027 |
| DecoderLayer._branch_output | model.layers.3 | 8 | 0.024 | 0.024 |

## Interpretation

- Use baseline decode_ms / generated_tokens for committed-token cost. Verify rows are speculative work, not committed throughput.
- Inspect dispatch.json for shape/dtype changes and returned_none fusion misses. Descriptors identify operands; they do not prove full-array casts or DRAM traffic.
- graph.dot contains only the still-lazy graph at export time. Evaluated or compiled regions can be opaque. Cast nodes prove graph conversion, not its measured cost.
- Inspect window.gputrace in Xcode for dispatched kernel names, resources and casts. Use Metal System Trace for device execution and CPU/GPU gaps; host timeline alone cannot distinguish bandwidth from compute saturation.
- Compare identical checkpoint, source hashes, prompt hash, steps and context. This replay excludes HTTP scheduling and network overhead.
