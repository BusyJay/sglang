# Route layout source findings

MLX `_group_expert_select` returns `argpartition(...)[..., :top_k]`. Actual
GLM FP32 activations use this fallback. MLX CustomKernel checks row_contiguous
and allocates/copies each noncontiguous input unless ensure_row_contiguous=False.
GatherQMM carries rhs-index strides into its native kernel. This supports an
implicit route layout cost hypothesis; it does not establish an actual copy count
in a captured serving dispatch. macro_contiguous materialization regresses native
by ~2.45 ms but does not change candidate timing. A runtime stride reader is
prepared for a direct no-copy ablation.

Source inspected: existing local MLX source archive and authenticated remote
installed mlx_lm. Device tests are required before adoption.
