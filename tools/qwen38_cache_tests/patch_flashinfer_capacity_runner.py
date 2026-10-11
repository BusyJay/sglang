#!/usr/bin/env python3
"""Give each CUTLASS MoE expert capacity its own FlashInfer native runner.

FlashInfer's native workspace cache remembers only the last expert count. Its
Python runner cache currently excludes the expert count, so heterogeneous MoE
layers repeatedly run the expensive workspace-size calculation. This patch is
for the FlashInfer version bundled in swallow-sglang:v0.5.21.
"""

from pathlib import Path


def patch(source: str) -> str:
    marker = "# Qwen38: keep a native runner per expert capacity."
    if marker in source:
        return source
    replacements = (
        (
            """                torch.dtype,
                torch.dtype,
                torch.dtype,
                bool,""",
            """                torch.dtype,
                torch.dtype,
                torch.dtype,
                int,
                bool,""",
        ),
        (
            """            output_dtype: torch.dtype,
            top_k: int,""",
            """            output_dtype: torch.dtype,
            num_experts: int,
            top_k: int,""",
        ),
        (
            """            self.output_dtype = output_dtype
            self.top_k = top_k""",
            """            self.output_dtype = output_dtype
            self.num_experts = num_experts
            self.top_k = top_k""",
        ),
        (
            """                output_dtype,
                use_deepseek_fp8_block_scale,""",
            """                output_dtype,
                # Qwen38: keep a native runner per expert capacity.
                num_experts,
                use_deepseek_fp8_block_scale,""",
        ),
        (
            """            output_dtype=output_dtype,
            top_k=token_selected_experts.size(1),""",
            """            output_dtype=output_dtype,
            num_experts=int(fc1_expert_weights.shape[0]),
            top_k=token_selected_experts.size(1),""",
        ),
    )
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"unexpected FlashInfer core.py shape: {old!r} found {count} times")
        source = source.replace(old, new, 1)
    return source


if __name__ == "__main__":
    import flashinfer.fused_moe.core as core

    path = Path(core.__file__)
    before = path.read_text()
    after = patch(before)
    if after != before:
        path.write_text(after)
    print(f"FlashInfer per-capacity MoE runner patch: {path}")
