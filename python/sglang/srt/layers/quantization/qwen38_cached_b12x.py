# SPDX-License-Identifier: Apache-2.0
"""Pinned FlashInfer 0.7.2 native ModelOpt small-M W4A16 adapter.

Weight views are borrowed only for this invocation. Cache transport and its
reader events remain owned by Qwen38ExpertCacheRuntime.
"""

from __future__ import annotations

import torch


class CachedB12xW4A16:
    max_tokens = 3
    max_topk = 32

    def __init__(self, device, hidden, width, max_experts):
        import flashinfer
        from b12x.moe._shared.kernels.w4a16 import kernel
        from b12x.moe._shared.kernels.w4a16.prepare import W4A16ModelOptWeights

        if flashinfer.__version__ != "0.7.2":
            raise RuntimeError("Qwen38 B12X adapter requires FlashInfer 0.7.2")
        if torch.cuda.get_device_capability(device)[0] != 12:
            raise RuntimeError("Qwen38 B12X requires SM12x")
        if not kernel.MoEMicroKernelW4A16SmallMDirect.is_supported(
            m=3,
            hidden_size=hidden,
            intermediate_size=width,
            topk=self.max_topk,
            num_experts=max_experts,
        ):
            raise ValueError("unsupported Qwen38 B12X geometry")
        self.kernel, self.weights_type = kernel, W4A16ModelOptWeights
        self.hidden, self.width, self.max_experts = hidden, width, max_experts
        # Allocate bounded scratch before KV sizing, independent of pool views.
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)
        self.globals = torch.empty(2, max_experts, dtype=torch.float32, device=device)
        chunks = ((width // 2) + 127) // 128
        self.intermediate = torch.empty(
            2 * self.max_tokens * chunks * 128 * self.max_topk,
            dtype=torch.bfloat16,
            device=device,
        )
        self.unused = torch.empty(0, dtype=torch.bfloat16, device=device)
        # Graph capture may wait on the last eager use of this scratch.
        # External nodes preserve that dependency across capture/replay.
        self.done = torch.cuda.Event(external=True)
        self.used = False

    @classmethod
    def eligible(cls, x, ids, phase):
        return (
            phase != "prefill"
            and x.dtype == torch.bfloat16
            and 1 <= x.shape[0] <= cls.max_tokens
            and ids.ndim == 2
            and ids.shape[0] == x.shape[0]
            and 1 <= ids.shape[1] <= cls.max_topk
        )

    def run(self, x, ids, router, tensors):
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
            validate_cache_tensors,
        )

        capacity = tensors["w13_weight"].shape[0]
        validate_cache_tensors(tensors, self.hidden, self.width, capacity)
        if capacity > self.max_experts:
            raise ValueError("B12X cache capacity exceeds reserved scratch")
        if not self.eligible(x, ids, "decode") or not x.is_cuda:
            raise ValueError("B12X requires 1..3 CUDA BF16 tokens")
        if not self.kernel._small_m_direct_supported(
            m=x.shape[0],
            hidden_size=self.hidden,
            intermediate_size=self.width,
            num_experts=capacity,
            topk=ids.shape[1],
            activation="silu",
            apply_router_weight_on_input=False,
            swiglu_limit=None,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            element_dtype="bf16",
            weight_layout="modelopt",
            w13_layout="w13",
        ):
            raise RuntimeError("B12X small-M direct disabled or unsupported")
        stream = torch.cuda.current_stream(x.device)
        if self.used:
            stream.wait_event(self.done)
        # Refresh after each fill: cache alpha includes input scale; A16
        # cancels that scale. No full weight/scales copy or startup snapshot.
        g1, g2 = self.globals[:, :capacity].unbind(0)
        torch.mul(tensors["g1_alphas"], tensors["w13_input_scale_quant"], out=g1)
        torch.mul(tensors["g2_alphas"], tensors["w2_input_scale_quant"], out=g2)
        prepared = self.weights_type(
            w13=tensors["w13_weight"],
            w13_scale=tensors["w13_blockscale_swizzled"],
            w13_global_scale=g1,
            w2=tensors["w2_weight"],
            w2_scale=tensors["w2_blockscale_swizzled"],
            w2_global_scale=g2,
            workspace=self.workspace,
            hidden_size=self.hidden,
            intermediate_size=self.width,
            num_experts=capacity,
            is_gated=True,
            params_dtype=torch.bfloat16,
            micro_w13_scale=tensors["w13_blockscale_swizzled"],
            micro_w2_scale=tensors["w2_blockscale_swizzled"],
            micro_w13_global_scale=g1,
            micro_w2_global_scale=g2,
            # B12X w13 means physical [up, gate]; w31 means [gate, up].
            w13_layout="w13",
        )
        output = torch.empty_like(x)
        try:
            return self.kernel.run_w4a16_moe(
                x.contiguous(),
                prepared,
                router.float().contiguous(),
                ids.contiguous(),
                activation="silu",
                intermediate_cache13=self.unused,
                intermediate_cache2=self.intermediate,
                output=output,
                route_mode="direct",
            )
        finally:
            self.done.record(stream)
            self.used = True
