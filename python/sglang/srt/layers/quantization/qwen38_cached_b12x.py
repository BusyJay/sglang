# SPDX-License-Identifier: Apache-2.0
"""Pinned FlashInfer 0.7.2 native ModelOpt small-M W4A16 adapter.

Weight views are borrowed only for this invocation. Cache transport and its
reader events remain owned by Qwen38ExpertCacheRuntime.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class _PreparedSmallMLaunch:
    """Bounded pointer bindings; no cached activations or expert pool tensors.

    Match FlashInfer 0.7.2's native small-M launch ABI exactly. The compiled
    program stays wrapped by B12X so graph program retention still works.
    """

    def __init__(self, kernel, workspace, intermediate):
        self.kernel = kernel
        self.bindings = OrderedDict()
        self.barrier_state = workspace[-2:]
        self.barriers = (workspace[-2:-1], workspace[-1:])
        self.intermediate = intermediate.view(torch.uint32)

    def run(self, x, ids, router, tensors, g1, g2, output, stream):
        k = self.kernel
        capacity = tensors["w13_weight"].shape[0]
        weights = (
            tensors["w13_weight"],
            tensors["w13_blockscale_swizzled"],
            tensors["w2_weight"],
            tensors["w2_blockscale_swizzled"],
        )
        key = (
            x.device,
            x.shape[0],
            ids.shape[1],
            ids.dtype,
            capacity,
            *(value.data_ptr() for value in weights),
            g1.data_ptr(),
            g2.data_ptr(),
        )
        binding = self.bindings.get(key)
        if binding is None:
            launch = k._compile_w4a16_small_m_direct(
                m=x.shape[0],
                hidden_size=x.shape[1],
                intermediate_size=tensors["w2_weight"].shape[2] * 2,
                num_experts=capacity,
                topk=ids.shape[1],
                activation="silu",
                fast_math=True,
                topk_ids_dtype=ids.dtype,
                device=x.device,
                w13_layout="w13",
            )

            def ptr(dtype, tensor):
                return k.make_ptr(
                    dtype, tensor.data_ptr(), k.cute.AddressSpace.gmem, assumed_align=16
                )

            # Native ABI passes the same raw global twice for FC1 and FC2.
            p_g1, p_g2 = ptr(k.cutlass.Float32, g1), ptr(k.cutlass.Float32, g2)
            fixed = (
                ptr(k.cutlass.Uint8, weights[0]),
                ptr(k.cutlass.Uint8, weights[1]),
                p_g1,
                p_g1,
                p_g2,
                ptr(k.cutlass.Uint32, self.intermediate),
                ptr(k.cutlass.Uint8, weights[2]),
                ptr(k.cutlass.Uint8, weights[3]),
                p_g2,
            )
            binding = (launch, fixed)
            self.bindings[key] = binding
            if len(self.bindings) > 8:
                self.bindings.popitem(last=False)
        self.bindings.move_to_end(key)
        launch, fixed = binding

        # x/IDs/router/output may change addresses every eager invocation.
        def ptr(dtype, tensor):
            return k.make_ptr(
                dtype, tensor.data_ptr(), k.cute.AddressSpace.gmem, assumed_align=16
            )

        ids_dtype = k.cutlass.Int64 if ids.dtype == torch.int64 else k.cutlass.Int32
        if k._small_m_direct_host_barrier_reset_enabled():
            # Same reset and epoch protocol as the validated upstream path.
            self.barrier_state.zero_()
        launch.compiled(
            ptr(k.cutlass.BFloat16, x),
            *fixed,
            ptr(ids_dtype, ids),
            ptr(k.cutlass.Float32, router),
            ptr(k.cutlass.BFloat16, output),
            *self.barriers,
            k.Int32(capacity),
            k.Int32(x.shape[0]),
            k.Int32(launch.grid_x),
            k.cuda.CUstream(stream.cuda_stream),
        )
        return output


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
        from sglang.srt.environ import envs

        self.prepared_launch = (
            _PreparedSmallMLaunch(kernel, self.workspace, self.intermediate)
            if envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.get()
            else None
        )

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
        if (
            x.shape[1] != self.hidden
            or router.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or any(t.device != x.device for t in (ids, router, *tensors.values()))
        ):
            raise ValueError("B12X input geometry, route dtype or device mismatch")
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
        if getattr(self, "prepared_launch", None) is not None:
            try:
                return self.prepared_launch.run(
                    x.contiguous(),
                    ids.contiguous(),
                    router.float().contiguous(),
                    tensors,
                    g1,
                    g2,
                    torch.empty_like(x),
                    stream,
                )
            finally:
                self.done.record(stream)
                self.used = True
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
