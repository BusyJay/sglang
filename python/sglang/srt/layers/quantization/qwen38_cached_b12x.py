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

    def __init__(self, kernel, workspace, intermediate, output_dtype=None):
        self.kernel = kernel
        self.output_dtype = output_dtype or torch.bfloat16
        self.bindings = OrderedDict()
        self.dynamic_pointers = OrderedDict()
        self.barrier_state = workspace[-2:]
        self.barriers = (workspace[-2:-1], workspace[-1:])
        self.intermediate = intermediate.view(torch.uint32)

    def run(self, x, ids, router, tensors, g1, g2, output, stream, hybrid=None):
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
            compile_capacity = (
                1 << (capacity - 1).bit_length()
                if getattr(k, "_qwen38_capacity_bucket", False)
                else capacity
            )
            launch = k._compile_w4a16_small_m_direct(
                m=x.shape[0],
                hidden_size=x.shape[1],
                intermediate_size=tensors["w2_weight"].shape[2] * 2,
                num_experts=compile_capacity,
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
            tail = (
                *self.barriers,
                k.Int32(capacity),
                k.Int32(x.shape[0]),
                k.Int32(launch.grid_x),
            )
            binding = (launch, fixed, tail)
            self.bindings[key] = binding
            if len(self.bindings) > 8:
                self.bindings.popitem(last=False)
        self.bindings.move_to_end(key)
        launch, fixed, tail = binding

        # x/IDs/router/output may change addresses every eager invocation.
        def ptr(dtype, tensor):
            key = (dtype, tensor.data_ptr())
            value = self.dynamic_pointers.get(key)
            if value is not None:
                self.dynamic_pointers.move_to_end(key)
                return value
            value = k.make_ptr(
                dtype, tensor.data_ptr(), k.cute.AddressSpace.gmem, assumed_align=16
            )
            self.dynamic_pointers[key] = value
            if len(self.dynamic_pointers) > 32:
                self.dynamic_pointers.popitem(last=False)
            return value

        ids_dtype = k.cutlass.Int64 if ids.dtype == torch.int64 else k.cutlass.Int32
        if k._small_m_direct_host_barrier_reset_enabled():
            # Same reset and epoch protocol as the validated upstream path.
            self.barrier_state.zero_()
        args = (
            ptr(k.cutlass.BFloat16, x),
            *fixed,
            ptr(ids_dtype, ids),
            ptr(k.cutlass.Float32, router),
            ptr(
                (
                    k.cutlass.Float32
                    if self.output_dtype == torch.float32
                    else k.cutlass.BFloat16
                ),
                output,
            ),
            *tail,
        )
        if hybrid is not None:
            plan, completed_address, result_address, failed_address, sequence = hybrid
            args += (
                ptr(k.cutlass.Int32, plan),
                k.make_ptr(
                    k.cutlass.Int32, completed_address, k.cute.AddressSpace.gmem
                ),
                k.make_ptr(k.cutlass.Float32, result_address, k.cute.AddressSpace.gmem),
                k.make_ptr(k.cutlass.Int32, failed_address, k.cute.AddressSpace.gmem),
                k.Int32(sequence),
            )
        launch.compiled(*args, k.cuda.CUstream(stream.cuda_stream))
        return output


class CachedB12xW4A16:
    max_tokens = 3
    max_topk = 32

    def __init__(
        self,
        device,
        hidden,
        width,
        max_experts,
        *,
        enable_fp32=False,
        scale_encoding=None,
        native_load_bits=0,
    ):
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
        self.scale_encoding = scale_encoding or {}
        self.kernel, self.weights_type = kernel, W4A16ModelOptWeights
        self.hidden, self.width, self.max_experts = hidden, width, max_experts
        # Allocate bounded scratch before KV sizing, independent of pool views.
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(sms * 4 + 2, dtype=torch.int32, device=device)
        # Each alpha row must keep the native 16-byte pointer alignment,
        # including non-multiple-of-four global slot capacities.
        alpha_stride = (max_experts + 3) // 4 * 4
        self.globals = torch.empty(2, alpha_stride, dtype=torch.float32, device=device)
        self.global_views = OrderedDict()
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

        self.fp32_launch = None
        self.hybrid_launch = None
        if enable_fp32:
            from sglang.srt.layers.quantization.qwen38_b12x_fp32 import (
                fp32_kernel_namespace,
            )

            self.fp32_launch = _PreparedSmallMLaunch(
                fp32_kernel_namespace(),
                self.workspace,
                self.intermediate,
                torch.float32,
            )
            from sglang.srt.layers.quantization.qwen38_b12x_hybrid import (
                hybrid_kernel_namespace,
            )

            self.hybrid_launch = _PreparedSmallMLaunch(
                hybrid_kernel_namespace(), self.workspace, self.intermediate
            )
        self.prepared_launch = (
            _PreparedSmallMLaunch(kernel, self.workspace, self.intermediate)
            if envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.get()
            else None
        )

        # Explicit tuning only: wider loads do not consistently improve the
        # complete expert call, especially when its working set stays hot.
        if native_load_bits:
            if self.scale_encoding or (hidden, width) != (2560, 640):
                raise ValueError(
                    "native vector loads require uncompressed Qwen38 weights"
                )
            from sglang.srt.layers.quantization.qwen38_b12x_vector import (
                vector_kernel_namespace,
            )

            self.prepared_launch = _PreparedSmallMLaunch(
                vector_kernel_namespace(kernel, native_load_bits),
                self.workspace,
                self.intermediate,
            )
            if enable_fp32:
                self.fp32_launch = _PreparedSmallMLaunch(
                    vector_kernel_namespace(fp32_kernel_namespace(), native_load_bits),
                    self.workspace,
                    self.intermediate,
                    torch.float32,
                )
                self.hybrid_launch = _PreparedSmallMLaunch(
                    vector_kernel_namespace(
                        hybrid_kernel_namespace(), native_load_bits
                    ),
                    self.workspace,
                    self.intermediate,
                )

        if self.scale_encoding:
            from sglang.srt.layers.quantization.qwen38_b12x_csf import (
                csf_kernel_namespace,
            )

            self.prepared_launch = _PreparedSmallMLaunch(
                csf_kernel_namespace(kernel, self.scale_encoding),
                self.workspace,
                self.intermediate,
            )
            if enable_fp32:
                self.fp32_launch = _PreparedSmallMLaunch(
                    csf_kernel_namespace(fp32_kernel_namespace(), self.scale_encoding),
                    self.workspace,
                    self.intermediate,
                    torch.float32,
                )
                self.hybrid_launch = _PreparedSmallMLaunch(
                    csf_kernel_namespace(
                        hybrid_kernel_namespace(), self.scale_encoding
                    ),
                    self.workspace,
                    self.intermediate,
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

    def run(self, x, ids, router, tensors, *, output_dtype=None, hybrid=None):
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
            validate_cache_tensors,
        )

        capacity = tensors["w13_weight"].shape[0]
        validate_cache_tensors(tensors, self.hidden, self.width, capacity)
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            spec = self.scale_encoding.get(name)
            if (spec is None and tensors[name].ndim == 2) or (
                spec is not None and tensors[name].shape != (capacity, spec["stride"])
            ):
                raise ValueError(
                    "B12X scale tensors differ from the compiled CSF encoding"
                )
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
        views = self.global_views.get(capacity)
        if views is None:
            views = self.globals[:, :capacity].unbind(0)
            self.global_views[capacity] = views
            if len(self.global_views) > 8:
                self.global_views.popitem(last=False)
        self.global_views.move_to_end(capacity)
        g1, g2 = views
        if self.fp32_launch is not None:
            from sglang.kernels.ops.moe.qwen38_route_alphas import refresh_route_alphas

            # Other layers may fill asynchronously. Read only this window's
            # pinned routes and the immutable zero slot for masked IDs.
            refresh_route_alphas(
                ids,
                tensors["g1_alphas"],
                tensors["g2_alphas"],
                tensors["w13_input_scale_quant"],
                tensors["w2_input_scale_quant"],
                g1,
                g2,
            )
        else:
            torch.mul(tensors["g1_alphas"], tensors["w13_input_scale_quant"], out=g1)
            torch.mul(tensors["g2_alphas"], tensors["w2_input_scale_quant"], out=g2)
        launch = (
            self.hybrid_launch
            if hybrid is not None
            else (
                self.fp32_launch
                if output_dtype == torch.float32
                else self.prepared_launch
            )
        )
        if output_dtype == torch.float32 and launch is None:
            raise RuntimeError("B12X FP32 output requires enable_fp32 at startup")
        if output_dtype not in (None, torch.bfloat16, torch.float32):
            raise ValueError("unsupported B12X output dtype")
        if hybrid is not None and output_dtype not in (None, torch.bfloat16):
            raise ValueError("hybrid B12X writes final BF16 output")
        if hybrid is not None and launch is None:
            raise RuntimeError("B12X hybrid output requires enable_fp32 at startup")
        if launch is not None:
            try:
                return launch.run(
                    x.contiguous(),
                    ids.contiguous(),
                    router.float().contiguous(),
                    tensors,
                    g1,
                    g2,
                    torch.empty_like(x, dtype=output_dtype or x.dtype),
                    stream,
                    hybrid=hybrid,
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
