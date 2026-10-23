# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned, bounded W4A4 scratch shared across serial expert layers."""

import inspect
from collections import OrderedDict
from dataclasses import replace

import torch
from sglang.kernels.ops.moe.qwen38_b12x_w4a4 import (
    dynamic_kernel_namespace,
    prepare_dynamic_launch,
)


class CachedB12xW4A4:
    chunk_tokens = 128
    tile_m = 16

    def __init__(
        self,
        device,
        hidden,
        width,
        max_experts,
        topk,
        *,
        scale_encoding,
        deterministic_output=False,
    ):
        if torch.cuda.get_device_capability(device)[0] != 12:
            raise RuntimeError("Qwen38 B12X W4A4 requires SM12x")
        if hidden % 128 or width % 128:
            raise ValueError("B12X cached W4A4 requires H and I divisible by 128")
        self.hidden, self.width, self.max_experts, self.topk = (
            hidden,
            width,
            max_experts,
            topk,
        )
        self.specs = scale_encoding
        self.deterministic_output = deterministic_output
        self.inline = bool(scale_encoding)
        if self.inline:
            self.kernel = dynamic_kernel_namespace(
                dict(scale_encoding, _w13_rows=2 * width)
            )
        else:
            from b12x.moe.fused_moe import _impl

            self.kernel = _impl
        routed = self.chunk_tokens * topk
        self.workspace = self.kernel._alloc_workspace(
            "dynamic",
            "nvfp4",
            max_experts,
            max_experts,
            hidden,
            width,
            topk,
            device,
            torch.bfloat16,
            torch.ones(1, device=device),
            torch.ones(1, device=device),
            routed_rows=routed,
            max_rows=routed,
            input_scales_static=False,
            deterministic_output=deterministic_output,
            dynamic_tile_m=self.tile_m,
            dynamic_route_mode="grouped",
        )
        # The alpha/scalar vectors are small scratch, not another weight copy.
        aligned = (max_experts + 3) // 4 * 4
        self.alphas = torch.empty((2, aligned), dtype=torch.float32, device=device)
        self.parameters = inspect.signature(self.kernel._launch_dynamic_flat).parameters
        self.prepare_launch = prepare_dynamic_launch(self.kernel)
        self.pool_key = None
        self.values = None
        self.bindings = OrderedDict()
        self.dynamic_pointers = OrderedDict()
        self.done = torch.cuda.Event(external=True) if device.type == "cuda" else None
        self.used = False

    def _bind_pool(self, e, tensors):
        # Cache scratch views and raw pointers, never retain expert storage.
        key = (
            e,
            *(
                tensors[n].data_ptr()
                for n in (
                    "w13_weight",
                    "w2_weight",
                    "w13_blockscale_swizzled",
                    "w2_blockscale_swizzled",
                )
            ),
        )
        if key == self.pool_key:
            return self.values
        self.bindings.clear()
        ws = replace(
            self.workspace,
            weight_E=e,
            state_E=e,
            row_counts=self.workspace.row_counts[:e],
            expert_write_rows=self.workspace.expert_write_rows[:e],
            expert_tile_base=self.workspace.expert_tile_base[: e + 1],
            input_gs=self.workspace.input_gs[:e],
            down_input_scale=self.workspace.down_input_scale[:e],
        )
        values = {n: getattr(ws, n) for n in self.parameters if hasattr(ws, n)}
        values.update(
            w1_alpha=self.alphas[0, :e],
            w2_alpha=self.alphas[1, :e],
            E=e,
            k=self.hidden,
            n=self.width,
            num_topk=self.topk,
            max_rows=self.chunk_tokens * self.topk,
            physical_tiles_capacity=ws.physical_tiles_capacity,
            task_capacity=ws.task_capacity,
            fast_math=True,
            activation="silu",
            quant_mode="nvfp4",
            w4a8_repacked=False,
            w4a8_n64_repacked=False,
            nvfp4_materialize_intermediate=False,
            nvfp4_inline_scales=self.inline,
            share_input_across_experts=False,
            deterministic_output=self.deterministic_output,
            swiglu_limit=None,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            external_route_plan_requested=False,
            policy_max_active_clusters=-1,
            volatile_launch_state=True,
            planned_tile_m=self.tile_m,
            planned_direct_routing=False,
            planned_num_tokens=self.chunk_tokens,
        )
        for name in (
            "sfb_w13_mx",
            "sfb_down_mx",
            "w13_residual",
            "down_residual",
            "w13_rp",
            "w13_sfb_rp",
            "down_rp",
            "down_sfb_rp",
        ):
            values[name] = ws.row_counts
        self.pool_key, self.values = key, values
        return values

    def run(self, x, ids, router, tensors):
        stream = torch.cuda.current_stream(x.device) if self.done is not None else None
        if self.used:
            stream.wait_event(self.done)
        try:
            return self._run(x, ids, router, tensors)
        finally:
            if stream is not None:
                self.done.record(stream)
                self.used = True

    def _run(self, x, ids, router, tensors):
        if (
            x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[1] != self.hidden
            or ids.ndim != 2
            or ids.shape != router.shape
            or ids.shape[0] != x.shape[0]
            or ids.shape[1] != self.topk
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("B12X W4A4 requires BF16 inputs and matching top-k routes")
        e = tensors["w13_weight"].shape[0]
        if not 0 < e <= self.max_experts:
            raise ValueError("B12X W4A4 pool capacity exceeds reserved scratch")
        if tuple(tensors["w13_weight"].shape) != (
            e,
            2 * self.width,
            self.hidden // 2,
        ) or tuple(tensors["w2_weight"].shape) != (e, self.hidden, self.width // 2):
            raise ValueError("B12X W4A4 weight geometry differs from pack")
        for name, rows, cols in (
            ("w13_blockscale_swizzled", 2 * self.width, self.hidden // 16),
            ("w2_blockscale_swizzled", self.hidden, self.width // 16),
        ):
            t = tensors[name]
            spec = self.specs.get(name)
            shape = (e, spec["stride"]) if spec else (e, rows, cols)
            if tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError("B12X W4A4 scale slot layout differs from manifest")
        if (
            ids.device != x.device
            or router.device != x.device
            or any(
                t.device != x.device or not t.is_contiguous() for t in tensors.values()
            )
        ):
            raise ValueError(
                "B12X W4A4 requires contiguous operands on the input device"
            )
        values = self._bind_pool(e, tensors)
        values["input_gs"].copy_(tensors["w13_input_scale_quant"].expand(e))
        values["down_input_scale"].copy_(tensors["w2_input_scale_quant"].expand(e))
        # Alphas remain live after slot fills; no startup value snapshots.
        values["w1_alpha"].copy_(tensors["g1_alphas"])
        values["w2_alpha"].copy_(tensors["g2_alphas"])
        # The vendor scatter path atomically adds routed expert contributions.
        # Its public launcher supplies zeros; the prepared launcher bypasses
        # that allocation, so own the same initialization here. Deterministic
        # reduction writes every output element instead.
        allocate = torch.empty if self.deterministic_output else torch.zeros
        output = allocate(x.shape, dtype=x.dtype, device=x.device)
        w13 = tensors["w13_weight"].permute(1, 2, 0)
        down = tensors["w2_weight"].permute(1, 2, 0)
        # Conversion, when required by the ABI, happens once for the full call.
        # Contiguous BF16/FP32 inputs borrow caller storage without a copy.
        x, ids, router = x.contiguous(), ids.contiguous(), router.float().contiguous()
        k = self.kernel

        def ptr(dtype, t, align=16):
            key = (dtype, t.data_ptr(), align)
            value = self.dynamic_pointers.get(key)
            if value is not None:
                self.dynamic_pointers.move_to_end(key)
                return value
            value = k.make_ptr(
                dtype, t.data_ptr(), k.cute.AddressSpace.gmem, assumed_align=align
            )
            self.dynamic_pointers[key] = value
            if len(self.dynamic_pointers) > 32:
                self.dynamic_pointers.popitem(last=False)
            return value

        for start in range(0, x.shape[0], self.chunk_tokens):
            end = min(start + self.chunk_tokens, x.shape[0])
            a, ii, rr, result = (
                x[start:end],
                ids[start:end],
                router[start:end],
                output[start:end],
            )
            if self.deterministic_output:
                result = self.workspace.route_output
            key = (end - start, ids.dtype)
            binding = self.bindings.get(key)
            if binding is None:
                live = dict(
                    values,
                    a=a,
                    flat_ids=ii.reshape(-1),
                    flat_weights=rr.reshape(-1),
                    scatter_output=result,
                    w13_fp4=w13,
                    down_fp4=down,
                    w13_sf=tensors["w13_blockscale_swizzled"],
                    down_sf=tensors["w2_blockscale_swizzled"],
                    m=end - start,
                    routed_rows=(end - start) * self.topk,
                    topk_ids_are_i32=ids.dtype == torch.int32,
                )
                binding = self.prepare_launch(**live)
                if binding is not None:
                    compiled, args = binding
                    if len(args) != 45:
                        raise RuntimeError("B12X NVFP4 dynamic launch ABI changed")
                    # Dynamic pointers and stream must never be captured here.
                    args = list(args)
                    for index in (0, 1, 2, 22, 25, 34, 44):
                        args[index] = None
                    binding = (compiled, tuple(args))
                    self.bindings[key] = binding
                    if len(self.bindings) > 8:
                        self.bindings.popitem(last=False)
                else:
                    # Environment-enabled external planner keeps vendor behavior.
                    self.kernel._launch_dynamic_flat(**live)
                    if self.deterministic_output:
                        k._launch_dynamic_topk_sum(
                            route_output=result,
                            output=output[start:end],
                            m=end - start,
                            num_topk=self.topk,
                            k=self.hidden,
                            route_expert_ids=ii.reshape(-1),
                            stream=k.current_cuda_stream(),
                        )
                    continue
            self.bindings.move_to_end(key)
            compiled, fixed = binding
            args = list(fixed)
            args[0] = ptr(k.cutlass.BFloat16, a)
            args[1] = ptr(
                k.cutlass.Int32 if ids.dtype == torch.int32 else k.cutlass.Int64,
                ii,
                4 if ids.dtype == torch.int32 else 8,
            )
            args[2] = ptr(k.cutlass.Float32, rr, 4)
            args[22], args[25] = w13, down
            args[34] = ptr(k.cutlass.BFloat16, result)
            args[44] = k.current_cuda_stream()
            values["barrier_count"].zero_()
            values["barrier_epoch"].zero_()
            compiled(*args)
            if self.deterministic_output:
                k._launch_dynamic_topk_sum(
                    route_output=result,
                    output=output[start:end],
                    m=end - start,
                    num_topk=self.topk,
                    k=self.hidden,
                    route_expert_ids=ii.reshape(-1),
                    stream=k.current_cuda_stream(),
                )
        return output
