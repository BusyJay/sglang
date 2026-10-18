# SPDX-License-Identifier: Apache-2.0
"""Reusable auxiliary storage for expert-cache migration and computation."""

import torch


class AdaptiveMetadataBuffers:
    """Reserve policy/plan storage once; changing capacity only changes views."""

    def __init__(self, num_layers, num_experts, capacity, device):
        self.device = []
        self.host = []
        for _ in range(num_layers):
            sizes = {
                "lookup": (num_experts, torch.int32),
                "keys": (capacity, torch.int32),
                "frequency": (capacity, torch.float32),
                "ages": (capacity, torch.int64),
                "generation": (capacity, torch.int64),
                "clock": (1, torch.int64),
                "counts": (num_experts + 1, torch.int32),
                "plan": (4 + 4 * capacity, torch.int32),
                "prefill_plan": (num_experts + 1 + capacity, torch.int32),
            }
            self.device.append(
                {
                    name: torch.empty(size, dtype=dtype, device=device)
                    for name, (size, dtype) in sizes.items()
                }
            )
            self.host.append(
                {
                    name: torch.empty_like(
                        tensor, device="cpu", pin_memory=device.type == "cuda"
                    )
                    for name, tensor in self.device[-1].items()
                    if name not in ("counts", "plan", "prefill_plan")
                }
            )

    def reset(self, layer, slots):
        buffers = self.device[layer]
        if not 0 < slots <= buffers["keys"].numel():
            raise ValueError("adaptive metadata capacity exceeded")
        state = {
            name: tensor[:slots]
            if name in ("keys", "frequency", "ages", "generation")
            else tensor
            for name, tensor in buffers.items()
        }
        state["plan"] = buffers["plan"][: 4 + 4 * slots]
        state["prefill_plan"] = buffers["prefill_plan"][
            : buffers["counts"].numel() + slots
        ]
        for name, tensor in state.items():
            if name not in ("counts", "plan", "prefill_plan"):
                tensor.fill_(-1 if name in ("lookup", "keys") else 0)
        return state

    def restore(self, layer, state, records, *, non_blocking=False):
        """Stage policy on CPU, then copy into existing GPU storage."""
        host = self.host[layer]
        for name, tensor in host.items():
            tensor.fill_(-1 if name in ("lookup", "keys") else 0)
        for slot, (expert, frequency, age) in enumerate(records):
            host["lookup"][expert] = slot
            host["keys"][slot] = expert
            host["frequency"][slot] = frequency
            host["ages"][slot] = age
            host["generation"][slot] = 1
        host["clock"][0] = max((record[2] for record in records), default=0)
        for name, tensor in host.items():
            state[name].copy_(tensor[: state[name].numel()], non_blocking=non_blocking)

    def nbytes(self):
        return sum(
            t.numel() * t.element_size() for b in self.device for t in b.values()
        )


class GrowableSwigluBuffers:
    """Keep one set of parameter arrays, growing only at a new high water mark."""

    def __init__(self, config, device, maximum):
        self.config = config
        self.device = device
        self.maximum = maximum
        self.capacity = 0
        self.buffers = None

    def get(self, required):
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            materialize_swiglu_params_for_cutlass,
        )

        if not 0 < required <= self.maximum:
            raise ValueError("invalid SwiGLU expert capacity")
        if self.buffers is None or required > self.capacity:
            capacity = min(self.maximum, max(required, self.capacity * 2))
            self.buffers = materialize_swiglu_params_for_cutlass(
                self.config, capacity, self.device
            )
            self.capacity = capacity
        # The caller uses these views on its compute stream. Protect storage
        # against allocator reuse if a later call grows from another stream.
        if self.device.type == "cuda":
            for tensor in self.buffers:
                if tensor is not None:
                    tensor.record_stream(torch.cuda.current_stream(self.device))
        return tuple(
            tensor[:required] if tensor is not None else None for tensor in self.buffers
        )
