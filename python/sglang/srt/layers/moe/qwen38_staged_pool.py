# SPDX-License-Identifier: Apache-2.0
"""Contiguous adaptive expert pool with bidirectional physical backing stages."""

from math import prod

import torch

from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS


class StagedExpertPool:
    def __init__(self, manifest, device, base_layout, minimum):
        self.base_layout = list(base_layout)
        self.minimum = minimum
        self.maximum = manifest.num_experts
        self.base_budget = sum(base_layout)
        self.maximum_budget = len(base_layout) * self.maximum
        first = manifest.tensors[0]
        self.rows = {
            name: prod(first[name].shape[1:])
            * getattr(torch, first[name].dtype).itemsize
            for name in REQUIRED_TENSORS
        }
        dev = torch.device(device)
        device_id = dev.index if dev.index is not None else torch.cuda.current_device()
        self.arena = KvVmmArena(
            device_id,
            sum(self.rows.values()) * self.maximum_budget + len(self.rows) * (32 << 20),
        )
        self.tensors = {}
        self.offsets = {}
        with torch.cuda.use_mem_pool(self.arena.pool):
            for name in REQUIRED_TENSORS:
                self.tensors[name] = torch.empty(
                    (self.maximum_budget, *first[name].shape[1:]),
                    dtype=getattr(torch, first[name].dtype),
                    device=device,
                )
        for name, tensor in self.tensors.items():
            offset = tensor.data_ptr() - self.arena.base
            self.offsets[name] = offset
            self.arena.commit_range(offset, self.base_budget * self.rows[name])
            tensor[: self.base_budget].zero_()
        self.budgets = None
        self.loan_mappings = []
        self.stage_index = 0
        self.stage_layouts = None

    def physical_bytes(self, budget):
        return sum(self.arena._align(budget * row) for row in self.rows.values())

    def configure_loans(self, loan_bytes):
        if self.budgets is not None:
            raise RuntimeError("expert loan may only be configured once")
        base = self.physical_bytes(self.base_budget)
        self.budgets = []
        for loan in loan_bytes:
            low, high = self.base_budget, self.maximum_budget
            while low < high:
                mid = (low + high + 1) // 2
                if self.physical_bytes(mid) - base <= loan:
                    low = mid
                else:
                    high = mid - 1
            self.budgets.append(low)
        self.grow_stage(0, initial=True)
        return list(self.budgets)

    def pool_views(self, budget):
        return {name: tensor[:budget] for name, tensor in self.tensors.items()}

    def grow_stage(self, index, *, initial=False):
        if not initial and not 0 <= index < self.stage_index:
            raise ValueError("expert growth requires a smaller KV stage")
        previous = self.base_budget if initial else self.budgets[self.stage_index]
        # Commit intermediate boundaries so future returns can release exact
        # extension handles, even when jumping from 220K straight to 80K.
        for stage in range(1 if initial else self.stage_index - 1, index - 1, -1):
            budget = self.budgets[stage]
            mappings = []
            for name, offset in self.offsets.items():
                start = self.arena._align(previous * self.rows[name])
                end = self.arena._align(budget * self.rows[name])
                if end > start:
                    self.arena.commit_range(offset, end)
                    mappings.append((offset, start))
            self.loan_mappings.append((budget, mappings))
            previous = budget
        self.stage_index = index

    def release_stage(self, index):
        if not self.stage_index < index < len(self.budgets):
            raise ValueError("expert return requires a larger KV stage")
        target = self.budgets[index]
        while self.loan_mappings and self.loan_mappings[-1][0] > target:
            _, mappings = self.loan_mappings.pop()
            for offset, keep in mappings:
                self.arena.release_tail(offset, keep)
        self.stage_index = index

    def close(self):
        self.tensors.clear()
        self.arena.close()
