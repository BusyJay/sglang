#!/usr/bin/env python3
"""Compile the routing extension for SM120 without loading weights or a GPU."""

from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.moe.qwen38_expert_cache import _module

with override_jit_cuda_arch(12, 0):
    _module()
print("SM120 routing JIT extension compiled and loaded; kernel was not executed.")
