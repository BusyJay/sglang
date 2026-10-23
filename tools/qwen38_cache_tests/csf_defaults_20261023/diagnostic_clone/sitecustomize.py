"""Diagnostic only: remove shared scale-plane offsets without host fences."""

import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4


def wrap(cls):
    original = cls.run

    def run(self, x, ids, router, tensors, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            return original(self, x, ids, router, tensors, **kwargs)
        tensors = dict(tensors)
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            tensors[name] = tensors[name].clone()
        return original(self, x, ids, router, tensors, **kwargs)

    cls.run = run


wrap(CachedB12xW4A16)
wrap(CachedB12xW4A4)
