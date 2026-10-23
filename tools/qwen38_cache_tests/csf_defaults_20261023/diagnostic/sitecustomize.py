"""Scope a diagnostic fence to prefill or decode; explicit PYTHONPATH only."""

import os
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16

mode = os.environ.get("QWEN38_DIAG_SYNC", "prefill")


def wrap(cls, label):
    original = cls.run

    def run(self, *args, **kwargs):
        if label == mode and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().synchronize()
        result = original(self, *args, **kwargs)
        return result

    cls.run = run


wrap(CachedB12xW4A4, "prefill")
wrap(CachedB12xW4A16, "decode")
