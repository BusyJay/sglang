# SPDX-License-Identifier: Apache-2.0
"""Refresh only routed B12X scale rows during concurrent global-cache fills."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module(int64_ids):
    dtype = "int64_t" if int64_ids else "int32_t"
    return load_jit(
        "qwen38_route_alphas",
        cuda_files=["moe/qwen38_route_alphas.cuh"],
        cuda_wrappers=[("run", f"qwen38_route_alphas<{dtype}>")],
    )


def refresh_route_alphas(ids, alpha1, alpha2, scale1, scale2, output1, output2):
    """Caller pins routed source rows; output tables must not alias the sources."""
    # Cached W4A16 accepts any scalar shape. The common scalar/vector forms
    # pass straight through; preserve higher-rank singleton compatibility.
    if scale1.ndim > 1:
        scale1 = scale1.reshape(())
    if scale2.ndim > 1:
        scale2 = scale2.reshape(())
    _module(ids.dtype == torch.int64).run(
        ids, alpha1, alpha2, scale1, scale2, output1, output2
    )
