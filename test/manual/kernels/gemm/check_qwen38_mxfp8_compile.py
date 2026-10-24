"""Compile each exact projection and tile for SM120 without executing a GPU."""

import json

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.kernels.ops.gemm.qwen38_mxfp8_small_m import (
    PROJECTION_SHAPES,
    _small_m_kernel,
)

for role, (n, k) in dict(PROJECTION_SHAPES, gdn_ba_padded=(128, 2560)).items():
    for m in (1, 3):
        for bn, bk in [(16, 128), (32, 128), (64, 128), (32, 256)]:
            src = ASTSource(
                _small_m_kernel,
                signature=dict(
                    a_ptr="*fp8e4nv",
                    w_ptr="*fp8e4nv",
                    a_scale_ptr="*u8",
                    w_scale_ptr="*u8",
                    out_ptr="*bf16",
                ),
                constexprs=dict(M=m, N=n, K=k, BLOCK_N=bn, BLOCK_K=bk),
            )
            c = triton.compile(
                src, target=GPUTarget("cuda", 120, 32), options={"num_warps": 4}
            )
            ptx = c.asm["ptx"]
            assert "kind::mxf8f6f4" in ptx, "Expected native SM120 blockscaled MMA"
            print(
                json.dumps(
                    dict(
                        role=role,
                        m=m,
                        n=n,
                        k=k,
                        bn=bn,
                        bk=bk,
                        shared=c.metadata.shared,
                        native_blockscaled=True,
                        compiled=True,
                    )
                )
            )
