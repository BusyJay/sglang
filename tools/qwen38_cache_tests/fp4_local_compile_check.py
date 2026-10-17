#!/usr/bin/env python3
"""Compile FP4-Local Triton kernels for SM120 without executing CUDA."""

import json

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.srt.layers.quantization.qwen38_fp4_mtp import _project
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


def compile_kernels():
    target = GPUTarget("cuda", 120, 32)
    for first, k, n in [(True, 2560, 640), (False, 640, 2560)]:
        source = ASTSource(
            _project,
            signature={
                "X": "*bf16",
                "W": "*u8",
                "S": "*fp8e4nv",
                "G": "*fp32",
                "IDs": "*i64",
                "Router": "*fp32",
                "Y": "*bf16" if first else "*fp32",
            },
            constexprs={
                "K": k,
                "N": n,
                "TOPK": 10,
                "E": 512,
                "FIRST": first,
                "BN": 32,
                "BK": 256,
            },
        )
        kernel = triton.compile(source, target=target, options={"num_warps": 4})
        print(
            json.dumps(
                {
                    "kernel": "mtp_gate_up" if first else "mtp_down",
                    "num_warps": kernel.metadata.num_warps,
                    "compiled": True,
                }
            )
        )
    source = ASTSource(
        _gather_ple_embedding_from_pinned_kernel,
        signature={
            "weight_ptr": "i64",
            "ids_ptr": "*i64",
            "output_ptr": "*bf16",
            "embedding_dim": "i32",
            "tp_vocab_start": "i32",
            "tp_vocab_end": "i32",
        },
        constexprs={
            "is_fp8": False,
            "BLOCK_D": 256,
            "MASK_INVALID": True,
            "IS_NVFP4": True,
        },
    )
    triton.compile(source, target=target, options={"num_warps": 4})
    print(json.dumps({"kernel": "ple_gather", "compiled": True}))


if __name__ == "__main__":
    compile_kernels()
