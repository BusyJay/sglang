#!/usr/bin/env python3
"""Compile small-verify W4A16 kernels for SM120; no torch or CUDA device needed."""

import ast
import json
import sys
import types
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def load_functions(path, names, extra=None):
    module = types.ModuleType(path.stem + "_compile_check")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    module.__dict__.update(extra or {})
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(a.name.startswith("triton") for a in node.names)
            or isinstance(node, ast.FunctionDef)
            and node.name in names
        )
    ]
    # Execute only the selected local JIT definitions, without importing torch.
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def main():
    root = Path(__file__).resolve().parents[2]
    quant = root / "python/sglang/srt/layers/quantization"
    decode = load_functions(quant / "qwen38_fp4_mtp.py", {"_decode"})
    kernels = load_functions(
        quant / "qwen38_cached_w4a16.py",
        {
            "_scale_offset",
            "_cache_project_splitk",
            "_cache_project_reuse",
            "_splitk_swiglu",
            "_splitk_combine",
        },
        {"_decode": decode._decode},
    )
    target = GPUTarget("cuda", 120, 32)
    for routes in (20, 30):
        for split in (1, 2, 4, 8):
            for first, k, n in ((True, 2560, 640), (False, 640, 2560)):
                source = ASTSource(
                    kernels._cache_project_splitk,
                    signature={
                        "X": "*bf16",
                        "W": "*u8",
                        "S": "*fp8e4nv",
                        "G": "*fp32",
                        "Q": "*fp32",
                        "IDs": "*i32",
                        "Router": "*fp32",
                        "Y": "*fp32",
                        "E": "i32",
                    },
                    constexprs={
                        "K": k,
                        "N": n,
                        "TOPK": 10,
                        "FIRST": first,
                        "FUSE_SWIGLU": False,
                        "SPLIT_K": split,
                        "BN": 16,
                        "BK": 256,
                        "ROUTES": routes,
                    },
                )
                triton.compile(source, target=target, options={"num_warps": 4})
                if split == 1:
                    signature = {
                        "X": "*bf16",
                        "W": "*u8",
                        "S": "*fp8e4nv",
                        "G": "*fp32",
                        "Q": "*fp32",
                        "IDs": "*i32",
                        "Router": "*fp32",
                        "Y": "*bf16" if first else "*fp32",
                        "E": "i32",
                    }
                    constants = dict(
                        K=k, N=n, TOPK=10, FIRST=first, BN=16, BK=256, ROUTES=routes
                    )
                    if first:
                        triton.compile(
                            ASTSource(
                                kernels._cache_project_splitk,
                                signature=signature,
                                constexprs={
                                    **constants,
                                    "SPLIT_K": 1,
                                    "FUSE_SWIGLU": True,
                                },
                            ),
                            target=target,
                            options={"num_warps": 4},
                        )
                    triton.compile(
                        ASTSource(
                            kernels._cache_project_reuse,
                            signature=signature,
                            constexprs={
                                **constants,
                                "TOKENS": routes // 10,
                                "BT": 4,
                                "BR": 32,
                                "BTOP": 16,
                            },
                        ),
                        target=target,
                        options={"num_warps": 4},
                    )
            for name, signature, constants in (
                (
                    "_splitk_swiglu",
                    {"Partial": "*fp32", "Y": "*bf16"},
                    {"N": 640, "ROUTES": routes, "SPLIT_K": split, "BLOCK": 256},
                ),
                (
                    "_splitk_combine",
                    {"Partial": "*fp32", "Router": "*fp32", "Y": "*bf16"},
                    {
                        "N": 2560,
                        "ROUTES": routes,
                        "TOPK": 10,
                        "SPLIT_K": split,
                        "BLOCK": 128,
                    },
                ),
            ):
                triton.compile(
                    ASTSource(
                        getattr(kernels, name),
                        signature=signature,
                        constexprs=constants,
                    ),
                    target=target,
                    options={"num_warps": 4},
                )
            print(json.dumps({"routes": routes, "split_k": split, "compiled": True}))


if __name__ == "__main__":
    main()
