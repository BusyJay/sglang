"""Offline SM120 compile / CPU interpreter checks, or a GPU graph benchmark.

Run with --compile, --interpret (TRITON_INTERPRET=1), or --benchmark.
CPU interpretation verifies arithmetic only; it does not validate GPU execution.
"""

import argparse
import json

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.gemm.hc_mix import (
    _hc_mix_down_partial_kernel,
    _hc_mix_reduce_silu_kernel,
    _hc_mix_up_gate_kernel,
    fused_hc_mix,
)


def compile_check():
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    for dtype in ["bf16", "fp16"]:
        for rows, hc, hs, rank in [
            (1, 4, 2560, 320),
            (3, 4, 2560, 320),
            (16, 4, 2560, 320),
            (3, 4, 512, 40),
        ]:
            k = hc * hs
            chunks = triton.cdiv(k, 256)
            cases = [
                (
                    _hc_mix_down_partial_kernel,
                    dict(x_ptr=f"*{dtype}", w_ptr=f"*{dtype}", partial_ptr="*fp32"),
                    dict(
                        K=k,
                        LOWRANK=rank,
                        NUM_ROWS=rows,
                        ROWS=16,
                        BLOCK_N=32,
                        BLOCK_K=256,
                    ),
                ),
                (
                    _hc_mix_reduce_silu_kernel,
                    dict(partial_ptr="*fp32", t_ptr=f"*{dtype}"),
                    dict(
                        LOWRANK=rank,
                        ROWS=16,
                        CHUNKS=chunks,
                        HC=hc,
                        BLOCK_C=triton.next_power_of_2(chunks),
                        BLOCK_N=32,
                    ),
                ),
                (
                    _hc_mix_up_gate_kernel,
                    dict(
                        x_ptr=f"*{dtype}",
                        w_ptr=f"*{dtype}",
                        t_ptr=f"*{dtype}",
                        out_ptr=f"*{dtype}",
                    ),
                    dict(
                        LOWRANK=rank,
                        HS=hs,
                        NUM_ROWS=rows,
                        ROWS=16,
                        HC=hc,
                        BLOCK_J=32,
                        BLOCK_R=64,
                    ),
                ),
            ]
            for fn, signature, constants in cases:
                kernel = triton.compile(
                    ASTSource(fn, signature, constants),
                    target=GPUTarget("cuda", 120, 32),
                    options={"num_warps": 4},
                )
                print(
                    json.dumps(
                        dict(
                            kernel=fn.__name__,
                            rows=rows,
                            dtype=dtype,
                            hs=hs,
                            rank=rank,
                            shared_bytes=kernel.metadata.shared,
                            compiled=True,
                        )
                    )
                )


def interpreter_check():
    torch.manual_seed(42)
    # FP16 avoids the interpreter's unsupported BF16 host conversions.
    for rows, hs, rank in [
        (1, 512, 40),
        (3, 512, 40),
        (16, 512, 40),
        (1, 2560, 320),
        (3, 2560, 320),
    ]:
        hc = 4
        k = hc * hs
        x = torch.randn(rows, k, dtype=torch.float16)
        down = torch.randn(rank, k, dtype=torch.float16) * 0.02
        up = torch.randn(k, rank, dtype=torch.float16) * 0.02
        out = fused_hc_mix(x, down, up, hc, hs, split_k=True)
        xd = x.double()
        t = F.silu(F.linear(xd, down.double()) / hc)
        gates = F.linear(t, up.double()).sigmoid()
        ref = (gates.view(rows, hc, hs) * xd.view(rows, hc, hs)).mean(1)
        torch.testing.assert_close(out.double(), ref, rtol=2e-3, atol=1e-3)
        print(
            json.dumps(
                dict(
                    rows=rows,
                    hs=hs,
                    rank=rank,
                    max_abs_error=(out.double() - ref).abs().max().item(),
                    passed=True,
                )
            )
        )


def benchmark():
    from triton.testing import do_bench_cudagraph

    torch.manual_seed(42)
    hc, hs, rank = 4, 2560, 320
    for rows in [1, 3, 16]:
        for dtype in [torch.bfloat16, torch.float16]:
            x = torch.randn(rows, hc * hs, dtype=dtype, device="cuda")
            down = torch.randn(rank, hc * hs, dtype=dtype, device="cuda") * 0.02
            up = torch.randn(hc * hs, rank, dtype=dtype, device="cuda") * 0.02
            samples = {False: [], True: []}
            for trial in range(5):
                for split_k in [False, True] if trial % 2 == 0 else [True, False]:
                    fn = lambda: fused_hc_mix(x, down, up, hc, hs, split_k=split_k)
                    fn()
                    samples[split_k].append(do_bench_cudagraph(fn, rep=200) * 1000)
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        dtype=str(dtype),
                        persistent_us=samples[False],
                        split_k_us=samples[True],
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compile", action="store_true")
    group.add_argument("--interpret", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.compile:
        compile_check()
    elif args.interpret:
        interpreter_check()
    else:
        benchmark()
