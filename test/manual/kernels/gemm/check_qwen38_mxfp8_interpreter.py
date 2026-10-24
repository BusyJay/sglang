"""CPU arithmetic/layout checks; not evidence of GPU numerical equivalence."""

import json

import torch
import triton

from sglang.kernels.ops.gemm.qwen38_mxfp8_small_m import _small_m_kernel


def pack(rows, k):
    groups = k // 32
    q = torch.randn(rows, k).clamp(-3, 3).to(torch.float8_e4m3fn)
    linear = torch.randint(123, 131, (rows, groups), dtype=torch.uint8)
    scales = torch.zeros(
        ((rows + 127) // 128) * 128 * ((groups + 3) // 4) * 4, dtype=torch.uint8
    )
    r = torch.arange(rows)[:, None]
    g = torch.arange(groups)[None, :]
    idx = (
        ((r // 128) * ((groups + 3) // 4) + g // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + g % 4
    )
    scales[idx] = linear
    deq = (
        q.double().reshape(rows, groups, 32)
        * torch.exp2(linear.double() - 127)[..., None]
    ).reshape(rows, k)
    return q, scales, deq


torch.manual_seed(42)
for n, k in [(96, 2560), (1280, 2560), (2560, 640)]:
    a, sa, da = pack(3, k)
    w, sw, dw = pack(n, k)
    for m in (1, 3):
        out = torch.empty(m, n, dtype=torch.float32)
        _small_m_kernel[(triton.cdiv(n, 32),)](
            a[:m], w, sa, sw, out, m, n, k, BLOCK_N=32, BLOCK_K=128
        )
        ref = da[:m] @ dw.T
        torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-4)
        print(
            json.dumps(
                dict(
                    m=m,
                    n=n,
                    k=k,
                    max_abs_error=(out.double() - ref).abs().max().item(),
                    passed=True,
                )
            )
        )
