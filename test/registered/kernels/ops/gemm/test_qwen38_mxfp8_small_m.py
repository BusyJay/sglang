"""Native MXFP8 projection accuracy, row invariance and graph replay on SM120."""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm.qwen38_mxfp8_small_m import (
    PROJECTION_SHAPES,
    qwen38_mxfp8_small_m,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def inputs(n, k):
    import flashinfer

    torch.manual_seed(n + k)
    x = torch.randn(3, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    q, s = flashinfer.mxfp8_quantize(x, True)
    qw, sw = flashinfer.mxfp8_quantize(w, True)
    return x, q, s, qw, sw


def dequant(q, scales):
    rows, k = q.shape
    r = torch.arange(rows, device=q.device)[:, None]
    g = torch.arange(k // 32, device=q.device)[None, :]
    offsets = ((r // 128) * ((k // 32 + 3) // 4) + g // 4) * 512
    offsets = offsets + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
    sf = torch.exp2(scales.flatten()[offsets].double() - 127)
    return (q.double().reshape(rows, k // 32, 32) * sf[..., None]).reshape(rows, k)


@pytest.mark.parametrize("shape", list(PROJECTION_SHAPES.values()) + [(128, 2560)])
@pytest.mark.parametrize("tile", [(16, 128), (32, 128), (64, 128), (32, 256)])
def test_projection_accuracy_and_row_invariance(shape, tile):
    n, k = shape
    _, q, s, qw, sw = inputs(n, k)
    out = qwen38_mxfp8_small_m(q, qw, s, sw, block_n=tile[0], block_k=tile[1])
    ref = dequant(q, s) @ dequant(qw, sw).T
    # BF16 output rounding plus FP32 accumulation, relative to quantized inputs.
    torch.testing.assert_close(out.double(), ref, rtol=5e-3, atol=2e-3)
    from flashinfer import mm_mxfp8
    from flashinfer.autotuner import autotune

    with autotune(False, skip_ops={"mxfp8_gemm"}):
        baseline = (
            mm_mxfp8(q, qw.T, s, sw, out_dtype=torch.bfloat16, backend="cutlass")
            if n >= 128
            else None
        )
    if baseline is not None:
        candidate_error = (out.double() - ref).abs().max()
        baseline_error = (baseline.double() - ref).abs().max()
        assert candidate_error <= baseline_error * 1.05 + 1e-6
    for row in range(3):
        # Copy scale bytes by layout, rather than requantizing the activation.
        one_s = torch.zeros_like(s)
        groups = torch.arange(k // 32, device=q.device)
        dst = (groups // 4) * 512 + groups % 4
        src = dst + row * 16
        one_s[dst] = s[src]
        single = qwen38_mxfp8_small_m(
            q[row : row + 1].contiguous(),
            qw,
            one_s,
            sw,
            block_n=tile[0],
            block_k=tile[1],
        )
        torch.testing.assert_close(single[0], out[row], rtol=0, atol=0)


def test_graph_replay_updates_payload_and_scales():
    import flashinfer

    n, k = PROJECTION_SHAPES["shared_down"]
    x, q, s, qw, sw = inputs(n, k)
    qwen38_mxfp8_small_m(q, qw, s, sw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = qwen38_mxfp8_small_m(q, qw, s, sw)
    for factor in (0.0, 0.1, 1.0, 10.0):
        new_q, new_s = flashinfer.mxfp8_quantize(x * factor, True)
        q.copy_(new_q)
        s.copy_(new_s)
        graph.replay()
        ref = dequant(q, s) @ dequant(qw, sw).T
        torch.testing.assert_close(out.double(), ref, rtol=5e-3, atol=2e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
