import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 7, _FUSED_MIX_MAX_ROWS])
@pytest.mark.parametrize("split_k", [False, True])
def test_fused_hc_mix_matches_reference(dtype, num_tokens, split_k):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, split_k=split_k)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


@pytest.mark.parametrize("split_k", [False, True])
def test_fused_hc_mix_no_less_accurate_than_eager(split_k):
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, split_k=split_k)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_k_masks_partial_lowrank_tiles(dtype):
    x, w_down, w_up = _make_inputs(3, dtype)
    w_down = w_down[:40].contiguous()
    w_up = w_up[:, :40].contiguous()
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, split_k=True)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.double(), ref, **_TOLERANCES[dtype])


@pytest.mark.parametrize("num_tokens", [1, 3, _FUSED_MIX_MAX_ROWS])
def test_split_k_graph_replay_reads_updated_inputs(num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, torch.bfloat16)
    for _ in range(3):
        fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, split_k=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, split_k=True)
    for scale in [0.5, -1.0, 2.0, 0.0]:
        x.mul_(scale)
        graph.replay()
        first = out.clone()
        graph.replay()
        torch.testing.assert_close(out, first, rtol=0, atol=0)
        ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
        torch.testing.assert_close(out.double(), ref, **_TOLERANCES[x.dtype])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
