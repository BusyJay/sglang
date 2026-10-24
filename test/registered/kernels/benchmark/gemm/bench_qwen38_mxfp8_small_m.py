"""Per-projection M=1/3 tiles vs pinned CUTLASS, with rotating cold-L2 inputs."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.qwen38_mxfp8_small_m import (
    PROJECTION_SHAPES,
    qwen38_mxfp8_small_m,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)


def run(q, w, s, sw, provider, logical_n):
    if provider == "cutlass_pinned":
        from flashinfer import mm_mxfp8
        from flashinfer.autotuner import autotune

        with autotune(False, skip_ops={"mxfp8_gemm"}):
            return mm_mxfp8(q, w.T, s, sw, out_dtype=torch.bfloat16, backend="cutlass")[
                :, :logical_n
            ].contiguous()
    block_n, block_k = map(int, provider.split("x"))
    return qwen38_mxfp8_small_m(
        q, w[:logical_n], s, sw, block_n=block_n, block_k=block_k
    )


@marker.parametrize("role", list(PROJECTION_SHAPES), ["shared_gate_up", "shared_down"])
@marker.parametrize("m", [1, 3])
@marker.benchmark(
    "provider", ["cutlass_pinned", "16x128", "32x128", "64x128", "32x256"]
)
def benchmark(role, m, provider):
    import flashinfer

    if torch.cuda.get_device_capability() != (12, 0):
        return marker.skip("SM120 required")
    n, k = PROJECTION_SHAPES[role]
    logical_n = n
    if n < 128:
        n = 128  # same compute view as the serving CUTLASS path
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    q, s = flashinfer.mxfp8_quantize(x, True)
    w, sw = flashinfer.mxfp8_quantize(weight, True)
    return marker.do_bench(
        run,
        input_args=(q, w, s, sw, provider, logical_n),
        graph_clone_args=(0, 1, 2, 3),
        memory_args=(q, w, s, sw),
    )


if __name__ == "__main__":
    benchmark.run()
