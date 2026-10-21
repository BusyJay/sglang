# SPDX-License-Identifier: Apache-2.0
"""Compare selected alpha refresh, rotating inputs/outputs to defeat L2 reuse."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.qwen38_route_alphas import refresh_route_alphas
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def reference(ids, a, b, s1, s2, out1, out2):
    slots = ids.flatten().clamp_min(0).long()
    out1.index_copy_(0, slots, a[slots] * s1)
    out2.index_copy_(0, slots, b[slots] * s2)


@marker.parametrize("tokens", [1, 3])
@marker.parametrize("capacity", [1025, 7006])
@marker.benchmark("implementation", ["torch", "fused"])
def benchmark(tokens, capacity, implementation):
    ids = torch.randint(-1, capacity, (tokens, 10), device="cuda", dtype=torch.int32)
    a, b, out1, out2 = [torch.rand(capacity, device="cuda") for _ in range(4)]
    s1, s2 = torch.rand(2, device="cuda").unbind()
    return marker.do_bench(
        reference if implementation == "torch" else refresh_route_alphas,
        input_args=(ids, a, b, s1, s2, out1, out2),
        memory_output=(out1, out2),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
