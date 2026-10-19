"""GPU lookup routing without per-call host metadata transfers."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.benchmark("tokens", [1, 3, 2048, 8192])
def benchmark(tokens):
    ids = torch.randint(0, 512, (tokens, 10), device="cuda", dtype=torch.int32)
    lookup = torch.arange(512, device="cuda", dtype=torch.int32)
    return marker.do_bench(
        cache_partition_routes,
        input_args=(ids, lookup, 128),
        graph_clone_args=(0, 1),
        memory_args=(ids, lookup),
    )


if __name__ == "__main__":
    benchmark.run()
