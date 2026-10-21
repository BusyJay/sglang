# SPDX-License-Identifier: Apache-2.0
"""Native versus CSF W4A16/W4A4; rotate expert inputs to include memory traffic."""

import numpy as np
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import encode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("tokens", [1, 3, 16])
@marker.parametrize("activation_mode", ["w4a16", "w4a4"])
@marker.benchmark("encoding", ["native", "csf"])
def benchmark(tokens, activation_mode, encoding):
    h, i, e = 2560, 640, 16
    tensors = {
        "w13_weight": torch.randint(
            0, 256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8
        ),
        "w2_weight": torch.randint(
            0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8
        ),
        "g1_alphas": torch.ones(e, device="cuda"),
        "g2_alphas": torch.ones(e, device="cuda"),
        "w13_input_scale_quant": torch.tensor(1.0, device="cuda"),
        "w2_input_scale_quant": torch.tensor(1.0, device="cuda"),
    }
    specs = {}
    rng = np.random.default_rng(41)
    for name, rows, cols in (
        ("w13_blockscale_swizzled", 2 * i, h // 16),
        ("w2_blockscale_swizzled", h, i // 16),
    ):
        raw = rng.integers(20, 36, (e, rows, cols), dtype=np.uint8)
        raw.reshape(-1)[::100] = 65
        if encoding == "native":
            tensors[name] = torch.from_numpy(raw).cuda().view(torch.float8_e4m3fn)
        else:
            slabs = [encode(expert.tobytes(), rows, cols) for expert in raw]
            stride = (max(map(len, slabs)) + 15) // 16 * 16
            tensors[name] = torch.from_numpy(
                np.stack(
                    [np.frombuffer(s + bytes(stride - len(s)), np.uint8) for s in slabs]
                )
            ).cuda()
            specs[name] = dict(
                encoding="b12x_csf_bitmap_v1", rows=rows, columns=cols, stride=stride
            )
    x = (torch.randn(tokens, h, device="cuda") * 0.002).bfloat16()
    ids = torch.randint(0, e, (tokens, 10), device="cuda", dtype=torch.int32)
    router = torch.rand(tokens, 10, device="cuda")
    if activation_mode == "w4a4":
        if torch.cuda.get_device_capability()[0] != 12:
            return
        adapter = CachedB12xW4A4(
            torch.device("cuda"), h, i, e, 10, scale_encoding=specs
        )
        fn = adapter.run
    elif tokens <= 3 and torch.cuda.get_device_capability()[0] == 12:
        adapter = CachedB12xW4A16(torch.device("cuda"), h, i, e, scale_encoding=specs)
        fn = adapter.run
    else:
        fn = cached_w4a16_moe
    return marker.do_bench(
        fn, input_args=(x, ids, router, tensors), graph_clone_args="all"
    )


if __name__ == "__main__":
    benchmark.run()
