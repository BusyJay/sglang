# SPDX-License-Identifier: Apache-2.0
"""Measure CPU launch preparation only, with real B12X policy and a no-op kernel.

    PYTHONPATH=python python3 test/manual/kernels/moe/bench_qwen38_b12x_binding.py

No CUDA execution is measured. Both modes use the same cached workspace and
alpha refresh; vendor mode repeats vendor launch preparation for every chunk.
Large weight tensors reserve CPU virtual address space without touching pages.
"""

import json
import random
import statistics
import time
from unittest.mock import patch

import torch
from b12x.moe.fused_moe import _impl

from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4


def main():
    torch.set_num_threads(1)
    e, h, i, topk = 7000, 2560, 640, 10
    t = dict(
        w13_weight=torch.empty(e, 2 * i, h // 2, dtype=torch.uint8),
        w2_weight=torch.empty(e, h, i // 2, dtype=torch.uint8),
        w13_blockscale_swizzled=torch.empty(e, 131072, dtype=torch.uint8),
        w2_blockscale_swizzled=torch.empty(e, 65536, dtype=torch.uint8),
        g1_alphas=torch.ones(e),
        g2_alphas=torch.ones(e),
        w13_input_scale_quant=torch.tensor(0.7),
        w2_input_scale_quant=torch.tensor(1.1),
    )
    specs = {
        "w13_blockscale_swizzled": dict(
            encoding="b12x_csf_bitmap_v1", rows=1280, columns=160, stride=131072
        ),
        "w2_blockscale_swizzled": dict(
            encoding="b12x_csf_bitmap_v1", rows=2560, columns=40, stride=65536
        ),
    }
    with patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
        adapter = CachedB12xW4A4(
            torch.device("cpu"), h, i, e, topk, scale_encoding=specs
        )
    kernel = adapter.kernel
    boundary = dict(
        _get_dynamic_kernel=lambda *a, **kw: (lambda *args: None, 1),
        _get_impl_mac=lambda *a, **kw: 1,
        current_cuda_stream=lambda: _impl.cuda.CUstream(0),
        _dynamic_external_route_plan_supported=lambda **kw: False,
    )
    prepared = adapter.prepare_launch
    rng = random.Random(42)
    result = []
    with (
        patch.dict(kernel._launch_dynamic_flat.__globals__, boundary),
        patch.dict(prepared.__globals__, boundary),
        patch.object(kernel, "current_cuda_stream", boundary["current_cuda_stream"]),
    ):
        for m in (128, 129, 1024):
            x = torch.empty(m, h, dtype=torch.bfloat16)
            ids = torch.zeros(m, topk, dtype=torch.int32)
            router = torch.ones(m, topk)
            samples = {"vendor": [], "prepared": []}
            for iteration in range(14):
                modes = list(samples)
                rng.shuffle(modes)
                for mode in modes:
                    adapter.prepare_launch = (
                        prepared if mode == "prepared" else lambda **kw: None
                    )
                    adapter.bindings.clear()
                    adapter.run(x, ids, router, t)
                    start = time.perf_counter_ns()
                    for _ in range(100):
                        adapter.run(x, ids, router, t)
                    elapsed = (time.perf_counter_ns() - start) / 100 / 1000
                    if iteration >= 4:
                        samples[mode].append(elapsed)
            median = {k: statistics.median(v) for k, v in samples.items()}
            row = dict(
                tokens=m,
                capacity=e,
                median_us=median,
                reduction_percent=100 * (1 - median["prepared"] / median["vendor"]),
                samples_us=samples,
            )
            result.append(row)
            print(json.dumps(row), flush=True)
    print(
        json.dumps(dict(scope="CPU preparation; no GPU kernel execution", cases=result))
    )


if __name__ == "__main__":
    main()
