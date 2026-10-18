"""Compare both launch paths on identical live tensors and nonaligned capacities."""

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from tools.qwen38_cache_tests.w4a16_verify_gpu_check import swizzle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20261018)
    h, n, topk = 2560, 640, 10
    with envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.override(False):
        baseline = CachedB12xW4A16(torch.device("cuda"), h, n, 512)
    with envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.override(True):
        prepared = CachedB12xW4A16(torch.device("cuda"), h, n, 512)
    reports = []
    for capacity in (32, 133, 512):
        tensors = {}
        for prefix, rows, cols, alpha in [
            ("w13", 2 * n, h, "g1_alphas"),
            ("w2", h, n, "g2_alphas"),
        ]:
            tensors[prefix + "_weight"] = torch.randint(
                0, 256, (capacity, rows, cols // 2), dtype=torch.uint8, device="cuda"
            )
            scale = (
                torch.randint(1, 6, (capacity, rows, cols // 16))
                .float()
                .mul_(0.125)
                .to(torch.float8_e4m3fn)
            )
            tensors[prefix + "_blockscale_swizzled"] = swizzle(scale)
            tensors[prefix + "_input_scale_quant"] = torch.tensor(1.3, device="cuda")
            tensors[alpha] = torch.linspace(0.015, 0.035, capacity, device="cuda")
        for m in (1, 2, 3):
            for dtype in (torch.int32, torch.int64):
                x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda") * 0.2
                ids = torch.randint(0, capacity, (m, topk), dtype=dtype, device="cuda")
                router = torch.randn(m, topk, device="cuda").softmax(-1)
                for repeat in range(3):
                    a = baseline.run(x, ids, router, tensors)
                    b = prepared.run(x, ids, router, tensors)
                    a2 = baseline.run(x, ids, router, tensors)
                    b2 = prepared.run(x, ids, router, tensors)
                    torch.cuda.synchronize()
                    report = dict(
                        capacity=capacity,
                        tokens=m,
                        ids_dtype=str(dtype),
                        repeat=repeat,
                        cross_max_abs=float((a.float() - b.float()).abs().max()),
                        baseline_repeat_max_abs=float(
                            (a.float() - a2.float()).abs().max()
                        ),
                        prepared_repeat_max_abs=float(
                            (b.float() - b2.float()).abs().max()
                        ),
                    )
                    reports.append(report)
                    args.output.write_text(
                        json.dumps(dict(reports=reports), indent=2) + "\n"
                    )
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    torch.testing.assert_close(a, a2, atol=0, rtol=0)
                    torch.testing.assert_close(b, b2, atol=0, rtol=0)
                    x.mul_(0.9)
                    ids.copy_(ids.roll(1, 1))
                    tensors["g2_alphas"].mul_(0.99)
        del tensors
    args.output.write_text(
        json.dumps(dict(status="passed", reports=reports), indent=2) + "\n"
    )
    print("PASS", len(reports), "bit-exact paired launches and repeated launches")


if __name__ == "__main__":
    main()
