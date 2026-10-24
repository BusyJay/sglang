# SPDX-License-Identifier: Apache-2.0
"""Vectorized native weights preserve batched B12X output and route masking."""

import unittest

import torch

from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def native_tensors():
    torch.manual_seed(293)
    h, i, e = 2560, 640, 4
    tensors = {
        "w13_weight": torch.randint(
            0, 256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8
        ),
        "w2_weight": torch.randint(
            0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8
        ),
        "g1_alphas": torch.rand(e, device="cuda") * 0.4,
        "g2_alphas": torch.rand(e, device="cuda") * 0.4,
        "w13_input_scale_quant": torch.tensor(0.7, device="cuda"),
        "w2_input_scale_quant": torch.tensor(1.1, device="cuda"),
    }
    for name, shape in (
        ("w13_blockscale_swizzled", (e, 2 * i, h // 16)),
        ("w2_blockscale_swizzled", (e, h, i // 16)),
    ):
        tensors[name] = torch.randint(
            48, 64, shape, device="cuda", dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
    for tensor in tensors.values():
        if tensor.ndim:
            tensor[0].zero_()
    return tensors


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires SM12x B12X",
)
class TestNativeVector(CustomTestCase):
    def test_native_vector_parity(self):
        tensors = native_tensors()
        adapters = {
            bits: CachedB12xW4A16(
                torch.device("cuda"),
                2560,
                640,
                4,
                enable_fp32=True,
                native_load_bits=bits,
            )
            for bits in (0, 64, 128)
        }
        for m in (1, 2, 3):
            x = (torch.randn(m, 2560, device="cuda") * 0.002).bfloat16()
            ids = torch.randint(1, 4, (m, 10), device="cuda", dtype=torch.int64)
            ids[0, -2:] = torch.tensor([-1, 4], device="cuda")
            router = torch.rand(m, 10, device="cuda")
            for dtype in (torch.bfloat16, torch.float32):
                expected = adapters[0].run(x, ids, router, tensors, output_dtype=dtype)
                self.assertTrue(torch.isfinite(expected).all().item())
                self.assertGreater(expected.abs().max().item(), 0)
                for bits in (64, 128):
                    with self.subTest(m=m, bits=bits, dtype=dtype):
                        actual = adapters[bits].run(
                            x, ids, router, tensors, output_dtype=dtype
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            plan = torch.tensor([0, 0, 1, 0], device="cuda", dtype=torch.int32)
            completed = torch.ones(1, device="cuda", dtype=torch.int32)
            failed = torch.zeros(1, device="cuda", dtype=torch.int32)
            cpu_result = torch.randn(3, 2560, device="cuda") * 0.001
            hybrid = (
                plan,
                completed.data_ptr(),
                cpu_result.data_ptr(),
                failed.data_ptr(),
                1,
            )
            expected = adapters[0].run(x, ids, router, tensors, hybrid=hybrid)
            for bits in (64, 128):
                actual = adapters[bits].run(x, ids, router, tensors, hybrid=hybrid)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Replay uses live routes and freshly updated native alpha values.
        for bits in (64, 128):
            adapter = adapters[bits]
            adapter.done = torch.cuda.Event()
            adapter.used = False
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = adapter.run(x, ids, router, tensors)
            for route in (-1, 3):
                ids.fill_(route)
                tensors["g1_alphas"][3].mul_(1.01)
                expected = adapters[0].run(x, ids, router, tensors)
                graph.replay()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
