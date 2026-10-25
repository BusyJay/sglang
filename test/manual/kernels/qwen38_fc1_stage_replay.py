"""Native phase parity, 256->128->256 rebinding, >8-key eviction on SM120.

Tests phase/graph correctness with resident synthetic weights. Full mailbox
transport races and performance still require the fixed-payload service replay.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.test.test_utils import CustomTestCase
from tools.qwen38_cache_tests.w4a16_verify_gpu_check import swizzle


class NativeStageReplay(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest("SM120/SM121 required")
        torch.manual_seed(20261025)
        cls.tensors = {}
        for prefix, rows, cols in (("w13", 1280, 2560), ("w2", 2560, 640)):
            cls.tensors[prefix + "_weight"] = torch.randint(
                0, 256, (256, rows, cols // 2), dtype=torch.uint8, device="cuda"
            )
            scales = torch.full((256, rows, cols // 16), 0.125, device="cuda").to(
                torch.float8_e4m3fn
            )
            cls.tensors[prefix + "_blockscale_swizzled"] = swizzle(scales.cpu())
            cls.tensors[prefix + "_input_scale_quant"] = torch.ones((), device="cuda")
        cls.tensors["g1_alphas"] = torch.full((256,), 0.02, device="cuda")
        cls.tensors["g2_alphas"] = torch.full((256,), 0.03, device="cuda")

    def test_phase_rebinding_eviction_and_route_masks(self):
        previous_namespaces = None
        for split in (False, True):
            with envs.SGLANG_OPT_QWEN38_SPLIT_FILL.override(split):
                adapter = CachedB12xW4A16(
                    torch.device("cuda"),
                    2560,
                    640,
                    256,
                    enable_fp32=True,
                    enable_fc1_overlap=True,
                )
                adapter.precompile_fc1(10, 128, 256)
                namespaces = (adapter.fc1_namespace, adapter.fc2_namespace)
                if previous_namespaces is not None:
                    for previous, current in zip(previous_namespaces, namespaces):
                        self.assertIs(previous, current)
                previous_namespaces = namespaces
                compiled_sizes = [
                    len(ns._SMALL_M_DIRECT_CACHE)
                    for ns in (adapter.fc1_namespace, adapter.fc2_namespace)
                ]
                for index, capacity in enumerate(
                    (256, 128, 256, 129, 130, 131, 132, 133, 134, 135, 128)
                ):
                    tensors = {
                        name: value[:capacity] if value.ndim else value
                        for name, value in self.tensors.items()
                    }
                    # Mailbox header plus source/destination for up to 30 routes.
                    plan = torch.zeros(4 + 2 * 30, dtype=torch.int32, device="cuda")
                    resident = (plan, tensors["g1_alphas"], tensors["g2_alphas"])
                    for tokens in (1, 2, 3):
                        x = (
                            torch.randn(
                                tokens, 2560, device="cuda", dtype=torch.bfloat16
                            )
                            * 0.1
                        )
                        ids = torch.arange(
                            tokens * 10, device="cuda", dtype=torch.int32
                        ).reshape(tokens, 10)
                        if index % 2:
                            ids[:, 1] = ids[:, 0]
                            ids[:, -1] = -1
                        weights = torch.full((tokens, 10), 0.1, device="cuda")
                        for hits in (0, 5, 10):
                            stage = adapter.fc1_stage(
                                x, ids, weights, tensors, resident_alphas=resident
                            )
                            stage.remapped.copy_(ids)
                            stage.early.copy_(ids)
                            stage.early[:, hits:] = -1
                            stage.begin(x)
                            actual = stage.finish().clone()
                            reference = adapter.run(
                                x, ids, weights, tensors, output_dtype=torch.float32
                            )
                            torch.testing.assert_close(
                                actual, reference, rtol=0, atol=0
                            )
                self.assertLessEqual(len(adapter.fc1_stages), 8)
                self.assertEqual(
                    compiled_sizes,
                    [
                        len(ns._SMALL_M_DIRECT_CACHE)
                        for ns in (adapter.fc1_namespace, adapter.fc2_namespace)
                    ],
                )
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
