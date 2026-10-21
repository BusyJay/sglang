# SPDX-License-Identifier: Apache-2.0
"""Compressed/native GPU parity, including GPU-only prefill and pool relocation."""

import unittest

import numpy as np
import torch
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.layers.quantization.qwen38_cached_w4a16 import cached_w4a16_moe
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import encode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def tensors():
    torch.manual_seed(19)
    experts, h, i = 4, 2560, 640
    native = {
        "w13_weight": torch.randint(
            0, 256, (experts, 2 * i, h // 2), dtype=torch.uint8, device="cuda"
        ),
        "w2_weight": torch.randint(
            0, 256, (experts, h, i // 2), dtype=torch.uint8, device="cuda"
        ),
        "g1_alphas": torch.full((experts,), 0.8, device="cuda"),
        "g2_alphas": torch.full((experts,), 1.2, device="cuda"),
        "w13_input_scale_quant": torch.tensor(0.7, device="cuda"),
        "w2_input_scale_quant": torch.tensor(1.1, device="cuda"),
    }
    compressed = dict(native)
    specs = {}
    rng = np.random.default_rng(37)
    for name, rows, cols in (
        ("w13_blockscale_swizzled", 2 * i, h // 16),
        ("w2_blockscale_swizzled", h, i // 16),
    ):
        slabs = []
        raw = rng.integers(20, 36, (experts, rows, cols), dtype=np.uint8)
        raw.reshape(-1)[::619] = 65
        native[name] = torch.from_numpy(raw).cuda().view(torch.float8_e4m3fn)
        for e in range(experts):
            slabs.append(encode(raw[e].tobytes(), rows, cols))
        stride = (max(map(len, slabs)) + 15) // 16 * 16
        compressed[name] = torch.from_numpy(
            np.stack(
                [np.frombuffer(s + bytes(stride - len(s)), np.uint8) for s in slabs]
            )
        ).cuda()
        specs[name] = dict(
            encoding="b12x_csf_bitmap_v1", rows=rows, columns=cols, stride=stride
        )
    # Zero slot is read by masked B12X routes, and must be a valid empty slab.
    for t in native.values():
        if t.ndim:
            t[0].zero_()
    for t in compressed.values():
        if t.ndim:
            t[0].zero_()
    return native, compressed, specs


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCsfGpu(CustomTestCase):
    def test_gpu_prefill_and_relocated_slots(self):
        native, compressed, _ = tensors()
        # Copy opaque CSF rows exactly as cache rebalancing does.
        order = torch.tensor([0, 3, 1, 2], device="cuda")
        native = {n: t[order].clone() if t.ndim else t for n, t in native.items()}
        compressed = {
            n: t[order].clone() if t.ndim else t for n, t in compressed.items()
        }
        for m in (1, 3, 17):
            x = (torch.randn(m, 2560, device="cuda") * 0.002).bfloat16()
            ids = torch.randint(1, 4, (m, 3), device="cuda", dtype=torch.int32)
            ids[0, -1] = -1
            router = torch.rand(m, 3, device="cuda")
            a = cached_w4a16_moe(x, ids, router, native)
            b = cached_w4a16_moe(x, ids, router, compressed)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
        "requires SM12x B12X",
    )
    def test_w4a4_prefill_shared_storage_masks_and_native_fallback(self):
        native, compressed, specs = tensors()
        rng = np.random.default_rng(95)
        # Scales and activations produce nonzero output; compare against the
        # native W4A4 kernel, including the same intermediate quantization.
        for name, rows, cols in (
            ("w13_blockscale_swizzled", 1280, 160),
            ("w2_blockscale_swizzled", 2560, 40),
        ):
            raw = rng.integers(48, 64, (4, rows, cols), dtype=np.uint8)
            raw.reshape(-1)[::61] = 75
            native[name] = torch.from_numpy(raw).cuda().view(torch.float8_e4m3fn)
            slabs = [encode(t.tobytes(), rows, cols) for t in raw]
            stride = (max(map(len, slabs)) + 15) // 16 * 16
            compressed[name] = torch.from_numpy(
                np.stack(
                    [np.frombuffer(t + bytes(stride - len(t)), np.uint8) for t in slabs]
                )
            ).cuda()
            specs[name] = dict(
                encoding="b12x_csf_bitmap_v1", rows=rows, columns=cols, stride=stride
            )
        for name in ("g1_alphas", "g2_alphas"):
            native[name].fill_(0.03)
        # BF16 atomic scatter varies even across identical native launches.
        # Use the vendor deterministic reduction to isolate scale decoding.
        a = CachedB12xW4A4(
            torch.device("cuda"),
            2560,
            640,
            4,
            3,
            scale_encoding={},
            deterministic_output=True,
        )
        for fallback in (None, "w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            active_specs = dict(specs)
            values = dict(compressed)
            if fallback:
                active_specs.pop(fallback)
                values[fallback] = native[fallback]
            b = CachedB12xW4A4(
                torch.device("cuda"),
                2560,
                640,
                4,
                3,
                scale_encoding=active_specs,
                deterministic_output=True,
            )
            # Reorder/relocate both representations exactly like an expert
            # pool migration, then shrink live views of the same storage.
            order = torch.tensor([3, 0, 2, 1], device="cuda")
            nn = {n: t[order].clone() if t.ndim else t for n, t in native.items()}
            cc = {n: t[order].clone() if t.ndim else t for n, t in values.items()}
            pointers = {n: t.data_ptr() for n, t in cc.items()}
            for capacity in (4, 3):
                nview = {n: t[:capacity] if t.ndim else t for n, t in nn.items()}
                cview = {n: t[:capacity] if t.ndim else t for n, t in cc.items()}
                for m in (1, 3, 17, 129):
                    x = (torch.randn(m, 2560, device="cuda") * 0.05).bfloat16()
                    ids = torch.randint(
                        0, capacity, (m, 3), device="cuda", dtype=torch.int32
                    )
                    ids[:, 1] = ids[:, 0]  # duplicate expert routes
                    ids[0] = -1  # all masked token
                    if m > 1:
                        ids[1, -1] = capacity  # upper bound mask
                    router = torch.rand(m, 3, device="cuda") / 3
                    expected = a.run(x, ids, router, nview)
                    got = b.run(x, ids, router, cview)
                    torch.testing.assert_close(got, expected, rtol=0.02, atol=0.0001)
                    self.assertEqual(got[0].count_nonzero().item(), 0)
                    if m > 1:
                        self.assertGreater(got[1:].abs().max().item(), 0)
                    self.assertEqual(pointers, {n: t.data_ptr() for n, t in cc.items()})
                    if m == 129:
                        # Warm binding with new live inputs on another stream.
                        # The previous output must remain independently owned.
                        saved = got.clone()
                        x2 = (x * 0.5).contiguous()
                        expected2 = a.run(x2, ids, router, nview)
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            got2 = b.run(x2, ids, router, cview)
                        torch.cuda.current_stream().wait_stream(side)
                        torch.testing.assert_close(
                            got2, expected2, rtol=0.02, atol=0.0001
                        )
                        torch.testing.assert_close(got, saved, rtol=0, atol=0)
                        self.assertNotEqual(got.data_ptr(), got2.data_ptr())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
        "requires SM12x B12X",
    )
    def test_b12x_bf16_and_fp32_inline_decode(self):
        native, compressed, specs = tensors()
        a = CachedB12xW4A16(torch.device("cuda"), 2560, 640, 4, enable_fp32=True)
        b = CachedB12xW4A16(
            torch.device("cuda"), 2560, 640, 4, enable_fp32=True, scale_encoding=specs
        )
        for m in (1, 2, 3):
            x = (torch.randn(m, 2560, device="cuda") * 0.002).bfloat16()
            ids = torch.randint(1, 4, (m, 3), device="cuda", dtype=torch.int32)
            ids[0, -1] = -1
            router = torch.rand(m, 3, device="cuda")
            for dtype in (torch.bfloat16, torch.float32):
                aa = a.run(x, ids, router, native, output_dtype=dtype)
                bb = b.run(x, ids, router, compressed, output_dtype=dtype)
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
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
            aa = a.run(x, ids, router, native, hybrid=hybrid)
            bb = b.run(x, ids, router, compressed, hybrid=hybrid)
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
