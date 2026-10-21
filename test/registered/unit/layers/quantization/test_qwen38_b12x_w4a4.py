# SPDX-License-Identifier: Apache-2.0
"""Common B12X operand layout and bounded prefill launch bindings."""

import importlib.util
import os
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import CachedB12xW4A4
from sglang.srt.model_loader.qwen38_nvfp4_pack.csf import (
    encode,
    native_rows,
    validate_bitmap,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, stage="base-a", runner_config="cpu")


@unittest.skipUnless(
    importlib.util.find_spec("b12x") is not None, "requires pinned B12X"
)
class TestB12xBitmap(CustomTestCase):
    def test_matches_vendor_inline_layout(self):
        from b12x._lib.quant.nvfp4_csf_inline import prepare_inline_scales

        rows, cols = 1280, 160
        rng = np.random.default_rng(82)
        raw = rng.integers(30, 46, rows * cols, dtype=np.uint8)
        raw[::43] = 81
        slab = encode(raw.tobytes(), rows, cols)
        fixed_bytes = rows * (1 + cols // 2)
        # Reconstruct the vendor's source-format inputs from row bases and
        # source scales, not through either inference decoder.
        fixed = np.frombuffer(slab, np.uint8, fixed_bytes, 1024).copy()
        lane = np.arange(128)
        row_order = lane // 4 + lane % 4 * 32
        bases = np.empty(rows, np.uint8)
        for r in range(0, rows, 128):
            bases[r + row_order] = fixed[
                r // 128 * (128 * (1 + cols // 2)) : r // 128 * (128 * (1 + cols // 2))
                + 128
            ]
        delta = raw.astype(np.int16) - bases[native_rows(rows, cols)].astype(np.int16)
        positions = np.flatnonzero((delta < 0) | (delta > 15)).astype(np.uint32)
        native_row = native_rows(rows, cols)[positions]
        col = (positions // 512 % (cols // 4)) * 4 + positions % 4
        logical = native_row * cols + col
        exceptions = logical | (raw[positions].astype(np.uint32) << 24)
        batch = SimpleNamespace(
            validate=lambda: None,
            codec=0,
            layout=0,
            num_experts=1,
            rows=rows,
            columns=cols,
            fixed=torch.from_numpy(fixed),
            exceptions=torch.from_numpy(exceptions),
            task_offsets=torch.tensor([[0, len(exceptions)]], dtype=torch.int64),
        )
        vendor = prepare_inline_scales(batch).storage.numpy().tobytes()
        self.assertEqual(slab[1024:], vendor[1024:])
        # Both headers expose an identical zero operand at the staged extent.
        self.assertEqual(slab[:704], vendor[:704])
        broken = bytearray(slab)
        struct.pack_into("<I", broken, 1024 + fixed_bytes + 4, 0xFFFFFFFF)
        with self.assertRaisesRegex(ValueError, "bitmap/rank"):
            validate_bitmap(broken, rows, cols)

    def test_compressed_pack_accepts_w4a4_prefill_and_retains_a16_decode(self):
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        manifest = SimpleNamespace(
            scale_encoding={
                "w13_blockscale_swizzled": dict(encoding="b12x_csf_bitmap_v1")
            }
        )
        with patch.dict(
            os.environ,
            {
                "SGLANG_QWEN38_CACHE_W4A16": "1",
                "SGLANG_QWEN38_CACHE_W4A16_BACKEND": "b12x",
                "SGLANG_QWEN38_CACHE_PREFILL_W4A16": "0",
            },
            clear=True,
        ):
            method = Qwen38CachedNvFp4MoeMethod(manifest, "p")
            self.assertFalse(method.uses_w4a16("prefill"))
            self.assertTrue(method.uses_w4a16("decode"))
            self.assertTrue(method.uses_w4a16("verify"))

    def test_prefill_chunking_and_capacity_rebinding(self):
        from b12x.moe.fused_moe import _impl

        from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import (
            dynamic_kernel_namespace,
        )

        # The real adapter allocates real bounded workspace on CPU. Only CUDA
        # device detection and compilation/launch cross the hardware boundary.
        namespace = dynamic_kernel_namespace({"_w13_rows": 1280})
        launches = []

        def compiled(*args):
            launches.append(args)

        replacements = dict(
            _get_dynamic_kernel=lambda *a, **kw: (compiled, 1),
            current_cuda_stream=lambda: _impl.cuda.CUstream(0),
            _get_impl_mac=lambda *a, **kw: 1,
            _dynamic_external_route_plan_supported=lambda **kw: False,
        )
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch(
                "sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4.dynamic_kernel_namespace",
                return_value=namespace,
            ),
            patch.dict(namespace._launch_dynamic_flat.__globals__, replacements),
            patch.object(
                namespace, "current_cuda_stream", replacements["current_cuda_stream"]
            ),
        ):
            a = CachedB12xW4A4(
                torch.device("cpu"),
                2560,
                640,
                7,
                3,
                scale_encoding={
                    "w13_blockscale_swizzled": dict(
                        encoding="b12x_csf_bitmap_v1",
                        rows=1280,
                        columns=160,
                        stride=131072,
                    )
                },
            )
            x = torch.randn(129, 2560).bfloat16()
            ids = torch.zeros(129, 3, dtype=torch.int32)
            router = torch.ones(129, 3)
            previous_out = None
            for e in (7, 7, 4):
                tensors = {
                    "w13_weight": torch.zeros(e, 1280, 1280, dtype=torch.uint8),
                    "w2_weight": torch.zeros(e, 2560, 320, dtype=torch.uint8),
                    "w13_blockscale_swizzled": torch.zeros(
                        e, 131072, dtype=torch.uint8
                    ),
                    "w2_blockscale_swizzled": torch.zeros(
                        e, 2560, 40, dtype=torch.uint8
                    ),
                    "g1_alphas": torch.full((e,), 0.8),
                    "g2_alphas": torch.full((e,), 1.2),
                    "w13_input_scale_quant": torch.tensor(0.7),
                    "w2_input_scale_quant": torch.tensor(1.1),
                }
                out = a.run(x, ids, router, tensors)
                self.assertEqual(out.shape, x.shape)
                first, last = launches[-2:]
                # The actual 45-argument B12X ABI retains live token counts and
                # binds views to the current pool, without copying weights.
                self.assertEqual((first[-8], last[-8]), (128, 1))
                self.assertEqual(first[22].data_ptr(), tensors["w13_weight"].data_ptr())
                self.assertEqual(first[27].shape, (e,))
                torch.testing.assert_close(first[31], tensors["g1_alphas"])
                torch.testing.assert_close(first[32], tensors["g2_alphas"])
                torch.testing.assert_close(first[30], torch.full((e,), 0.7))
                torch.testing.assert_close(first[33], torch.full((e,), 1.1))
                fixed = first[3]
                bound_weight = first[22]
                x2, ids2, router2 = x.clone(), ids.clone(), router.clone()
                tensors["g1_alphas"].fill_(2.3)
                tensors["w13_input_scale_quant"].fill_(0.9)
                out2 = a.run(x2, ids2, router2, tensors)
                repeat = launches[-2]
                self.assertIs(repeat[3], fixed)
                self.assertEqual(repeat[22].data_ptr(), bound_weight.data_ptr())
                self.assertEqual(repeat[0]._pointer, x2.data_ptr())
                self.assertEqual(repeat[1]._pointer, ids2.data_ptr())
                self.assertEqual(repeat[2]._pointer, router2.data_ptr())
                self.assertEqual(repeat[34]._pointer, out2.data_ptr())
                self.assertNotEqual(out2.data_ptr(), out.data_ptr())
                if previous_out is not None:
                    self.assertNotEqual(out.data_ptr(), previous_out.data_ptr())
                previous_out = out2
                torch.testing.assert_close(repeat[31], torch.full((e,), 2.3))
                torch.testing.assert_close(repeat[30], torch.full((e,), 0.9))
                # Compare the complete prepared ABI with the vendor launch.
                live = dict(
                    a.values,
                    a=x2[:128],
                    flat_ids=ids2[:128].reshape(-1),
                    flat_weights=router2[:128].reshape(-1),
                    scatter_output=out2[:128],
                    w13_fp4=tensors["w13_weight"].permute(1, 2, 0),
                    down_fp4=tensors["w2_weight"].permute(1, 2, 0),
                    w13_sf=tensors["w13_blockscale_swizzled"],
                    down_sf=tensors["w2_blockscale_swizzled"],
                    m=128,
                    routed_rows=384,
                    topk_ids_are_i32=True,
                )
                namespace._launch_dynamic_flat(**live)
                vendor = launches[-1]
                for got, want in zip(repeat, vendor):
                    if isinstance(got, torch.Tensor):
                        self.assertEqual(got.data_ptr(), want.data_ptr())
                        self.assertEqual(got.shape, want.shape)
                        self.assertEqual(got.stride(), want.stride())
                    elif hasattr(got, "_pointer"):
                        self.assertEqual(got._pointer, want._pointer)
                    else:
                        self.assertEqual(int(got), int(want))
                # Reused input addresses keep pointer descriptors, while the
                # output still belongs to this invocation. Int64 IDs rebind.
                a.run(x2, ids2, router2, tensors)
                same = launches[-2]
                self.assertIs(same[0], repeat[0])
                self.assertIs(same[1], repeat[1])
                self.assertIs(same[2], repeat[2])
                ids64 = ids2.long()
                a.run(x2, ids64, router2, tensors)
                self.assertEqual(launches[-2][1]._pointer, ids64.data_ptr())
                # No cached tensor may retain expert or activation storage.
                borrowed = {
                    v.untyped_storage().data_ptr()
                    for v in (x2, ids2, ids64, router2, *tensors.values())
                }
                for value in a.values.values():
                    if isinstance(value, torch.Tensor):
                        self.assertNotIn(value.untyped_storage().data_ptr(), borrowed)
                for _, args in a.bindings.values():
                    for value in args:
                        if isinstance(value, torch.Tensor):
                            self.assertNotIn(
                                value.untyped_storage().data_ptr(), borrowed
                            )
            with self.assertRaisesRegex(ValueError, "BF16"):
                a.run(x.float(), ids, router, tensors)


if __name__ == "__main__":
    unittest.main()
