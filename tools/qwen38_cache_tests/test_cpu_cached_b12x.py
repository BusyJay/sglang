"""B12X dispatch contract and cache ownership without CUDA execution."""

import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedNvFp4MoeMethod


class TestCachedB12x(unittest.TestCase):
    def test_precompile_covers_every_capacity_in_legal_interval(self):
        from sglang.srt.layers.quantization.qwen38_cached_b12x import (
            reachable_capacity_buckets,
        )

        for minimum, maximum in ((10, 256), (128, 256), (129, 256), (107, 210), (1, 1)):
            buckets = reachable_capacity_buckets(minimum, maximum)
            observed = {
                1 << (capacity - 1).bit_length()
                for capacity in range(minimum, maximum + 1)
            }
            self.assertEqual(set(buckets), observed)
        with self.assertRaises(ValueError):
            reachable_capacity_buckets(0, 128)

    def test_switch_validation_and_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                Qwen38CachedNvFp4MoeMethod(None, "p").w4a16_backend, "triton"
            )
        for env in (
            {"SGLANG_QWEN38_CACHE_W4A16_BACKEND": "bad"},
            {"SGLANG_QWEN38_CACHE_W4A16_BACKEND": "b12x"},
        ):
            with (
                mock.patch.dict(os.environ, env, clear=True),
                self.assertRaises(ValueError),
            ):
                Qwen38CachedNvFp4MoeMethod(None, "p")
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_QWEN38_CACHE_W4A16_BACKEND": "b12x",
                "SGLANG_QWEN38_CACHE_W4A16": "1",
            },
            clear=True,
        ):
            self.assertEqual(
                Qwen38CachedNvFp4MoeMethod(None, "p").w4a16_backend, "b12x"
            )

    def test_phase_dtype_and_shape_bounds(self):
        for tokens in (0, 1, 2, 3, 4):
            for dtype in (torch.bfloat16, torch.float16):
                x = torch.empty(tokens, 128, dtype=dtype)
                ids = torch.empty(tokens, 10, dtype=torch.int32)
                for phase in ("prefill", "decode", "verify"):
                    self.assertEqual(
                        CachedB12xW4A16.eligible(x, ids, phase),
                        1 <= tokens <= 3
                        and dtype == torch.bfloat16
                        and phase != "prefill",
                    )
        self.assertFalse(
            CachedB12xW4A16.eligible(
                torch.empty(1, 128, dtype=torch.bfloat16), torch.empty(1, 33), "verify"
            )
        )

    def test_native_views_scale_refresh_capacity_and_stream_event(self):
        from b12x.moe._shared.kernels.w4a16.prepare import W4A16ModelOptWeights

        backend = CachedB12xW4A16.__new__(CachedB12xW4A16)
        backend.hidden, backend.width, backend.max_experts = 128, 32, 4
        backend.scale_encoding = {}
        backend.globals = torch.empty(2, 4)
        backend.global_views = OrderedDict()
        backend.workspace = torch.zeros(10, dtype=torch.int32)
        backend.intermediate = torch.empty(3 * 128 * 32 * 2, dtype=torch.bfloat16)
        backend.unused = torch.empty(0, dtype=torch.bfloat16)
        backend.done, backend.used = mock.Mock(), False
        backend.fp32_launch = backend.prepared_launch = None
        backend.weights_type = W4A16ModelOptWeights
        calls = []

        def launch(x, prepared, router, ids, **kwargs):
            calls.append(prepared)
            kwargs["output"].zero_()
            return kwargs["output"]

        backend.kernel = SimpleNamespace(
            _small_m_direct_supported=lambda **kw: True, run_w4a16_moe=launch
        )
        x = torch.empty(1, 128, dtype=torch.bfloat16)
        ids, router = torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1)
        stream = mock.Mock()
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch("torch.cuda.current_stream", return_value=stream),
        ):
            for capacity, alpha in ((2, 3.0), (4, 5.0), (2, 7.0)):
                tensors = {
                    "w13_weight": torch.empty(capacity, 64, 64, dtype=torch.uint8),
                    "w2_weight": torch.empty(capacity, 128, 16, dtype=torch.uint8),
                    "w13_blockscale_swizzled": torch.empty(
                        capacity, 128, 8, dtype=torch.float8_e4m3fn
                    ),
                    "w2_blockscale_swizzled": torch.empty(
                        capacity, 128, 4, dtype=torch.float8_e4m3fn
                    ),
                    "g1_alphas": torch.full((capacity,), alpha),
                    "g2_alphas": torch.full((capacity,), alpha * 2),
                    "w13_input_scale_quant": torch.tensor(0.5),
                    "w2_input_scale_quant": torch.tensor(0.25),
                }
                result = backend.run(x, ids, router, tensors)
                prepared = calls[-1]
                self.assertIs(prepared.w13, tensors["w13_weight"])
                self.assertIs(prepared.w2_scale, tensors["w2_blockscale_swizzled"])
                self.assertEqual(prepared.w13_layout, "w13")
                self.assertEqual(prepared.num_experts, capacity)
                torch.testing.assert_close(
                    prepared.w13_global_scale, torch.full((capacity,), alpha * 0.5)
                )
                torch.testing.assert_close(
                    prepared.w2_global_scale, torch.full((capacity,), alpha * 0.5)
                )
                self.assertEqual(result.shape, x.shape)
            self.assertEqual(stream.wait_event.call_count, 2)
            self.assertEqual(backend.done.record.call_count, 3)
        # Adapter retains only scratch, never a prepared object or pool tensor.
        self.assertFalse(any(value is calls[-1] for value in vars(backend).values()))
