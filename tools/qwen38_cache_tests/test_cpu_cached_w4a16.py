"""Cache ABI byte layout, scale semantics and W4A16 option wiring."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.qwen38_cached_w4a16 import validate_cache_tensors
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedNvFp4MoeMethod
from sglang.srt.layers.quantization.utils import swizzle_blockscale


class TestCachedW4A16(unittest.TestCase):
    def test_swizzled_address_contract_all_rows_and_columns(self):
        # Independently verify the Triton address expression against the
        # existing exporter swizzle, including M/K padding and many experts.
        for e, rows, cols in [(3, 1280, 160), (2, 2560, 40), (1, 129, 5)]:
            raw = (
                torch.arange(e * rows * cols)
                .remainder(256)
                .to(torch.uint8)
                .reshape(e, rows, cols)
            )
            packed = swizzle_blockscale(raw.view(torch.float8_e4m3fn)).view(torch.uint8)
            r = torch.arange(rows)[:, None]
            c = torch.arange(cols)[None, :]
            padded_cols = (cols + 3) // 4 * 4
            offsets = (
                ((r // 128 * (padded_cols // 4) + c // 4) * 32 + r % 32) * 4
                + (r % 128) // 32
            ) * 4 + c % 4
            self.assertTrue(torch.equal(packed.flatten(1)[:, offsets], raw))

    def test_mode_defaults_off_explicit_on_and_invalid_values(self):
        manifest = SimpleNamespace()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Qwen38CachedNvFp4MoeMethod(manifest, "p").use_w4a16)
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_CACHE_W4A16": "1"}):
            self.assertTrue(Qwen38CachedNvFp4MoeMethod(manifest, "p").use_w4a16)
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_CACHE_W4A16": "0"}):
            self.assertFalse(Qwen38CachedNvFp4MoeMethod(manifest, "p").use_w4a16)
        for key, value in [
            ("SGLANG_QWEN38_CACHE_W4A16", "auto"),
            ("SGLANG_QWEN38_CACHE_W4A16_CHUNK_TOKENS", "0"),
            ("SGLANG_QWEN38_CACHE_W4A16_CHUNK_TOKENS", "65"),
        ]:
            with (
                mock.patch.dict(os.environ, {key: value}),
                self.assertRaises(ValueError),
            ):
                Qwen38CachedNvFp4MoeMethod(manifest, "p")

    def test_explicit_phase_modes_include_short_prefill_and_verify(self):
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_CACHE_W4A16": "1",
                                         "SGLANG_QWEN38_CACHE_PREFILL_W4A16": "0"}):
            method = Qwen38CachedNvFp4MoeMethod(SimpleNamespace(), "p")
        self.assertFalse(method.uses_w4a16("prefill"))
        for phase in ("decode", "verify", None):
            self.assertTrue(method.uses_w4a16(phase))
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_CACHE_W4A16": "0",
                                         "SGLANG_QWEN38_CACHE_PREFILL_W4A16": "1"}):
            method = Qwen38CachedNvFp4MoeMethod(SimpleNamespace(), "p")
        self.assertTrue(method.uses_w4a16("prefill"))
        self.assertFalse(method.uses_w4a16("decode"))
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_CACHE_PREFILL_W4A16": "auto"}):
            with self.assertRaises(ValueError):
                Qwen38CachedNvFp4MoeMethod(SimpleNamespace(), "p")

    def test_alpha_recovery_and_cache_shape_rejection(self):
        # CUTLASS alphas = weight global * raw activation global.
        # A16 alpha = CUTLASS alpha * reciprocal activation global.
        weights = torch.tensor([0.01, 0.07, 0.002])
        for activation in (0.1, 2.0, 10.0):
            self.assertTrue(
                torch.allclose(weights * activation * (1 / activation), weights)
            )
        e, h, n = 2, 2560, 640
        tensors = {
            "w13_weight": torch.empty(
                e, 2 * n, h // 2, dtype=torch.uint8, device="meta"
            ),
            "w2_weight": torch.empty(e, h, n // 2, dtype=torch.uint8, device="meta"),
            "w13_blockscale_swizzled": torch.empty(
                e, 2 * n, h // 16, dtype=torch.float8_e4m3fn, device="meta"
            ),
            "w2_blockscale_swizzled": torch.empty(
                e, h, n // 16, dtype=torch.float8_e4m3fn, device="meta"
            ),
            "g1_alphas": torch.empty(e, device="meta"),
            "g2_alphas": torch.empty(e, device="meta"),
            "w13_input_scale_quant": torch.empty((), device="meta"),
            "w2_input_scale_quant": torch.empty((), device="meta"),
        }
        validate_cache_tensors(tensors, h, n, e)
        tensors["w13_weight"] = torch.empty(
            e, 2 * n, h, dtype=torch.uint8, device="meta"
        )
        with self.assertRaises(ValueError):
            validate_cache_tensors(tensors, h, n, e)

    def test_rejects_unsupported_activation_configuration(self):
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
            validate_runner_config,
        )

        config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            gate_up_interleaved=False,
        )
        validate_runner_config(config)
        for field, value in [
            ("activation", "gelu"),
            ("is_gated", False),
            ("swiglu_limit", 7.0),
            ("gemm1_alpha", 1.7),
            ("no_combine", True),
        ]:
            with (
                mock.patch.object(config, field, value, create=True),
                self.assertRaises(ValueError),
            ):
                validate_runner_config(config)
