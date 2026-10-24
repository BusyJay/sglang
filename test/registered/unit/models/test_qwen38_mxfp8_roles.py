"""Keep delegate fallback opaque while marking opted-in native MXFP8 roles."""

import types
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedModelOptConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, stage="base-a", runner_config="cpu")


class TestMxfp8Roles(CustomTestCase):
    def wrapper(self, method):
        def unexpected_resolve(prefix):
            raise AssertionError("returned quant method must not be resolved again")

        delegate = types.SimpleNamespace(
            packed_modules_mapping={},
            get_quant_method=lambda layer, prefix: method,
            resolve_quant_algo=unexpected_resolve,
        )
        return Qwen38CachedModelOptConfig(delegate, None, gdn_value_heads=2)

    def test_opt_in_roles_preserve_quantized_delegate_without_second_resolution(self):
        for enabled in (False, True):
            for native_mxfp8 in (False, True):
                with self.subTest(enabled=enabled, native_mxfp8=native_mxfp8):
                    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
                    method.use_mxfp8 = native_mxfp8
                    wrapper = self.wrapper(method)
                    layer = torch.nn.Module()
                    layer.params_dtype = torch.bfloat16
                    with envs.SGLANG_OPT_QWEN38_MXFP8_SMALL_M.override(enabled):
                        result = wrapper.get_quant_method(
                            layer, "model.layers.0.linear_attn.in_proj_qkvz"
                        )
                    self.assertIs(result, method)
                    self.assertEqual(
                        getattr(layer, "_qwen38_mxfp8_role", None),
                        "gdn_qkvz" if enabled and native_mxfp8 else None,
                    )
                    self.assertEqual(wrapper._pending_gdn_weights, {})

    def test_non_fp8_method_and_unrelated_role_keep_fallback_opaque(self):
        native = Fp8LinearMethod.__new__(Fp8LinearMethod)
        native.use_mxfp8 = True
        for method, prefix in (
            (object(), "model.layers.0.linear_attn.in_proj_qkvz"),
            (None, "model.layers.0.mlp.shared_expert.down_proj"),
            (native, "model.layers.0.linear_attn.conv1d"),
        ):
            for enabled in (False, True):
                with self.subTest(method=method, prefix=prefix, enabled=enabled):
                    layer = torch.nn.Module()
                    layer.params_dtype = torch.bfloat16
                    with envs.SGLANG_OPT_QWEN38_MXFP8_SMALL_M.override(enabled):
                        result = self.wrapper(method).get_quant_method(layer, prefix)
                    self.assertIs(result, method)
                    self.assertFalse(hasattr(layer, "_qwen38_mxfp8_role"))


if __name__ == "__main__":
    unittest.main()
