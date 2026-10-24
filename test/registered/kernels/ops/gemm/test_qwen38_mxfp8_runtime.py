"""Exercise the model-owned role marker and FP8 linear candidate dispatch."""

import types
import unittest

import torch

from sglang.kernels.ops.gemm.qwen38_mxfp8_small_m import qwen38_mxfp8_small_m
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_blockscaled_linear
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedModelOptConfig
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestMxfp8Runtime(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")

    def tearDown(self):
        reset_context()

    def test_bf16_and_prequantized_inputs_use_candidate_and_preserve_fallback(self):
        import flashinfer

        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.use_mxfp8 = True
        method.use_marlin = False
        method.block_fp8_as_mxfp8 = False
        method.mxfp8_prefill_autotune_min_tokens = None
        method.mxfp8_dense_backend = types.SimpleNamespace(
            is_flashinfer_cutlass=lambda: True,
            is_flashinfer_cutedsl=lambda: False,
        )
        method.w8a8_mxfp8_linear = flashinfer_mxfp8_blockscaled_linear
        delegate = types.SimpleNamespace(
            packed_modules_mapping={}, get_quant_method=lambda *_: method
        )
        wrapper = Qwen38CachedModelOptConfig(delegate, None)
        layer = torch.nn.Module()
        torch.manual_seed(42)
        weight = torch.randn(2560, 640, device="cuda", dtype=torch.bfloat16) * 0.02
        qw, sw = flashinfer.mxfp8_quantize(weight, True)
        layer.weight = torch.nn.Parameter(qw, requires_grad=False)
        layer.register_buffer("weight_scale_inv_swizzled", sw)
        with envs.SGLANG_OPT_QWEN38_MXFP8_SMALL_M.override(True):
            self.assertIs(
                wrapper.get_quant_method(
                    layer, "model.layers.0.mlp.shared_expert.down_proj"
                ),
                method,
            )
            for rows in (1, 3):
                x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
                q, scales = flashinfer.mxfp8_quantize(x, True, alignment=32)
                expected = qwen38_mxfp8_small_m(q, qw, scales, sw)
                # Removing the actual marker must be observable: the delegate
                # below fails if an intended candidate silently falls back.
                original = method.w8a8_mxfp8_linear

                def unexpected_fallback(**kwargs):
                    raise AssertionError("marked candidate silently fell back")

                method.w8a8_mxfp8_linear = unexpected_fallback
                try:
                    dense = method.apply(layer, x)
                    quantized = method.apply(layer, Mxfp8SwizzledInput(q, scales))
                finally:
                    method.w8a8_mxfp8_linear = original
                torch.testing.assert_close(dense, expected, rtol=0, atol=0)
                torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
                with envs.SGLANG_OPT_QWEN38_MXFP8_SMALL_M.override(False):
                    fallback = method.apply(layer, x)
                baseline = flashinfer_mxfp8_blockscaled_linear(x, qw, sw)
                torch.testing.assert_close(fallback, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
