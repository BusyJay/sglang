"""MTP A4 scale/layout and dispatch contracts using image PyTorch on CPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch


class TestMtpW4a4(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.quantization.qwen38_fp4_mtp import Qwen38Fp4MtpMethod

        self.cls = Qwen38Fp4MtpMethod
        self.method = self.cls.__new__(self.cls)
        self.method.activation_mode = "w4a4"
        self.layer = SimpleNamespace(
            num_local_experts=2,
            w13_weight=torch.nn.Parameter(
                torch.cat(
                    (
                        torch.full((2, 128, 64), 17, dtype=torch.uint8),
                        torch.full((2, 128, 64), 34, dtype=torch.uint8),
                    ),
                    1,
                ),
                requires_grad=False,
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.cat(
                    (torch.full((2, 128, 8), 0.5), torch.full((2, 128, 8), 1.0)), 1
                ).to(torch.float8_e4m3fn),
                requires_grad=False,
            ),
            w2_weight=torch.zeros((2, 128, 64), dtype=torch.uint8),
            w2_weight_scale=torch.ones((2, 128, 8)).to(torch.float8_e4m3fn),
            w13_weight_scale_2=torch.tensor([[0.03, 0.05], [0.08, 0.02]]),
            w2_weight_scale_2=torch.tensor([0.04, 0.06]),
        )

    def test_mode_validation_and_rollback(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        with mock.patch.object(
            ModelOptNvFp4FusedMoEMethod, "__init__", return_value=None
        ):
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(self.cls(None).activation_mode, "w4a4")
            with mock.patch.dict(
                "os.environ", {"SGLANG_QWEN38_MTP_ACTIVATION": "w4a16"}
            ):
                self.assertEqual(self.cls(None).activation_mode, "w4a16")
            with mock.patch.dict(
                "os.environ", {"SGLANG_QWEN38_MTP_ACTIVATION": "w4a4"}
            ):
                self.assertEqual(self.cls(None).activation_mode, "w4a4")
            with mock.patch.dict("os.environ", {"SGLANG_QWEN38_MTP_ACTIVATION": "a8"}):
                with self.assertRaisesRegex(ValueError, "w4a16 or w4a4"):
                    self.cls(None)

    def test_loading_order_and_scale_correction(self):
        w_pointer = self.layer.w13_weight.data_ptr()
        scale_pointers = [
            self.layer.w13_weight_scale.data_ptr(),
            self.layer.w2_weight_scale.data_ptr(),
        ]
        checkpoint_scales = self.layer.w13_weight_scale.float().clone()
        expected_scales = torch.cat(
            (checkpoint_scales[:, 128:], checkpoint_scales[:, :128]), 1
        )
        self.assertFalse(self.method.load_up_proj_weight_first)
        self.method.process_weights_after_loading(self.layer)
        self.assertEqual(w_pointer, self.layer.w13_weight.data_ptr())
        self.assertTrue((self.layer.w13_weight[:, :128] == 34).all())
        self.assertTrue((self.layer.w13_weight[:, 128:] == 17).all())
        # Invert the existing swizzle to prove both halves and every scale survive.
        recovered = self.layer.w13_blockscale_swizzled.reshape(2, 2, 2, 32, 4, 4)
        recovered = recovered.permute(0, 1, 4, 3, 2, 5).reshape(2, 256, 8)
        torch.testing.assert_close(recovered.float(), expected_scales)
        self.assertEqual(
            self.layer.w13_blockscale_swizzled.data_ptr(), scale_pointers[0]
        )
        self.assertEqual(
            self.layer.w2_blockscale_swizzled.data_ptr(), scale_pointers[1]
        )
        self.assertEqual(self.layer.w13_weight_scale.data_ptr(), scale_pointers[0])
        self.assertEqual(self.layer.w2_weight_scale.data_ptr(), scale_pointers[1])
        torch.manual_seed(7)
        gate, up = torch.randn(2, 13), torch.randn(2, 13)
        g = self.layer.w13_weight_scale_2[:, :1]
        u = self.layer.w13_weight_scale_2[:, 1:]
        expected = torch.nn.functional.silu(gate * g) * up * u
        corrected = torch.nn.functional.silu(gate * g) * up * g
        corrected *= self.layer.mtp_up_scale_ratio[:, None]
        torch.testing.assert_close(corrected, expected)
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            self.method.process_weights_after_loading(self.layer)

    def test_invalid_global_scale_rejected(self):
        self.layer.w13_weight_scale_2[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.method.process_weights_after_loading(self.layer)

    def test_a16_rollback_leaves_checkpoint_order(self):
        self.method.activation_mode = "w4a16"
        self.method.process_weights_after_loading(self.layer)
        self.assertTrue((self.layer.w13_weight[:, :128] == 17).all())
        self.assertFalse(hasattr(self.layer, "mtp_up_scale_ratio"))

    def test_dispatch_padding_duplicates_and_bounded_prefill(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        self.method.process_weights_after_loading(self.layer)
        calls = []

        def run(dispatch, quant):
            calls.append((dispatch, quant))
            return StandardCombineInput(hidden_states=dispatch.hidden_states.clone())

        self.method.runner = SimpleNamespace(run=run)
        x = torch.ones((257, 128), dtype=torch.bfloat16)
        ids = torch.tensor([[0, 0, 1, -1, 2]] * 257)
        weights = torch.ones_like(ids, dtype=torch.float32)
        topk = StandardTopKOutput(weights, ids, torch.zeros(257, 2))
        result = self.method.apply(self.layer, StandardDispatchOutput(x, None, topk))
        torch.testing.assert_close(result.hidden_states, x)
        self.assertEqual([d.hidden_states.shape[0] for d, _ in calls], [128, 128, 1])
        torch.testing.assert_close(
            calls[0][0].topk_output.topk_weights[0],
            torch.tensor([10 / 3, 0, 1 / 4, 0, 0]),
        )
        torch.testing.assert_close(
            calls[0][0].topk_output.topk_ids[0],
            torch.tensor([0, -1, 1, -1, -1], dtype=torch.int32),
        )
        self.assertEqual(calls[0][1].quant_type, "fp4")
        calls.clear()
        empty = StandardTopKOutput(weights[:0], ids[:0], torch.zeros(0, 2))
        self.method.apply(self.layer, StandardDispatchOutput(x[:0], None, empty))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
