"""MTP CUTLASS routing and graph regressions at real expert dimensions."""

import unittest
from types import SimpleNamespace

import torch


class TestMtpW4a4Gpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 required")
        from sglang.test.layer_ut_utils import init_single_process_dist

        init_single_process_dist()

    def test_routes_match_independent_cpu_reference(self):
        from sglang.srt.layers.quantization.qwen38_fp4_mtp import _prepare_mtp_routes

        torch.manual_seed(18)
        ids = torch.randint(-2, 34, (257, 10), dtype=torch.int32)
        ids[:, 1] = ids[:, 0]
        ids[0] = 3
        ids[1] = -1
        weights = torch.rand(257, 10)
        ratio = torch.rand(32) + 0.1
        expected_ids = torch.full_like(ids, -1)
        expected_weights = torch.zeros_like(weights)
        for token in range(len(ids)):
            seen = set()
            for slot, expert in enumerate(ids[token].tolist()):
                if 0 <= expert < 32 and expert not in seen:
                    seen.add(expert)
                    expected_ids[token, slot] = expert
                    expected_weights[token, slot] = (
                        weights[token, ids[token] == expert].sum() * ratio[expert]
                    )
        for device in ("cpu", "cuda"):
            actual_ids, actual_weights = _prepare_mtp_routes(
                ids.to(device), weights.to(device), ratio.to(device), 32
            )
            torch.testing.assert_close(actual_ids.cpu(), expected_ids)
            torch.testing.assert_close(actual_weights.cpu(), expected_weights)

    def test_duplicate_invalid_routes_and_graph_updates(self):
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization.qwen38_fp4_mtp import Qwen38Fp4MtpMethod

        torch.manual_seed(103)
        e, h, n = 32, 2560, 640
        layer = SimpleNamespace(num_local_experts=e)
        for prefix, rows, cols in (("w13", 2 * n, h), ("w2", h, n)):
            setattr(
                layer,
                prefix + "_weight",
                torch.randint(
                    256, (e, rows, cols // 2), device="cuda", dtype=torch.uint8
                ),
            )
            setattr(
                layer,
                prefix + "_weight_scale",
                (torch.rand((e, rows, cols // 16), device="cuda") * 0.5 + 0.25).to(
                    torch.float8_e4m3fn
                ),
            )
        layer.w13_weight_scale_2 = torch.tensor([[0.03, 0.05]] * e, device="cuda")
        layer.w2_weight_scale_2 = torch.full((e,), 0.03, device="cuda")
        method = Qwen38Fp4MtpMethod.__new__(Qwen38Fp4MtpMethod)
        method.activation_mode = "w4a4"
        method.create_moe_runner(
            layer,
            MoeRunnerConfig(
                num_experts=e,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size_per_partition=n,
                top_k=10,
                params_dtype=torch.bfloat16,
            ),
        )
        from sglang.srt.layers.quantization.utils import swizzle_blockscale

        pointers = [layer.w13_weight_scale.data_ptr(), layer.w2_weight_scale.data_ptr()]
        expected13 = swizzle_blockscale(
            torch.cat((layer.w13_weight_scale[:, n:], layer.w13_weight_scale[:, :n]), 1)
        )
        expected2 = swizzle_blockscale(layer.w2_weight_scale)
        method.process_weights_after_loading(layer)
        torch.testing.assert_close(
            layer.w13_blockscale_swizzled.float(), expected13.float(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            layer.w2_blockscale_swizzled.float(), expected2.float(), atol=0, rtol=0
        )
        self.assertEqual(layer.w13_blockscale_swizzled.data_ptr(), pointers[0])
        self.assertEqual(layer.w2_blockscale_swizzled.data_ptr(), pointers[1])
        for tokens in (1, 3, 128, 257):
            x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.2
            ids = torch.tensor(
                [[0, 0, 1, -1, e, 2, 2, 3, -2, 3]] * tokens,
                device="cuda",
                dtype=torch.int32,
            )
            weights = torch.rand(tokens, 10, device="cuda") / 10

            def run():
                return method.apply(
                    layer,
                    StandardDispatchOutput(
                        x, None, StandardTopKOutput(weights, ids, None)
                    ),
                ).hidden_states

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for step in range(3):
                x.mul_(0.75)
                weights.mul_(0.9)
                ids.copy_(ids.roll(1, 1))
                graph.replay()
                eager = run()
                torch.testing.assert_close(captured, eager, atol=0.001, rtol=0.01)
            ids.fill_(-1)
            graph.replay()
            self.assertEqual(captured.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
