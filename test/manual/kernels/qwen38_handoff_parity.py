"""Handoff gate/cast numerical parity and graph-buffer isolation on SM120."""

import unittest

import torch

from sglang.kernels.ops.activation.softcap import cast_to_float32_logits
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
from sglang.kernels.ops.elementwise.hc_combine_decode import (
    hc_combine_apply,
    hc_combine_gate,
)
from sglang.kernels.ops.moe.shared_expert_gate import (
    shared_expert_gate,
    shared_expert_gate_add,
)
from sglang.test.test_utils import CustomTestCase


class HandoffParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest("SM120/SM121 required")
        torch.manual_seed(20261025)

    def test_logits_buffer_preserves_column_stride_fallback(self):
        from types import SimpleNamespace

        from sglang.srt.layers.logits_processor import LogitsProcessor

        for rows in (1, 3):
            for dtype in (torch.float16, torch.bfloat16):
                for input_stride, output_stride in ((2, 1), (1, 2), (2, 2)):
                    logits = torch.randn(
                        rows, 64 * input_stride, device="cuda", dtype=dtype
                    )[:, ::input_stride]
                    storage = torch.full(
                        (rows, 64 * output_stride),
                        7.0,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    buffer = storage[:, ::output_stride]
                    actual = LogitsProcessor._copy_logits_to_buffer(
                        SimpleNamespace(vocab_size=64),
                        logits,
                        SimpleNamespace(next_token_logits_buffer=buffer),
                    )
                    self.assertEqual(actual.data_ptr(), buffer.data_ptr())
                    torch.testing.assert_close(actual, logits.float(), rtol=0, atol=0)
                    if output_stride == 2:
                        torch.testing.assert_close(
                            storage[:, 1::2],
                            torch.full_like(storage[:, 1::2], 7.0),
                            rtol=0,
                            atol=0,
                        )

    def test_shared_gate_keeps_fused_bf16_rounding(self):
        for rows in (1, 2, 3):
            hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.02
            shared = torch.randn_like(hidden)
            routed = torch.randn_like(hidden)
            reference = routed.clone()
            fused_gate_sigmoid_mul_add(hidden, weight, shared, reference)
            actual = routed.clone()
            gate = shared_expert_gate(hidden, weight)
            shared_expert_gate_add(gate, shared, actual)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_hc_partial_graphs_do_not_overwrite_each_other(self):
        for rows in (1, 2, 3):
            residual = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            normed = torch.randn_like(residual)
            block = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            weights = [
                torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
                for _ in range(2)
            ]
            graphs, outputs = [], []
            for weight in weights:
                for _ in range(3):
                    hc_combine_apply(block, residual, hc_combine_gate(normed, weight))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    partials = hc_combine_gate(normed, weight)
                    result = hc_combine_apply(block, residual, partials)
                graphs.append(graph)
                outputs.append(result)
            for _ in range(20):
                normed.mul_(0.99)
                for graph in reversed(graphs):
                    graph.replay()
                for weight, output in zip(weights, outputs):
                    reference = hc_combine_split(
                        block, residual, normed, weight, 4, 2560
                    )
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_shared_call_ffn_state_next_norm_breakable_segments(self):
        from types import SimpleNamespace

        import torch.nn.functional as F

        from sglang.srt.environ import envs
        from sglang.srt.layers.hyperconnection import (
            GatedResidual,
            GroupedGemmaRMSNorm,
            HyperConnectionConfig,
        )
        from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
        from sglang.srt.layers.moe.qwen38_cache_overlap import (
            current_fill_overlap,
            run_shared_overlap,
        )
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            _capture_expert_output,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        @eager_on_graph(True, capture_stub=_capture_expert_output)
        def expert(_block, hidden, *, shared_call, shared_input):
            shared_call.run(shared_input)
            return hidden * 0.25

        for rows in (1, 2, 3):
            hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.1
            residual = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            normed = torch.randn_like(residual)
            layers = []
            for _ in range(2):
                hc = GatedResidual(
                    HyperConnectionConfig(hidden_size=2560, hc_per_branch_norm=True),
                    use_mix=False,
                ).cuda()
                hc.block_inject_weight.weight.data.normal_(0, 0.02)
                next_norm = (
                    GroupedGemmaRMSNorm(10240, group_size=2560).cuda().bfloat16()
                )
                next_norm.weight.data.normal_(0, 0.02)
                gate = SimpleNamespace(
                    weight=torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
                    * 0.02
                )
                # A small dense shared FFN exercises the same call/gate/state
                # boundary without allocating the full model's expert stack.
                up = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
                down = torch.randn(2560, 64, device="cuda", dtype=torch.bfloat16) * 0.02
                block = SimpleNamespace(shared_expert_gate=gate, reduce_results=False)
                block._forward_shared_experts = lambda x, apply_gate, u=up, d=down: (
                    F.linear(F.silu(F.linear(x, u)), d)
                )

                def routed(x, b=block):
                    call, original = current_fill_overlap()
                    return expert(b, x, shared_call=call, shared_input=original)

                block._forward_router_experts = routed
                state = GatedResidualState(
                    expand=lambda x: x,
                    attn_mix=None,
                    ffn_mix=None,
                    attn_combine=hc.combine,
                    ffn_combine=hc.combine,
                )
                layers.append((hc, next_norm, block, state))
            for hc_enabled, shared_enabled in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                with (
                    envs.SGLANG_OPT_QWEN38_HC_GATE_OVERLAP.override(hc_enabled),
                    envs.SGLANG_OPT_QWEN38_SHARED_GATE_OVERLAP.override(shared_enabled),
                ):

                    def forward():
                        outputs = []
                        for hc, next_norm, block, state in layers:
                            state.normed = normed
                            state.gate_partials = (
                                torch.empty((rows, 8, 4), device="cuda")
                                if hc_enabled
                                else None
                            )
                            payload = (
                                (
                                    state.normed,
                                    hc.block_inject_weight.weight,
                                    state.gate_partials,
                                )
                                if hc_enabled
                                else None
                            )
                            out = run_shared_overlap(
                                block, hidden, "verify", hc_gate=payload
                            )
                            combined = state.apply_ffn_combine(out, residual)
                            fused, normalized = hc.combine_and_normalize(
                                out,
                                (residual, state.normed, state.gate_partials),
                                next_norm,
                            )
                            if normalized is None:
                                normalized = next_norm(fused)
                            outputs.append((combined, fused, normalized))
                            state.clear_coefficients()
                        return outputs

                    def check(outputs):
                        for (hc, next_norm, block, state), (
                            combined,
                            fused,
                            normalized,
                        ) in zip(layers, outputs):
                            routed_out = hidden * 0.25
                            shared = block._forward_shared_experts(
                                hidden, apply_gate=False
                            )
                            fused_gate_sigmoid_mul_add(
                                hidden,
                                block.shared_expert_gate.weight.squeeze(),
                                shared,
                                routed_out,
                            )
                            reference = hc.combine(routed_out, (residual, normed, None))
                            torch.testing.assert_close(
                                combined, reference, rtol=0, atol=0
                            )
                            torch.testing.assert_close(fused, reference, rtol=0, atol=0)
                            torch.testing.assert_close(
                                normalized, next_norm(reference), rtol=0, atol=0
                            )
                            self.assertIsNone(state.gate_partials)
                            self.assertIsNone(state.normed)

                    check(forward())
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            forward()
                    stream.synchronize()
                    graph = BreakableCUDAGraph()
                    with BreakableCUDAGraphCapture(
                        graph, stream=stream, pool=torch.cuda.graph_pool_handle()
                    ):
                        outputs = forward()
                    for _ in range(5):
                        hidden.mul_(0.9)
                        normed.mul_(0.95)
                        residual.add_(0.01)
                        torch.cuda.synchronize()
                        graph.replay()
                        torch.cuda.synchronize()
                        check(outputs)

    def test_logits_cast_handles_vocab_padding_and_strided_rows(self):
        for dtype in (torch.bfloat16, torch.float16):
            for rows in (1, 2, 3):
                logits = torch.randn(rows, 248352, device="cuda", dtype=dtype)[
                    :, :248320
                ]
                output = torch.full((rows, 248336), float("nan"), device="cuda")
                cast_to_float32_logits(logits, output[:, :248320])
                torch.testing.assert_close(
                    output[:, :248320], logits.float(), rtol=0, atol=0
                )
                self.assertTrue(bool(output[:, 248320:].isnan().all()))


if __name__ == "__main__":
    unittest.main()
