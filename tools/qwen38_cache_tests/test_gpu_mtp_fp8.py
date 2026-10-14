"""SM120 block-FP8 draft MoE parity at the checkpoint's actual expert shapes."""

import unittest

import torch
import torch.nn.functional as F


class TestMtpFp8Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("MTP parity requires SM120")
        from sglang.test.layer_ut_utils import init_single_process_dist

        init_single_process_dist()

    def test_checkpoint_shapes_and_verify_widths(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )
        from sglang.srt.runtime_context import get_context, get_flags, get_parallel

        torch.manual_seed(5090)
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON, a2a_backend=MoeA2ABackend.NONE
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
            torch.device("cuda"),
        ):
            quant = ModelOptMixedPrecisionConfig.from_config(
                {
                    "quant_algo": "MIXED_PRECISION",
                    "packed_modules_mapping": {},
                    "quantized_layers": {
                        "mtp.layers.0.mlp.experts": {
                            "quant_algo": "FP8_BLOCK_SCALES",
                            "group_size": 128,
                        }
                    },
                }
            )
            layer = FusedMoE(
                num_experts=10,
                top_k=10,
                hidden_size=2560,
                intermediate_size=640,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                layer_id=0,
                inplace=False,
                prefix="mtp.layers.0.mlp.experts",
            )
            references = []
            for expert in range(10):
                parts = {}
                for shard, shape in (
                    ("w1", (640, 2560)),
                    ("w3", (640, 2560)),
                    ("w2", (2560, 640)),
                ):
                    weight = torch.randn(shape, dtype=torch.float32) * 0.02
                    blocks = weight.reshape(shape[0] // 128, 128, shape[1] // 128, 128)
                    scale = (blocks.abs().amax(dim=(1, 3)) / 448).to(torch.bfloat16)
                    expanded = (
                        scale.float()
                        .repeat_interleave(128, 0)
                        .repeat_interleave(128, 1)
                    )
                    packed = (weight / expanded).to(torch.float8_e4m3fn)
                    parts[shard] = (packed.float() * expanded).to(torch.bfloat16)
                    prefix = "w13" if shard != "w2" else "w2"
                    for suffix, tensor in (
                        ("weight", packed),
                        ("weight_scale_inv", scale),
                    ):
                        name = f"{prefix}_{suffix}"
                        layer.weight_loader(
                            getattr(layer, name),
                            tensor,
                            name,
                            shard_id=shard,
                            expert_id=expert,
                        )
                references.append(parts)
            layer.quant_method.process_weights_after_loading(layer)
            self.assertTrue(layer.quant_method.runner.runner_backend.is_triton())
            for width in (1, 2, 3, 4):
                hidden = torch.randn(width, 2560, dtype=torch.bfloat16)
                ids = torch.arange(10, dtype=torch.int32).expand(width, 10).contiguous()
                weights = torch.softmax(torch.randn(width, 10), dim=-1)
                actual = layer(hidden, StandardTopKOutput(weights, ids, None))
                reference = torch.zeros_like(hidden, dtype=torch.float32)
                for expert, parts in enumerate(references):
                    gate, up = (
                        F.linear(hidden, parts["w1"]),
                        F.linear(hidden, parts["w3"]),
                    )
                    out = F.linear(
                        (F.silu(gate.float()) * up.float()).to(torch.bfloat16),
                        parts["w2"],
                    )
                    reference += out.float() * weights[:, expert, None]
                # Includes dynamic activation FP8 quantization in the runtime.
                relative_error = (actual.float() - reference).norm() / reference.norm()
                self.assertLess(relative_error.item(), 0.08, f"verify width={width}")
                from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                    BreakableCUDAGraph,
                    BreakableCUDAGraphCapture,
                )

                static_hidden, static_weights, static_ids = (
                    hidden.clone(),
                    weights.clone(),
                    ids.clone(),
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        layer(
                            static_hidden,
                            StandardTopKOutput(static_weights, static_ids, None),
                        )
                stream.synchronize()
                graph = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(graph, stream=stream):
                    graph_output = layer(
                        static_hidden * 1.0,
                        StandardTopKOutput(static_weights * 1.0, static_ids + 0, None),
                    )
                torch.cuda.current_stream().wait_stream(stream)
                for _ in range(3):
                    static_hidden.copy_(torch.randn_like(hidden))
                    static_weights.copy_(
                        torch.softmax(torch.randn_like(weights), dim=-1)
                    )
                    static_ids.copy_(
                        torch.stack(
                            [
                                torch.randperm(10, dtype=torch.int32)
                                for _ in range(width)
                            ]
                        )
                    )
                    expected = layer(
                        static_hidden,
                        StandardTopKOutput(static_weights, static_ids, None),
                    )
                    graph.replay()
                    torch.testing.assert_close(
                        graph_output, expected, rtol=0.02, atol=0.02
                    )
