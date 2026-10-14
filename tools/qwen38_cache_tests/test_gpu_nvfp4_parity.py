"""SM120 synthetic checkpoint -> real ModelOpt transform -> cached parity."""

import tempfile
import unittest
from pathlib import Path

import torch


class TestNvfp4CachedParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("CUTLASS runtime parity requires SM120")
        if torch.cuda.mem_get_info()[0] < 2 << 30:
            raise unittest.SkipTest("need 2 GiB free for small NVFP4 parity")
        from sglang.test.layer_ut_utils import init_single_process_dist

        init_single_process_dist()

    def _exercise(self, mode, decode_graph=False, graph_width=1):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
        from sglang.srt.model_loader.qwen38_nvfp4_pack.export import (
            export_runtime_layer,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.parity import (
            check_exported_layer,
        )
        from sglang.srt.runtime_context import get_context, get_flags, get_parallel
        from sglang.test.quant_ref_utils import (
            FLOAT4_E2M1_MAX,
            FLOAT8_E4M3_MAX,
            quantize_nvfp4_shard,
        )

        torch.cuda.reset_peak_memory_stats()
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
                a2a_backend=MoeA2ABackend.NONE,
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
            layer = FusedMoE(
                num_experts=8,
                hidden_size=1024,
                intermediate_size=1024,
                layer_id=0,
                top_k=2,
                params_dtype=torch.bfloat16,
                quant_config=ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True, group_size=16
                ),
                gate_up_interleaved=False,
                inplace=False,
            )
            torch.manual_seed(773)
            for expert in range(8):
                for shard in ["w1", "w3", "w2"]:
                    weight = (
                        torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
                        / 10
                    )
                    packed, scales, global_scale, _ = quantize_nvfp4_shard(weight)
                    target = "w13" if shard in ("w1", "w3") else "w2"
                    for suffix, loaded in [
                        ("weight", packed),
                        ("weight_scale", scales),
                        ("weight_scale_2", (1 / global_scale).clone()),
                        (
                            "input_scale",
                            torch.tensor(
                                1 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX), device="cuda"
                            ),
                        ),
                    ]:
                        name = f"{target}_{suffix}"
                        layer.weight_loader(
                            getattr(layer, name),
                            loaded,
                            name,
                            shard_id=shard,
                            expert_id=expert,
                        )
            layer.quant_method.process_weights_after_loading(layer)
            with tempfile.TemporaryDirectory() as value:
                exported = Path(value) / "layer"
                export_runtime_layer(layer, exported)
                report = check_exported_layer(
                    layer,
                    exported,
                    mode,
                    slots=4,
                    decode_graph=decode_graph,
                    graph_width=graph_width,
                )
                self.assertEqual(report["status"], "passed")
                self.assertGreater(report["arena_loads"], 0)
            self.assertLess(torch.cuda.max_memory_allocated(), 1 << 30)

    def test_sync_cutlass_parity(self):
        self._exercise("sync")

    def test_async_cutlass_parity(self):
        self._exercise("async")

    def test_device_policy_cutlass_parity(self):
        self._exercise("device")

    def test_threaded_cutlass_parity(self):
        self._exercise("threaded")

    def test_threaded_breakable_decode_cutlass_parity(self):
        self._exercise("threaded", decode_graph=True)

    def test_threaded_breakable_verify_cutlass_parity(self):
        for width in (2, 3, 4):
            with self.subTest(width=width):
                self._exercise("threaded", decode_graph=True, graph_width=width)
