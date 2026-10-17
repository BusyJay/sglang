#!/usr/bin/env python3
"""Check real ModelOpt/FusedMoE allocation and loaders on CPU/meta."""

import json
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.qwen38_fp4_mtp import Qwen38Fp4MtpMethod
from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding
from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding


def check(config_path):
    config = json.loads(Path(config_path).read_text())
    q = ModelOptMixedPrecisionConfig.from_config(config["quantization_config"])
    # In a no-GPU validation harness only, force platform for constructor checks.
    with mock.patch(
        "sglang.srt.layers.quantization.modelopt_quant.get_platform"
    ) as platform:
        platform.return_value.is_blackwell = True
        with mock.patch(
            "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
            return_value=MoeRunnerBackend.TRITON,
        ):
            module = FusedMoE.__new__(FusedMoE)
            torch.nn.Module.__init__(module)
            method = q.get_quant_method(module, "mtp.layers.0.mlp.experts")
            assert isinstance(method, Qwen38Fp4MtpMethod)
            module.quant_method = method
            module.scheme = None
            module.use_triton_kernels = False
            module.use_presharded_weights = False
            module._has_fused_shared = False
            module.quant_config = method.quant_config
            module._expert_storage_rank = 0
            module._num_local_routed = 2
            module._expert_storage_world_size = 1
            module.num_fused_shared_experts = 0
            module.num_redundant_experts = 0
            module.enable_flashinfer_cutlass_moe = False
            module.num_local_experts = module.num_experts = 2
            module.moe_ep_rank = module.moe_tp_rank = 0
            module.moe_ep_size = module.moe_tp_size = 1
            module.tp_size = 1
            module.use_flashinfer_trtllm_moe = False
            module.use_flashinfer_trtllm_moe_fp8 = False
            module.use_flashinfer_cutedsl_moe_fp4_allgather = False
            module.is_checkpoint_fp8_serialized = False
            module._enable_gate_up_interleaved = False
            module.moe_runner_config = MoeRunnerConfig(
                num_experts=2,
                num_local_experts=2,
                hidden_size=32,
                top_k=1,
                activation="silu",
            )
            # No CUDA allocation: small checkpoint-compatible shapes.
            method.create_weights(
                module,
                num_experts=2,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.bfloat16,
                weight_loader=module.weight_loader,
            )
            for expert in range(2):
                for shard, rows, cols, target in [
                    ("w1", 16, 32, "w13"),
                    ("w3", 16, 32, "w13"),
                    ("w2", 32, 16, "w2"),
                ]:
                    for suffix, shape, dtype, value in [
                        ("weight", (rows, cols // 2), torch.uint8, 17),
                        ("weight_scale", (rows, cols // 16), torch.float8_e4m3fn, 0.5),
                        (
                            "weight_scale_2",
                            (),
                            torch.float32,
                            0.1 if shard == "w1" else 0.2,
                        ),
                    ]:
                        param = getattr(module, target + "_" + suffix)
                        module.weight_loader(
                            param,
                            torch.full(shape, value, dtype=dtype),
                            target + "_" + suffix,
                            shard_id=shard,
                            expert_id=expert,
                        )
            assert torch.allclose(
                module.w13_weight_scale_2, torch.tensor([[0.1, 0.2], [0.1, 0.2]])
            )
            method.process_weights_after_loading(module)
            method.create_moe_runner(module, module.moe_runner_config)
    norm = GroupedGemmaRMSNorm(32)
    norm._weight_loader(
        norm.weight, torch.full((32,), 0.123456789, dtype=torch.float32)
    )
    assert norm.weight.dtype == torch.float32
    assert norm(torch.ones(1, 32, dtype=torch.bfloat16)).dtype == torch.bfloat16
    text = dict(config["text_config"])
    text.update(
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        ple_offload_embedding=True,
        ple_offload_backend="lazy",
    )
    from types import SimpleNamespace

    parallel = SimpleNamespace(attn_tp_size=1, tp_size=1, attn_tp_rank=0, tp_rank=0)
    with (
        mock.patch(
            "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
            return_value=parallel,
        ),
        torch.device("meta"),
    ):
        embedding = Qwen4ExpNGramEmbedding(
            Qwen4ExpTextConfig(**text),
            2560,
            quant_config=q,
            prefix="model.layers.1.ple.ple_embedding",
        )
    assert isinstance(embedding.ngram_embedding, Qwen38Fp4PleEmbedding)
    assert embedding.ngram_embedding.weight.shape[1] == 80
    print("Real NVFP4 MTP loader, FP32 norm and packed PLE construction passed.")


if __name__ == "__main__":
    import sys

    check(sys.argv[1])
