# SPDX-License-Identifier: Apache-2.0
"""Load only the resident FP8/BF16 MTP branch; never instantiate target packs."""

from __future__ import annotations

import torch

from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    DefaultModelLoader,
    _get_quantization_config,
    _initialize_model,
)
from sglang.srt.model_loader.qwen38_mtp_common import (
    mtp_checkpoint_files,
    mtp_weights_iterator,
)
from sglang.srt.model_loader.qwen38_nvfp4_common import (
    QWEN38_EXPECTED,
    resolve_cached_checkpoint,
    validate_cached_checkpoint_revision,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader


def load_mtp_weights_strict(model, weights):
    """Audit real loader calls, including every slice of stacked/expert tensors.

    The inherited MTP loader's returned names also include skipped weights, so
    they cannot establish coverage. Observe actual parameter writes instead.
    """
    params = dict(model.named_parameters())
    loaded, consumed, slices = set(), set(), {}
    current = [None]
    originals = {}
    try:
        for name, param in params.items():
            original = getattr(param, "weight_loader", None)
            originals[name] = original

            def loader(
                p,
                tensor,
                *args,
                _name=name,
                _loader=original or default_weight_loader,
                **kwargs,
            ):
                result = _loader(p, tensor, *args, **kwargs)
                loaded.add(_name)
                consumed.add(current[0])
                shard = kwargs.get("shard_id", args[0] if args else None)
                expert = kwargs.get("expert_id")
                slices.setdefault(_name, set()).add((expert, shard))
                return result

            param.weight_loader = loader

        def audited_weights():
            for name, tensor in weights:
                current[0] = name
                yield name, tensor
                if name not in consumed:
                    raise ValueError(f"MTP checkpoint tensor was not loaded: {name}")

        model.load_weights(audited_weights())
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"MTP parameters left uninitialized: {sorted(missing)}")
        for name in params:
            if ".experts.w13_" in name:
                expected = {
                    (expert, shard) for expert in range(512) for shard in ("w1", "w3")
                }
            elif ".experts.w2_" in name:
                expected = {(expert, "w2") for expert in range(512)}
            elif ".qkv_proj." in name:
                expected = {(None, shard) for shard in ("q", "k", "v")}
            elif ".gate_up_proj." in name:
                expected = {(None, shard) for shard in (0, 1)}
            else:
                continue
            if not expected <= slices.get(name, set()):
                raise ValueError(f"MTP parameter slices left uninitialized: {name}")
    finally:
        for name, param in params.items():
            if originals[name] is None:
                del param.weight_loader
            else:
                param.weight_loader = originals[name]


class Qwen38MtpModelLoader(BaseModelLoader):
    def _prepare_checkpoint(self, model_config):
        extra = self.load_config.model_loader_extra_config or {}
        checkpoint = resolve_cached_checkpoint(
            model_config.model_path,
            revision=model_config.revision,
            cache_dir=self.load_config.download_dir,
        )
        validate_cached_checkpoint_revision(
            checkpoint, model_config.hf_config, extra.get("checkpoint_revision")
        )
        return mtp_checkpoint_files(checkpoint)

    def download_model(self, model_config):
        self._prepare_checkpoint(model_config)

    def load_model(self, *, model_config, device_config):
        config = model_config.hf_config
        text = getattr(config, "text_config", config)
        if config.architectures != ["Qwen4ExpForCausalLMMTP"]:
            raise ValueError("qwen38_mtp is a draft-only Qwen4Exp MTP loader")
        expected = {
            **QWEN38_EXPECTED,
            "num_hidden_layers": 1,
            "hc_count": 4,
            "num_nextn_predict_layers": 1,
        }
        for field, value in expected.items():
            if getattr(text, field, None) != value:
                raise ValueError(f"qwen38_mtp requires {field}={value}")
        if model_config.dtype != torch.bfloat16:
            raise ValueError("qwen38_mtp requires BF16 model dtype")
        selected, files = self._prepare_checkpoint(model_config)
        quant = _get_quantization_config(model_config, self.load_config)
        if quant is None or quant.get_name() != "modelopt_mixed":
            raise ValueError("qwen38_mtp requires original ModelOpt mixed precision")
        if quant.resolve_quant_algo(
            "mtp.layers.0.mlp.experts"
        ) != "FP8_BLOCK_SCALES" or quant.fp8_block_config.weight_block_size != [
            128,
            128,
        ]:
            raise ValueError("qwen38_mtp requires serialized 128x128 block-FP8 experts")
        device = torch.device(device_config.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError("qwen38_mtp currently requires an SM120 CUDA device")
        from sglang.srt.layers.moe.qwen38_startup_memory import (
            get_startup_memory_probe,
            memory_mark,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        probe = get_startup_memory_probe(device)
        with set_default_torch_dtype(model_config.dtype), device:
            memory_mark(probe, "mtp.before_construct")
            model = _initialize_model(model_config, self.load_config, quant)
            memory_mark(probe, "mtp.after_construct", model=model)
            methods = [
                getattr(module, "quant_method", None) for module in model.modules()
            ]
            fp8 = [method for method in methods if isinstance(method, Fp8MoEMethod)]
            if (
                len(fp8) != 1
                or not fp8[0].block_quant
                or not fp8[0].runner.runner_backend.is_triton()
            ):
                raise ValueError("MTP routed experts must dispatch to Triton block-FP8")
            weights = mtp_weights_iterator(selected, files)
            try:
                load_mtp_weights_strict(model, weights)
            finally:
                weights.close()
            memory_mark(
                probe,
                "mtp.after_load",
                model=model,
                metadata={"tensor_count": len(selected)},
            )
            DefaultModelLoader.postprocess_weights(
                model, device, startup_memory_probe=probe
            )
            memory_mark(probe, "mtp.after_postprocess", model=model)
        return model.eval()
