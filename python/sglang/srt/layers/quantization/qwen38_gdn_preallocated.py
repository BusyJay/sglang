# SPDX-License-Identifier: Apache-2.0
"""Allocate the two BF16 GDN projections in their final shared storage."""

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import set_weight_attrs


class Qwen38PreallocatedGdnLinearMethod(UnquantizedLinearMethod):
    def __init__(self, pending, key, projection, value_heads):
        self.pending = pending
        self.key = key
        self.projection = projection
        self.value_heads = value_heads

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        rows = sum(output_partition_sizes)
        if params_dtype != torch.bfloat16:
            raise ValueError("preallocated GDN projections require BF16")
        if self.projection == "in_proj_qkvz":
            ba_rows = 2 * self.value_heads // layer.tp_size
            if self.key in self.pending:
                raise ValueError("duplicate GDN QKVZ allocation")
            packed = torch.empty(
                (rows + ba_rows, input_size_per_partition), dtype=params_dtype
            )
            self.pending[self.key] = (packed, rows, ba_rows)
            data = packed[:rows]
        else:
            packed, qkvz_rows, ba_rows = self.pending.pop(self.key)
            if rows != ba_rows or packed.shape[1] != input_size_per_partition:
                raise ValueError("GDN BA dimensions differ from preallocated storage")
            data = packed[qkvz_rows:]
        weight = torch.nn.Parameter(data, requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        layer._qwen38_gdn_packed_weight = packed


def preallocated_gdn_weight(qkvz, ba):
    """Return the original packed storage only while both parameters alias it."""
    packed = getattr(qkvz, "_qwen38_gdn_packed_weight", None)
    if packed is None or getattr(ba, "_qwen38_gdn_packed_weight", None) is not packed:
        return None
    split = qkvz.weight.shape[0]
    if (
        packed.shape != (split + ba.weight.shape[0], qkvz.weight.shape[1])
        or ba.weight.shape[1] != packed.shape[1]
        or not qkvz.weight.is_contiguous()
        or not ba.weight.is_contiguous()
        or qkvz.weight.dtype != packed.dtype
        or ba.weight.dtype != packed.dtype
        or qkvz.weight.device != packed.device
        or ba.weight.device != packed.device
        or qkvz.weight.data_ptr() != packed.data_ptr()
        or ba.weight.data_ptr() != packed[split:].data_ptr()
    ):
        return None
    return packed
