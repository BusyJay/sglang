# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned immutable exception banks; cache slots retain small base slabs."""

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import csf_patch_headers


class CsfBanks:
    def __init__(self, pack, device):
        self.banks = {}
        self.regions = {}
        self.bytes = 0
        mapping = None
        for name, spec in pack.manifest.scale_encoding.items():
            if spec["encoding"] != "b12x_csf_tiered_v1":
                continue
            if mapping is None:
                mapping = pack.cuda_source_mapping()
            for layer, region in spec["overflow"].items():
                count = region["nbytes"]
                value = None
                if count:
                    value = torch.frombuffer(
                        mapping, dtype=torch.uint8, count=count, offset=region["offset"]
                    ).to(device)
                    self.bytes += value.nbytes
                self.banks[int(layer), name] = (
                    value,
                    value.data_ptr() - region["offset"] if value is not None else 0,
                )
                self.regions[int(layer), name] = region

    def delta(self, layer, name):
        return self.banks.get((layer, name), (None, 0))[1]

    def patch(self, layer, tensors):
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            if (layer, name) in self.banks:
                region = self.regions[layer, name]
                csf_patch_headers(
                    tensors[name],
                    self.delta(layer, name),
                    region["offset"],
                    region["offset"] + region["nbytes"],
                )

    def close(self):
        self.banks.clear()
        self.regions.clear()
        self.bytes = 0
