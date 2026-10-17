# SPDX-License-Identifier: Apache-2.0
"""BF16 token rows mapped from the checkpoint, registered for GPU gathers."""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import time

import torch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from torch import nn

logger = logging.getLogger(__name__)


def find_token_embedding(files):
    matches = []
    for path in files:
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated checkpoint: {path}")
            length = struct.unpack("<Q", raw)[0]
            if (
                not 0
                < length
                <= min(100_000_000, os.fstat(handle.fileno()).st_size - 8)
            ):
                raise ValueError(f"invalid checkpoint header: {path}")
            header = json.loads(handle.read(length))
            for name, entry in header.items():
                if name.endswith("embed_tokens.weight") and "mtp" not in name:
                    shape = entry["shape"]
                    begin, end = entry["data_offsets"]
                    if (
                        entry["dtype"] != "BF16"
                        or len(shape) != 2
                        or min(shape) <= 0
                        or begin < 0
                        or end - begin != shape[0] * shape[1] * 2
                        or (8 + length + begin) % 2
                        or 8 + length + end > os.fstat(handle.fileno()).st_size
                    ):
                        raise ValueError(f"unsupported BF16 token embedding: {name}")
                    matches.append(
                        (os.path.abspath(path), 8 + length + begin, tuple(shape))
                    )
    if len(matches) != 1:
        raise ValueError(f"expected one token embedding tensor, got {len(matches)}")
    return matches[0]


class Qwen38MappedTokenEmbedding(VocabParallelEmbedding):
    """TP=1 BF16 embedding; output/head stays on GPU, mapping lives with module."""

    def __init__(self, vocab_size, hidden_size, descriptor):
        with torch.device("meta"):
            super().__init__(vocab_size, hidden_size, enable_tp=False)
        self.quant_method = None  # Generic post-load staging must skip this table.
        self._descriptor = descriptor
        self._mapping = self._registered_bytes = None
        self._device_pointer = None
        self.output_device = torch.device("cuda", torch.cuda.current_device())
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        path, offset, shape = self._descriptor
        if (
            shape != (self.org_vocab_size, self.embedding_dim)
            or tuple(loaded_weight.shape) != shape
            or loaded_weight.dtype != torch.bfloat16
        ):
            raise ValueError("mapped token embedding checkpoint geometry mismatch")
        if self._mapping is not None:
            raise RuntimeError("mapped token embedding loaded twice")
        started = time.perf_counter()
        aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        delta = offset - aligned
        nbytes = shape[0] * shape[1] * 2
        with open(path, "rb") as handle:
            # Writable COW is required on discrete GPUs; the checkpoint is immutable.
            mapping = mmap.mmap(
                handle.fileno(), delta + nbytes, access=mmap.ACCESS_COPY, offset=aligned
            )
        raw = torch.frombuffer(mapping, dtype=torch.uint8)
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        try:
            with torch.cuda.device(self.output_device):
                pointer = _module().mailbox_register(raw)
        except BaseException:
            del raw
            mapping.close()
            raise
        self._mapping = mapping
        self._registered_bytes = raw
        self._device_pointer = pointer + delta
        weight = nn.Parameter(
            raw[delta : delta + nbytes].view(torch.bfloat16).reshape(shape),
            requires_grad=False,
        )
        for name, value in vars(param).items():
            setattr(weight, name, value)
        self.weight = weight
        logger.info(
            "Qwen38 mapped token embedding: %.1f MiB host, register %.3fs",
            nbytes / (1 << 20),
            time.perf_counter() - started,
        )

    def _embed_local_shard(self, input_):
        if self._device_pointer is None:
            raise RuntimeError("mapped token embedding has not been loaded")
        if input_.device != self.output_device:
            raise ValueError("mapped token embedding requires its CUDA device")
        import triton
        from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel

        ids = input_.reshape(-1).contiguous()
        output = torch.empty(
            (*input_.shape, self.embedding_dim),
            dtype=torch.bfloat16,
            device=input_.device,
        )
        if ids.numel():
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                self._device_pointer,
                ids,
                output,
                self.embedding_dim,
                0,
                self.org_vocab_size,
                is_fp8=False,
                BLOCK_D=triton.next_power_of_2(self.embedding_dim),
                MASK_INVALID=True,
            )
        return output

    def close(self):
        if self._registered_bytes is None:
            return
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        with torch.cuda.device(self.output_device):
            torch.cuda.synchronize(self.output_device)
            _module().mailbox_unregister(self._registered_bytes)
        self.weight.data = torch.empty(0, dtype=torch.bfloat16, device="cpu")
        self._registered_bytes = None
        self._device_pointer = None
        self._mapping.close()
        self._mapping = None

    def __del__(self):
        # Normal explicit close reports errors; interpreter CUDA teardown may not.
        try:
            self.close()
        except Exception:
            logger.debug("mapped embedding cleanup during teardown", exc_info=True)
