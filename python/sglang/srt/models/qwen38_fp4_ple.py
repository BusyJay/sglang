# SPDX-License-Identifier: Apache-2.0
"""Packed NVFP4 PLE rows: bounded CPU caches, GPU dequantization on gather."""

from __future__ import annotations

import logging
import os
import time

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.models.qwen4_exp_ple_lazy import PleCheckpointShard, PleRowCache
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table, make_ple_file_prefetcher, make_ple_file_rss_trimmer,
)

logger = logging.getLogger(__name__)


class PackedPleRowCache:
    def __init__(self, weights, scales, *, capacity_rows, width, pin_memory=True):
        self.weights = PleRowCache(
            weights,
            row_bytes=width // 2,
            capacity_rows=capacity_rows,
            pin_memory=pin_memory,
        )
        try:
            self.scales = PleRowCache(
                scales,
                row_bytes=width // 16,
                capacity_rows=capacity_rows,
                pin_memory=pin_memory,
            )
        except BaseException:
            self.weights.close()
            raise
        self.hits = self.misses = self.read_bytes = self.read_calls = 0

    def fill(self, rows, out):
        split = self.weights.row_bytes
        # Both caches share row order and capacity; combined storage is 9H/16.
        # Views have a row stride; PleRowCache's byte I/O honors NumPy strides.
        self.weights.fill(rows, out[:, :split])
        self.scales.fill(rows, out[:, split:])
        self.hits, self.misses = self.weights.hits, self.weights.misses
        self.read_bytes = self.weights.read_bytes + self.scales.read_bytes
        self.read_calls = self.weights.read_calls + self.scales.read_calls

    def close(self):
        self.weights.close()
        self.scales.close()


class Qwen38Fp4PleEmbedding(Qwen4ExpPinnedHostEmbedding):
    """Packed host PLE with explicit pinned, file or bounded lazy storage.

    Build metadata first; materialize only compressed rows after validating all
    checkpoint shards. Both host backends gather/dequantize directly on GPU.
    """

    def __init__(self, embedding, *, backend="pinned", table_dir=None,
                 cache_tables=1, checkpoint_rows=None):
        if backend not in ("pinned", "file", "lazy"):
            raise ValueError(f"Unsupported FP4 PLE backend: {backend}")
        super().__init__(embedding, backend="lazy", cache_tables=cache_tables,
                         checkpoint_rows=checkpoint_rows)
        self._backend = backend
        self._table_dir = table_dir
        self._lazy = backend == "lazy"
        self._packed_ready = False
        self._shared_cache_spec = None
        self._shared_ple = None
        rows = self.weight.shape[0]
        self.weight = nn.Parameter(
            torch.empty(
                rows, self.embedding_dim // 2, dtype=torch.uint8, device="meta"
            ),
            requires_grad=False,
        )
        self._scale_shards = []
        self._fp4_ple = True
        self._loaded_parts = set()
        # Keep the checkpoint's FP32 global factor separate from the legacy
        # outer weight_scale (one). Apply it before the gather rounds to BF16.
        self.register_buffer(
            "fp4_global_scale",
            torch.full(
                (1,),
                float("nan"),
                dtype=torch.float32,
                device=embedding.weight_scale.device,
            ),
        )

    def load_checkpoint_part(self, shard, kind, tensor, num_shards):
        if not isinstance(tensor, PleCheckpointShard):
            raise ValueError("FP4 PLE requires indexed checkpoint descriptors")
        width = self.embedding_dim // (2 if kind == "weight" else 16)
        dtype = torch.uint8 if kind == "weight" else torch.float8_e4m3fn
        if tensor.dtype != dtype or tensor.shape[1] != width:
            raise ValueError(f"invalid packed PLE {kind} dtype/width")
        key = (shard, kind)
        if key in self._loaded_parts:
            raise ValueError("duplicate packed PLE shard")
        self._loaded_parts.add(key)
        size = (self.org_vocab_size + num_shards - 1) // num_shards
        start = shard * size
        end = start + tensor.shape[0]
        collection = self._lazy_shards if kind == "weight" else self._scale_shards
        collection.append((start, end, tensor))

    def finalize_lazy(self):
        if hasattr(self, "fp4_global_scale"):
            scale = self.fp4_global_scale
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("FP4 PLE requires a finite positive weight_scale_2")
        weights = sorted(self._lazy_shards, key=lambda x: x[0])
        scales = sorted(self._scale_shards, key=lambda x: x[0])
        if not weights or [(a, b) for a, b, _ in weights] != [
            (a, b) for a, b, _ in scales
        ]:
            raise ValueError("packed PLE weight/scale coverage differs")
        cursor = 0
        for start, end, _ in weights:
            if start != cursor or end <= start or end > self.org_vocab_size:
                raise ValueError("invalid packed PLE row coverage")
            cursor = end
        if cursor < self._checkpoint_rows:
            raise ValueError("packed PLE does not cover the n-gram vocabulary")
        self._row_bytes = self.embedding_dim // 2 + self.embedding_dim // 16
        if not self._lazy:
            if self._packed_ready:
                return
            local_rows = self.weight.shape[0]
            start = self.shard_indices.org_vocab_start_index
            valid_rows = max(0, min(start + local_rows, cursor) - start)
            packed_cache_enabled = os.getenv("SGLANG_QWEN38_PLE_PACKED_CACHE", "1")
            if packed_cache_enabled not in ("0", "1"):
                raise ValueError("SGLANG_QWEN38_PLE_PACKED_CACHE must be 0 or 1")
            packed_cache_dir = self._table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get()
            cached_path = None
            if self._backend == "pinned" and packed_cache_enabled == "1" and valid_rows == local_rows:
                from sglang.srt.models.qwen38_ple_packed_cache import ensure_packed_ple_cache

                def write_cache(view):
                    packed = torch.frombuffer(view, dtype=torch.uint8).view(local_rows, self._row_bytes)
                    try:
                        copy_packed_ple_table(packed, weights, scales, valid_rows,
                                              self.embedding_dim, row_start=start)
                    finally:
                        del packed

                cached_path, _, _ = ensure_packed_ple_cache(
                    packed_cache_dir, weights, scales, local_rows, self.embedding_dim,
                    start, write_cache,
                )
            allocation_start = time.perf_counter()
            shared_spec = getattr(self, "_shared_cache_spec", None)
            if cached_path is not None and shared_spec is not None:
                from sglang.srt.models.qwen38_shared_ple import try_shared_ple

                self._shared_ple = try_shared_ple(
                    cached_path, shared_spec, self.fp4_global_scale.device
                )
            if getattr(self, "_shared_ple", None) is not None:
                table = self._shared_ple.raw.view(local_rows, self._row_bytes)
            else:
                table = allocate_ple_host_table(
                    shape=(local_rows, self._row_bytes),
                    dtype=torch.uint8,
                    backend=self._backend,
                    table_dir=self._table_dir,
                    tag=f"fp4-rows{start}-{start + local_rows}",
                )
            allocation_seconds = time.perf_counter() - allocation_start
            copy_start = time.perf_counter()
            if getattr(self, "_shared_ple", None) is not None:
                pass  # The registered tmpfs mapping already contains final rows.
            elif cached_path is not None:
                from sglang.srt.models.qwen38_ple_packed_cache import (
                    read_packed_ple_cache,
                )

                # NumPy exports the tensor storage; no full-table temporary copy.
                read_packed_ple_cache(cached_path, table.numpy())
            else:
                copy_packed_ple_table(
                    table,
                    weights,
                    scales,
                    valid_rows,
                    self.embedding_dim,
                    row_start=start,
                )
            logger.info(
                "FP4 PLE host preparation: backend=%s cached=%s allocate_seconds=%.3f load_seconds=%.3f",
                self._backend,
                cached_path is not None,
                allocation_seconds,
                time.perf_counter() - copy_start,
            )
            if valid_rows < local_rows:
                table[valid_rows:].zero_()
            self.weight = nn.Parameter(table, requires_grad=False)
            self._file_prefetcher = make_ple_file_prefetcher(table)
            self._file_rss_trimmer = make_ple_file_rss_trimmer(table)
            self._packed_ready = True
            logger.info(
                "FP4 PLE %s ready: rows=%d row_bytes=%d table=%.2f GiB; packed GPU gather",
                self._backend,
                local_rows,
                self._row_bytes,
                table.numel() / (1 << 30),
            )
            return
        budget = envs.SGLANG_QWEN4_PLE_LAZY_CACHE_MB.get()
        batch = envs.SGLANG_QWEN4_PLE_LAZY_BATCH_ROWS.get()
        self._row_bytes = self.embedding_dim // 2 + self.embedding_dim // 16
        capacity = min(
            cursor, int(budget * (1 << 20) / self._cache_tables / self._row_bytes)
        )
        if capacity < 1 or batch < 1:
            raise ValueError("invalid packed PLE cache budget")
        self._lazy_cache = PackedPleRowCache(
            weights, scales, capacity_rows=capacity, width=self.embedding_dim
        )
        self._batch_rows = min(capacity, batch)
        self._lazy_staging = torch.empty(
            (self._batch_rows, self._row_bytes), dtype=torch.uint8, device="cpu", pin_memory=True
        )

        if os.getenv("SGLANG_QWEN38_PLE_DOUBLE_STAGING", "1") == "1":
            self._lazy_staging_ring = [
                self._lazy_staging,
                torch.empty_like(self._lazy_staging, device="cpu", pin_memory=True),
            ]
            self._lazy_staging_events = [None, None]
            self._lazy_staging_index = 0

    def gather(self, input_ids, out=None):
        if self._lazy:
            return super().gather(input_ids, out=out)
        if not self._packed_ready:
            raise RuntimeError("FP4 PLE packed host table has not been finalized")
        shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            out = self.allocate_output(shape, input_ids.device)
        elif tuple(out.shape) != shape or out.dtype != torch.bfloat16 or out.device != input_ids.device:
            raise ValueError("Invalid FP4 PLE prefetch output shape, dtype or device")
        ids = input_ids.reshape(-1).long()
        if ids.numel():
            start = self.shard_indices.org_vocab_start_index
            end = min(self.shard_indices.org_vocab_end_index, self._checkpoint_rows)
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(ids, vocab_start=start, vocab_end=end)
            from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                (
                    self._shared_ple.pointer
                    if getattr(self, "_shared_ple", None) is not None
                    else self.weight.data_ptr()
                ),
                ids,
                out,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=start,
                tp_vocab_end=end,
                is_fp8=False,
                BLOCK_D=self._block_d,
                MASK_INVALID=True,
                IS_NVFP4=True,
                global_scale_ptr=self.fp4_global_scale,
            )
        return out

    def close(self):
        shared = getattr(self, "_shared_ple", None)
        if shared is not None:
            # Drop the Parameter's storage alias before closing the mmap. The
            # owner retains its raw tensor until queued GPU readers finish.
            self.weight.data = torch.empty(0, dtype=torch.uint8, device="cpu")
            shared.close()
            self._shared_ple = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.debug("shared PLE cleanup during teardown", exc_info=True)


def copy_packed_ple_table(
    target, weights, scales, rows, width, chunk_rows=65536, *, row_start=0
):
    """Stream contiguous checkpoint planes into interleaved compressed rows.

    No per-row Python cache or full-table dequantization. Temporary storage is
    bounded by chunk_rows * row_bytes, including for the 320M-row table.
    """
    if chunk_rows < 1 or rows < 0 or row_start < 0:
        raise ValueError("Invalid packed PLE copy bounds")
    if [(a, b) for a, b, _ in weights] != [(a, b) for a, b, _ in scales]:
        raise ValueError("Packed PLE plane coverage differs")
    if target.dtype != torch.uint8 or target.device.type != "cpu" or target.shape[0] < rows or target.shape[1] != width // 2 + width // 16:
        raise ValueError("Invalid packed PLE copy target")
    written = 0
    for (start, end, weight), (_, _, scale) in zip(weights, scales):
        low, high = max(start, row_start), min(end, row_start + rows)
        if low >= high:
            continue
        if low != row_start + written:
            raise ValueError("Missing or overlapping packed PLE rows")
        with open(weight.path, "rb", buffering=0) as wf, open(scale.path, "rb", buffering=0) as sf:
            for handle, desc in ((wf, weight), (sf, scale)):
                st = os.fstat(handle.fileno())
                if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != desc.identity:
                    raise ValueError("PLE checkpoint changed during pinned loading")
            for base in range(low, high, chunk_rows):
                count = min(chunk_rows, high - base)
                dst = target[base - row_start:base - row_start + count]
                for handle, desc, byte_width, view in (
                    (wf, weight, width // 2, dst[:, :width // 2]),
                    (sf, scale, width // 16, dst[:, width // 2:]),
                ):
                    handle.seek(desc.offset + (base - start) * byte_width)
                    payload = bytearray(count * byte_width)
                    mv, done = memoryview(payload), 0
                    while done < len(mv):
                        n = handle.readinto(mv[done:])
                        if not n:
                            raise OSError("Short packed PLE checkpoint read")
                        done += n
                    view.copy_(torch.frombuffer(payload, dtype=torch.uint8).view(count, byte_width))
        written += high - low
    if written != rows:
        raise ValueError("Packed PLE checkpoint does not cover requested rows")
