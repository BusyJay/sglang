"""Scheduler-owned slot metadata for direct MLX pool attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class MlxPoolDecodeContext:
    pool: Any
    slots: mx.array
    lengths: mx.array
    new_token_slots: mx.array

    @classmethod
    def from_requests(
        cls,
        *,
        pool: Any,
        req_ids: list[str],
        seq_lens: list[int],
        req_pool_idx: dict[str, int],
        req_to_token_pool: Any,
    ) -> MlxPoolDecodeContext:
        """Snapshot owned rows before building a lazy graph.

        This must only be called for an admitted scheduler decode, never a
        lookahead step: unused row tails can still contain recycled slot IDs.
        """
        table = req_to_token_pool.req_to_token
        width = max(seq_lens) + 1
        if width > table.shape[1]:
            raise ValueError("MLX pool decode exceeds the scheduler request row")
        rows = np.zeros((len(req_ids), width), dtype=np.int32)
        new_slots = []
        for i, (rid, offset) in enumerate(zip(req_ids, seq_lens, strict=True)):
            row = table[req_pool_idx[rid], : offset + 1].numpy()
            if np.any(row <= 0) or np.any(row >= pool.pool_size):
                raise ValueError(f"MLX pool decode has invalid owned slots for {rid}")
            rows[i, : offset + 1] = row
            new_slots.append(int(row[offset]))
        if len(set(new_slots)) != len(new_slots):
            raise ValueError("MLX pool decode requests must own distinct write slots")
        return cls(
            pool=pool,
            slots=mx.array(rows),
            lengths=mx.array([n + 1 for n in seq_lens], dtype=mx.int32),
            new_token_slots=mx.array(new_slots, dtype=mx.int32),
        )
