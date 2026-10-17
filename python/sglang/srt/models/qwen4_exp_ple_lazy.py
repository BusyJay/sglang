# SPDX-License-Identifier: Apache-2.0
"""Read-only checkpoint row index and bounded PLE cache (no CUDA imports)."""

from __future__ import annotations

import bisect
import json
import os
import re
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

PLE_SHARD = re.compile(r"\.ngram_embedding\.shard_(\d+)\.(?:weight|weight_scale)$")
_DTYPES = {
    "U8": (torch.uint8, 1),
    "BF16": (torch.bfloat16, 2),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
}


@dataclass(frozen=True)
class PleCheckpointShard:
    path: str
    shape: tuple[int, int]
    dtype: torch.dtype
    offset: int
    identity: tuple[int, int, int, int]


def read_ple_checkpoint_index(path: str) -> dict[str, PleCheckpointShard]:
    """Read only the safetensors header, never map or read tensor payloads."""
    path = os.path.abspath(path)
    with open(path, "rb", buffering=0) as handle:
        stat = os.fstat(handle.fileno())
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size <= min(100_000_000, stat.st_size - 8):
            raise ValueError(f"invalid safetensors header size: {path}")
        header = json.loads(handle.read(size))
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    result = {}
    for name, entry in header.items():
        if not PLE_SHARD.search(name):
            continue
        shape = tuple(entry["shape"])
        if len(shape) != 2 or min(shape) <= 0 or entry["dtype"] not in _DTYPES:
            raise ValueError(f"unsupported lazy PLE tensor: {name}: {entry}")
        dtype, itemsize = _DTYPES[entry["dtype"]]
        start, end = entry["data_offsets"]
        if (
            start < 0
            or end - start != shape[0] * shape[1] * itemsize
            or 8 + size + end > stat.st_size
        ):
            raise ValueError(f"invalid lazy PLE offsets: {name}")
        result[name] = PleCheckpointShard(
            path, shape, dtype, 8 + size + start, identity
        )
    return result


class PleRowCache:
    """Pinned raw-byte SLRU; batched hits, bounded parallel coalesced miss I/O.

    Callers own staging and must wait for its GPU consumer before reusing it.
    This cache itself is never read by CUDA, so eviction needs no GPU fences.
    """

    def __init__(
        self, shards, *, row_bytes: int, capacity_rows: int, pin_memory=True, workers=4
    ):
        if row_bytes <= 0 or capacity_rows <= 0 or workers <= 0:
            raise ValueError("PLE cache dimensions and workers must be positive")
        self.row_bytes = row_bytes
        self.capacity = max(1, capacity_rows)
        self.data = torch.empty(
            (self.capacity, row_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.probation = OrderedDict()
        self.protected = OrderedDict()
        self.protected_limit = int(self.capacity * 0.8)
        self.used = 0
        self.hits = self.misses = self.read_bytes = self.read_calls = 0
        self._workers = workers
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ple-read"
        )
        self._fds = {}
        self._shards = sorted(shards, key=lambda x: x[0])
        self._starts = [x[0] for x in self._shards]
        try:
            for _, _, descriptor in self._shards:
                if descriptor.path not in self._fds:
                    fd = os.open(descriptor.path, os.O_RDONLY)
                    self._fds[descriptor.path] = fd
                    stat = os.fstat(fd)
                    if (
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                    ) != descriptor.identity:
                        raise ValueError(
                            f"PLE checkpoint changed during loading: {descriptor.path}"
                        )
                    # Restrict readahead to demanded ranges, do not drop expert pages.
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __del__(self):
        if hasattr(self, "_fds"):
            self.close()

    def _slot(self, row):
        slot = self.protected.pop(row, None)
        if slot is not None:
            self.protected[row] = slot
            return slot
        slot = self.probation.pop(row, None)
        if slot is not None:
            self.protected[row] = slot
            if len(self.protected) > self.protected_limit:
                key, value = self.protected.popitem(last=False)
                self.probation[key] = value
        return slot

    def _read_run(self, run, target):
        fd, first, last, positions, offsets = run
        count = last - first + self.row_bytes
        payload = bytearray(count)
        view = memoryview(payload)
        done = 0
        calls = 0
        while done < count:
            n = os.preadv(fd, [view[done:]], first + done)
            calls += 1
            if n == 0:
                raise OSError("short PLE checkpoint read")
            done += n
        raw = np.frombuffer(payload, dtype=np.uint8)
        selected = raw[np.asarray(offsets)[:, None] - first + np.arange(self.row_bytes)]
        target[np.asarray(positions)] = selected
        return count, calls

    def fill(self, rows: list[int], out: torch.Tensor):
        """Fill unique global rows; out is caller-owned pinned uint8 storage."""
        if len(rows) > self.capacity or tuple(out.shape) != (len(rows), self.row_bytes):
            raise ValueError("PLE cache batch exceeds capacity or staging shape")
        hit_positions, hit_slots, missing = [], [], []
        for pos, row in enumerate(rows):
            slot = self._slot(row)
            if slot is None:
                missing.append((pos, row))
            else:
                hit_positions.append(pos)
                hit_slots.append(slot)
        self.hits += len(hit_slots)
        self.misses += len(missing)
        if hit_slots:
            # Byte indexing also works for FP8 on CPU.
            out.index_copy_(
                0,
                torch.tensor(hit_positions),
                self.data.index_select(0, torch.tensor(hit_slots)),
            )
        runs = []
        for pos, row in sorted(missing, key=lambda x: x[1]):
            shard_idx = bisect.bisect_right(self._starts, row) - 1
            if shard_idx < 0 or row >= self._shards[shard_idx][1]:
                raise ValueError(f"PLE row not covered by checkpoint: {row}")
            start, _, desc = self._shards[shard_idx]
            offset = desc.offset + (row - start) * self.row_bytes
            fd = self._fds[desc.path]
            # Merge nearby rows within a bounded read, including page neighbors.
            if (
                runs
                and runs[-1][0] == fd
                and 0 <= offset - runs[-1][2] <= self.row_bytes + 4096
                and offset + self.row_bytes - runs[-1][1] <= 256 * 1024
            ):
                runs[-1][2] = offset
                runs[-1][3].append(pos)
                runs[-1][4].append(offset)
            else:
                runs.append([fd, offset, offset, [pos], [offset]])
        # At most `workers` reads are in flight; no unbounded future queue.
        target = out.numpy()

        def read_partition(partition):
            totals = [0, 0]
            for run in partition:
                size, calls = self._read_run(run, target)
                totals[0] += size
                totals[1] += calls
            return totals

        # Small decode lookups should not pay thread scheduling on cache hits
        # or a handful of reads. Parallelize only sizeable miss batches.
        if len(runs) <= 8:
            size, calls = read_partition(runs)
            self.read_bytes += size
            self.read_calls += calls
            futures = []
        else:
            futures = [
                self._pool.submit(read_partition, runs[i :: self._workers])
                for i in range(min(self._workers, len(runs)))
            ]
        try:
            for future in futures:
                size, calls = future.result()
                self.read_bytes += size
                self.read_calls += calls
        finally:
            # Do not reuse staging until all writers have stopped, even on error.
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass
        new_slots = []
        for _, row in missing:
            if self.used < self.capacity:
                slot = self.used
                self.used += 1
            else:
                queue = self.probation if self.probation else self.protected
                _, slot = queue.popitem(last=False)
            new_slots.append(slot)
        for (_, row), slot in zip(missing, new_slots):
            self.probation[row] = slot
        if new_slots:
            self.data.index_copy_(
                0,
                torch.tensor(new_slots),
                out.index_select(0, torch.tensor([p for p, _ in missing])),
            )
