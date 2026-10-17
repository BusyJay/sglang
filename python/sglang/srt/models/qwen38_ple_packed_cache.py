# SPDX-License-Identifier: Apache-2.0
"""Persistent packed PLE rows; bounded I/O, atomic publication, no CUDA imports."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import mmap
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _source_metadata(shards):
    result = []
    for start, end, desc in shards:
        st = os.stat(desc.path)
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if identity != tuple(desc.identity):
            raise ValueError(f"PLE checkpoint changed: {desc.path}")
        result.append(
            {
                "start": start,
                "end": end,
                "path": os.path.realpath(desc.path),
                "identity": identity,
                "offset": desc.offset,
                "shape": list(desc.shape),
                "dtype": str(desc.dtype),
            }
        )
    return result


def packed_cache_metadata(weights, scales, rows, width, row_start):
    if rows <= 0 or width <= 0 or width % 16 or row_start < 0:
        raise ValueError("Invalid packed PLE cache geometry")
    return {
        "version": 1,
        "rows": rows,
        "width": width,
        "row_start": row_start,
        "row_bytes": width // 2 + width // 16,
        "weights": _source_metadata(weights),
        "scales": _source_metadata(scales),
    }


def ensure_packed_ple_cache(directory, weights, scales, rows, width, row_start, writer):
    """writer(memoryview) fills final row bytes; reuse only matching source metadata.

    Checkpoint files are immutable during building/loading. File identity rather
    than a full payload checksum invalidates the cache when the snapshot changes.
    The caller owns the returned file; readers never modify it.
    """
    started = time.perf_counter()
    metadata = packed_cache_metadata(weights, scales, rows, width, row_start)
    encoded = json.dumps(metadata, sort_keys=True).encode()
    # This hashes small identity metadata for a filename, not checkpoint content.
    key = hashlib.sha256(encoded).hexdigest()[:32]
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fp4-ple-v1-{key}.bin"
    meta_path = path.with_suffix(".json")
    expected = rows * metadata["row_bytes"]
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = False
        try:
            hit = path.stat().st_size == expected and json.loads(
                meta_path.read_text()
            ) == json.loads(encoded)
        except (OSError, ValueError):
            pass
        if not hit:
            temporary = temporary_meta = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=directory, prefix=path.name + ".", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.truncate(expected)
                    with mmap.mmap(
                        handle.fileno(), expected, access=mmap.ACCESS_WRITE
                    ) as mapping:
                        view = memoryview(mapping)
                        try:
                            writer(view)
                        finally:
                            view.release()
                        mapping.flush()
                    os.fsync(handle.fileno())
                # Do not publish a mixture of checkpoint versions.
                if (
                    packed_cache_metadata(weights, scales, rows, width, row_start)
                    != metadata
                ):
                    raise ValueError("PLE checkpoint changed during cache construction")
                with tempfile.NamedTemporaryFile(
                    dir=directory, prefix=meta_path.name + ".", delete=False
                ) as handle:
                    temporary_meta = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                os.replace(temporary_meta, meta_path)
            finally:
                for temporary_path in (temporary, temporary_meta):
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    logger.info(
        "FP4 PLE packed cache %s: path=%s bytes=%d seconds=%.3f",
        "hit" if hit else "built",
        path,
        expected,
        elapsed,
    )
    return path, hit, elapsed


def read_packed_ple_cache(path, destination, chunk_bytes=8 << 20):
    """Read directly into a contiguous writable byte buffer, including pinned RAM."""
    if chunk_bytes < 1:
        raise ValueError("Invalid cache read chunk size")
    view = memoryview(destination).cast("B")
    started = time.perf_counter()
    try:
        with open(path, "rb", buffering=0) as handle:
            if os.fstat(handle.fileno()).st_size != view.nbytes:
                raise ValueError("Packed PLE cache size differs from destination")
            offset = 0
            while offset < view.nbytes:
                part = view[offset : min(offset + chunk_bytes, view.nbytes)]
                try:
                    count = handle.readinto(part)
                finally:
                    part.release()
                if not count:
                    raise OSError("Short packed PLE cache read")
                offset += count
    finally:
        view.release()
    return time.perf_counter() - started
