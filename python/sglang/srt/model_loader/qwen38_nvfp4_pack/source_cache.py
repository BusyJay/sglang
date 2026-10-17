# SPDX-License-Identifier: Apache-2.0
"""Persistent tmpfs expert sources; CUDA registration still belongs to the reader."""

import ctypes
import fcntl
import hashlib
import json
import mmap
import os
import tempfile
from pathlib import Path


def _identity(fd):
    stat = os.fstat(fd)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _require_tmpfs(fd):
    # Linux struct statfs starts with a native long f_type. Allocate more than
    # the struct size rather than depending on the remaining platform fields.
    result = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.fstatfs(fd, ctypes.byref(result)):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if ctypes.c_long.from_buffer(result).value != 0x01021994:  # TMPFS_MAGIC
        raise ValueError("Qwen38 shared_thp source cache must be on tmpfs")


def map_shared_source(source_fd, cache_dir, stop):
    """Return a writable THP mapping and cache-hit flag, without a private copy.

    The source pack must remain immutable. Its inode, size and nanosecond change
    times identify the cache. A writer lock and atomic rename keep incomplete
    files invisible to other readers, including after an interrupted startup.
    Cache files intentionally outlive readers; the operator owns their lifetime.
    """
    identity = _identity(source_fd)
    size = identity[2]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _require_tmpfs(directory_fd)
    finally:
        os.close(directory_fd)
    path = directory / f"experts-thp-v1-{key}.bin"
    with open(path.with_suffix(".lock"), "a+b") as lock:
        while True:
            if stop.is_set():
                raise RuntimeError("expert source preparation cancelled")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.1)
        if path.exists() and path.stat().st_size == size:
            with path.open("r+b", buffering=0) as cached:
                mapping = mmap.mmap(cached.fileno(), size, access=mmap.ACCESS_WRITE)
            try:
                mapping.madvise(mmap.MADV_HUGEPAGE)
            except BaseException:
                mapping.close()
                raise
            return mapping, True
        # Remove abandoned, unpublished files from an earlier killed writer for
        # this exact key, while holding its exclusive writer lock.
        for abandoned in directory.glob(path.name + ".*.tmp"):
            abandoned.unlink()
        path.unlink(missing_ok=True)  # A wrong-sized cache is not usable.
        space = os.statvfs(directory)
        if space.f_bavail * space.f_frsize < size:
            raise OSError("Not enough tmpfs space for Qwen38 shared source cache")
        fd, temporary = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=directory
        )
        mapping = None
        try:
            os.ftruncate(fd, size)
            mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            mapping.madvise(mmap.MADV_HUGEPAGE)
            view = memoryview(mapping)
            try:
                offset = 0
                while offset < size:
                    if stop.is_set():
                        raise RuntimeError("expert source preparation cancelled")
                    end = min(offset + (8 << 20), size)
                    part = view[offset:end]
                    try:
                        count = os.preadv(source_fd, [part], offset)
                    finally:
                        part.release()
                    if count <= 0:
                        raise OSError("short expert source read")
                    # The tmpfs cache is now the durable-in-RAM copy. Release
                    # redundant disk page-cache pages progressively, not after
                    # another entire 63 GiB copy has accumulated.
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(
                            source_fd, offset, count, os.POSIX_FADV_DONTNEED
                        )
                    offset += count
            finally:
                view.release()
            if _identity(source_fd) != identity:
                raise RuntimeError(
                    "Expert pack changed while building shared source cache"
                )
            os.replace(temporary, path)
            return mapping, False
        except BaseException:
            if mapping is not None:
                mapping.close()
            raise
        finally:
            os.close(fd)
            Path(temporary).unlink(missing_ok=True)
