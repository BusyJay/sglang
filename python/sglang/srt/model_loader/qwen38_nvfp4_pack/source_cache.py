# SPDX-License-Identifier: Apache-2.0
"""Persistent tmpfs expert sources; CUDA registration still belongs to the reader."""

import ctypes
import errno
import fcntl
import hashlib
import json
import mmap
import os
import tempfile
import weakref
from pathlib import Path

RESOURCE_ERRORS = {
    errno.ENOSPC,
    errno.ENOMEM,
    errno.EACCES,
    errno.EROFS,
    errno.ENODEV,
    errno.EOPNOTSUPP,
    errno.EFAULT,
}


class CacheLocationError(ValueError):
    pass


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
        raise CacheLocationError("Qwen38 shared_thp source cache must be on tmpfs")


def shared_source_path(
    cache_dir, *, source_fd=None, cache_key=None, prefix="experts-thp-v1"
):
    if cache_key is None:
        cache_key = hashlib.sha256(
            json.dumps(_identity(source_fd)).encode()
        ).hexdigest()
    return Path(cache_dir) / f"{prefix}-{cache_key}.bin"


def _lease(mapping, path):
    # Separate from the writer lock: multiple CUDA readers may hold shared
    # leases while another process decides whether an unused cache can be freed.
    fd = os.open(path.with_suffix(".lease"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        weakref.finalize(mapping, os.close, fd)
    except BaseException:
        os.close(fd)
        raise


def remove_unused_source(path, *, remove_cache=True):
    """Remove a cache only while neither a writer nor a mapped reader owns it."""
    path = Path(path)
    with path.with_suffix(".lock").open("a+b") as writer, path.with_suffix(
        ".lease"
    ).open("a+b") as lease:
        try:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if remove_cache:
            path.unlink(missing_ok=True)
        for abandoned in path.parent.glob(path.name + ".*.tmp"):
            abandoned.unlink()
        return True


def clean_cache_directory(directory, expected):
    """This dedicated directory owns its data; retain only matching live inputs.

    No image-version comparisons. Reader leases and writer locks protect active
    data. Empty coordination files remain so waiting processes keep using the
    same lock inode. Do not use a general-purpose tmpfs directory as this cache.
    """
    directory = Path(directory).resolve()
    if directory in (Path("/dev/shm"), Path("/tmp"), Path("/run"), Path("/")):
        raise CacheLocationError(
            "startup cache requires a dedicated tmpfs subdirectory"
        )
    report = dict(removed_files=0, removed_bytes=0, in_use=0)
    with (directory / ".directory.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        candidates = []
        for root, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs[:]:
                path = Path(root) / name
                if path.is_symlink():
                    candidates.append(path)
                    dirs.remove(name)
            candidates.extend(Path(root) / name for name in files)
        for path in candidates:
            if (
                path.suffix in (".lock", ".lease")
                or path.name == ".cache-manifest.json"
            ):
                continue
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            reusable = (
                not path.is_symlink()
                and path.parent == directory
                and expected.get(path.name) == stat.st_size
            )
            # An unpublished temporary belongs to the corresponding writer,
            # even when that writer has not created its final .bin file yet.
            base = (
                path.with_name(path.name.split(".bin.", 1)[0] + ".bin")
                if ".bin." in path.name and path.suffix == ".tmp"
                else path
            )
            if reusable:
                continue
            if remove_unused_source(
                base, remove_cache=(base.name not in expected or base == path)
            ):
                if not path.exists():
                    report["removed_files"] += 1
                    report["removed_bytes"] += stat.st_blocks * 512
            else:
                report["in_use"] += 1
        # Manifest describes reusable inputs, not an image or container version.
        fd, temporary = tempfile.mkstemp(
            dir=directory, prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(expected, handle, sort_keys=True)
            os.replace(temporary, directory / ".cache-manifest.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
    return report


def map_shared_source(
    source_fd, cache_dir, stop, *, cache_key=None, prefix="experts-thp-v1"
):
    """Return a writable THP mapping and cache-hit flag, without a private copy.

    The source pack must remain immutable. Its inode, size and nanosecond change
    times identify the cache. A writer lock and atomic rename keep incomplete
    files invisible to other readers, including after an interrupted startup.
    Cache files intentionally outlive readers; the operator owns their lifetime.
    """
    identity = _identity(source_fd)
    size = identity[2]
    directory = Path(cache_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _require_tmpfs(directory_fd)
    finally:
        os.close(directory_fd)
    path = shared_source_path(
        directory, source_fd=source_fd, cache_key=cache_key, prefix=prefix
    )
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
                _lease(mapping, path)
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
            raise OSError(
                errno.ENOSPC, "Not enough tmpfs space for Qwen38 shared source cache"
            )
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
            _lease(mapping, path)
            return mapping, False
        except BaseException:
            if mapping is not None:
                mapping.close()
            raise
        finally:
            os.close(fd)
            Path(temporary).unlink(missing_ok=True)
