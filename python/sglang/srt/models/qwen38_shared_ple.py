# SPDX-License-Identifier: Apache-2.0
"""One tmpfs PLE mapping registered for direct GPU reads, without a pinned copy."""

import gc
import logging
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)


class SharedPleTable:
    def __init__(self, path, directory, key, device):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module
        from sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache import (
            map_shared_source,
        )

        self.mapping = self.raw = self.pointer = None
        self.device = torch.device(device)
        started = time.perf_counter()
        with open(path, "rb", buffering=0) as source:
            mapping, hit = map_shared_source(
                source.fileno(),
                directory,
                threading.Event(),
                cache_key=key,
                prefix="ple-thp-v1",
            )
        prepare_seconds = time.perf_counter() - started
        raw = torch.frombuffer(mapping, dtype=torch.uint8)
        started = time.perf_counter()
        try:
            with torch.cuda.device(self.device):
                pointer = _module().mailbox_register(raw)
        except BaseException:
            del raw
            try:
                mapping.close()
            except BufferError:
                # A CUDA wrapper's exception traceback can still own its tensor
                # argument. Preserve the registration error; the last view will
                # release the mapping/lease once the caller clears the traceback.
                pass
            raise
        self.mapping, self.raw, self.pointer = mapping, raw, pointer
        logger.info(
            "Qwen38 startup shared PLE ready: GiB=%.2f cache_hit=%s prepare_seconds=%.3f register_seconds=%.3f",
            raw.nbytes / (1 << 30),
            hit,
            prepare_seconds,
            time.perf_counter() - started,
        )

    def close(self):
        """All external tensor aliases must be released by the embedding first."""
        if self.mapping is None:
            return
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module

        if self.pointer is not None:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                _module().mailbox_unregister(self.raw)
            self.pointer = None
        self.raw = None
        self.mapping.close()
        self.mapping = None


def try_shared_ple(path, spec, device):
    from sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache import (
        RESOURCE_ERRORS,
        remove_unused_source,
        shared_source_path,
    )

    directory, key, checkpoint = spec
    for name, dev, ino, size, mtime, ctime in checkpoint:
        stat = os.stat(name)
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ) != (dev, ino, size, mtime, ctime):
            raise ValueError("PLE checkpoint changed after startup cache selection")
    try:
        return SharedPleTable(path, directory, key, device)
    except OSError as exc:
        if exc.errno not in RESOURCE_ERRORS:
            raise
        logger.warning(
            "Qwen38 shared PLE unavailable; using pinned table: %s", str(exc)
        )
    except RuntimeError as exc:
        if "register expert mmap" not in str(exc):
            raise
        logger.warning(
            "Qwen38 shared PLE registration unavailable; using pinned table: %s",
            str(exc),
        )
    # Release exception traceback references/leases before reclaiming a completed
    # cache whose registration failed. Otherwise fallback would keep two tables.
    gc.collect()
    cached = shared_source_path(directory, cache_key=key, prefix="ple-thp-v1")
    if cached.exists():
        remove_unused_source(cached)
    return None
