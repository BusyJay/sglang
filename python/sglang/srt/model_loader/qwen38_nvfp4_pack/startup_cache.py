# SPDX-License-Identifier: Apache-2.0
"""Choose persistent host caches before loading weights or allocating host pools."""

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
GIB = 1 << 30


def choose_cache_tier(expert_bytes, ple_bytes, budget):
    """The order is intentional: both, experts, PLE, neither."""
    for experts, ple in ((True, True), (True, False), (False, True), (False, False)):
        required = expert_bytes * experts + ple_bytes * ple
        if required <= budget:
            return experts and expert_bytes > 0, ple and ple_bytes > 0


def memory_headroom():
    info = {
        key: int(value.split()[0]) * 1024
        for key, value in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    total, available = info["MemTotal"], info["MemAvailable"]
    cgroup_headroom = None
    # cgroup v2's mount is namespaced to this container. Do not count reclaimable
    # tmpfs pages as free memory; inactive_file is disk-backed cache.
    try:
        root = Path("/sys/fs/cgroup")
        limit = (root / "memory.max").read_text().strip()
        if limit != "max":
            limit = int(limit)
            used = int((root / "memory.current").read_text())
            stats = dict(
                line.split() for line in (root / "memory.stat").read_text().splitlines()
            )
            total = min(total, limit)
            cgroup_headroom = max(0, limit - used + int(stats.get("inactive_file", 0)))
    except (OSError, ValueError):
        pass
    return total, available, cgroup_headroom


def prepare_startup_cache(model, manifest, weight_files):
    """Set each PLE's shared-cache descriptor and return the expert source mode.

    Only persistent copies are selected here; non-selected components retain
    their existing loading backend. This is not an admission check for the
    model's entire host footprint (which also includes HiCache and other pools).
    """
    from sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache import (
        CacheLocationError,
        _require_tmpfs,
        clean_cache_directory,
        remove_unused_source,
        shared_source_path,
    )

    directory = (
        Path(
            os.getenv(
                "SGLANG_QWEN38_SOURCE_CACHE_DIR", "/dev/shm/sglang/qwen38-experts"
            )
        )
        .expanduser()
        .resolve()
    )
    # Metadata only, never read checkpoint tensor payloads. Include every source
    # file identity and local table geometry in each final-row cache key.
    checkpoint = []
    for name in weight_files:
        stat = os.stat(name)
        checkpoint.append(
            (
                os.path.realpath(name),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    tables = []
    for name, module in model.named_modules():
        if not (getattr(module, "_fp4_ple", False) and module._backend == "pinned"):
            continue
        if os.getenv("SGLANG_QWEN38_PLE_PACKED_CACHE", "1") != "1":
            continue
        module._shared_cache_spec = None
        shape = (
            int(module.weight.shape[0]),
            module.embedding_dim // 2 + module.embedding_dim // 16,
        )
        encoded = json.dumps(
            ("packed-ple-v1", sorted(item[1:] for item in checkpoint), name, shape),
            sort_keys=True,
        )
        key = hashlib.sha256(encoded.encode()).hexdigest()
        path = shared_source_path(directory, cache_key=key, prefix="ple-thp-v1")
        tables.append((module, key, path, shape[0] * shape[1]))
    with manifest.payload_path.open("rb") as source:
        expert_path = shared_source_path(directory, source_fd=source.fileno())
    sources = [(expert_path, manifest.payload_nbytes)] + [
        (path, size) for _, _, path, size in tables
    ]
    ple_bytes = sum(size for _, _, _, size in tables)
    expert_mode = "anonymous_thp"
    report = dict(
        tier="none", expert_bytes=manifest.payload_nbytes, ple_bytes=ple_bytes
    )
    try:
        if any(
            Path(name).absolute().is_relative_to(directory)
            or Path(name).resolve().is_relative_to(directory)
            for name in [manifest.payload_path, *weight_files]
        ):
            raise CacheLocationError(
                "dedicated tmpfs cache directory must not contain source model files"
            )
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _require_tmpfs(fd)
        finally:
            os.close(fd)
        report["cleanup"] = clean_cache_directory(
            directory, {path.name: size for path, size in sources}
        )
        reusable = 0
        for path, size in sources:
            try:
                stat = path.stat()
                if stat.st_size == size:
                    reusable += min(stat.st_blocks * 512, size)
            except FileNotFoundError:
                pass
        total, available, cgroup = memory_headroom()
        reserve = int(
            float(
                os.getenv(
                    "SGLANG_QWEN38_STARTUP_CACHE_RESERVE_GIB",
                    str(max(8, total / GIB * 0.2)),
                )
            )
            * GIB
        )
        if reserve < 0:
            raise ValueError("startup cache reserve must be nonnegative")
        ram_budget = max(0, available + reusable - reserve)
        if cgroup is not None:
            ram_budget = min(ram_budget, max(0, cgroup + reusable - reserve))
        space = os.statvfs(directory)
        tmpfs_budget = space.f_bavail * space.f_frsize + reusable
        budget = min(ram_budget, tmpfs_budget)
        limit = os.getenv("SGLANG_QWEN38_STARTUP_CACHE_MAX_GIB")
        if limit is not None:
            limit = int(float(limit) * GIB)
            if limit < 0:
                raise ValueError("startup cache maximum must be nonnegative")
            budget = min(budget, limit)
        experts, ple = choose_cache_tier(manifest.payload_nbytes, ple_bytes, budget)
        report.update(
            tier=(
                "both"
                if experts and ple
                else "experts" if experts else "ple" if ple else "none"
            ),
            available_bytes=available,
            reusable_bytes=reusable,
            reserve_bytes=reserve,
            ram_budget=ram_budget,
            tmpfs_budget=tmpfs_budget,
            budget=budget,
        )
        if experts:
            expert_mode = "shared_thp"
        for module, key, _, _ in tables:
            if ple:
                module._shared_cache_spec = (str(directory), key, checkpoint)
        unused = ([] if experts else [expert_path]) + (
            [] if ple else [path for _, _, path, _ in tables]
        )
        for path in unused:
            if path.exists() and not remove_unused_source(path):
                logger.info("Qwen38 startup cache still in use; retaining %s", path)
    except (OSError, CacheLocationError) as exc:
        # Missing/readonly/non-tmpfs cache locations must not break the existing
        # model-loading path. Invalid numeric overrides are configuration errors.
        expert_mode = "anonymous_thp"
        for module, _, _, _ in tables:
            module._shared_cache_spec = None
        report.update(tier="none", reason=str(exc))
    logger.info("Qwen38 startup cache plan: %s", json.dumps(report, sort_keys=True))
    model._qwen38_startup_cache_plan = report
    return expert_mode
