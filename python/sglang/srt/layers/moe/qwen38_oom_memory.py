# SPDX-License-Identifier: Apache-2.0
"""Opt-in allocator history and crash-time evidence for offload diagnosis."""

import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)
_installed = set()


def install_oom_memory_observer(device):
    directory = os.getenv("SGLANG_QWEN38_OOM_MEMORY_DIR")
    if not directory:
        return
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return
    device = torch.device(
        "cuda", torch.cuda.current_device() if device.index is None else device.index
    )
    key = (os.getpid(), str(device))
    if key in _installed:
        return
    root = Path(directory)
    if not root.is_absolute():
        raise ValueError("SGLANG_QWEN38_OOM_MEMORY_DIR must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    entries = int(os.getenv("SGLANG_QWEN38_OOM_MEMORY_ENTRIES", "500000"))
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=entries,
    )

    def on_oom(device_index, requested, device_total, free):
        stem = root / f"oom-pid{os.getpid()}-{time.time_ns()}"
        metadata = {
            "pid": os.getpid(),
            "device": device_index,
            "requested_bytes": requested,
            "device_total_bytes": device_total,
            "device_free_bytes": free,
            "history_entries": entries,
        }
        try:
            torch.cuda.memory._dump_snapshot(str(stem) + ".pickle")
            metadata["snapshot"] = str(stem) + ".pickle"
        except Exception as exc:
            metadata["snapshot_error"] = repr(exc)
        Path(str(stem) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.error("Qwen38 OOM allocator evidence: %s", stem)

    torch._C._cuda_attach_out_of_memory_observer(on_oom)
    _installed.add(key)
    logger.info("Qwen38 OOM allocator history enabled: %s (%s entries)", root, entries)
