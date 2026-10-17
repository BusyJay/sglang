# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU/NVTX timing; startup memory queries never synchronize CUDA."""
import contextlib
import json
import logging
import os
import resource
import time
import threading

logger = logging.getLogger(__name__)
_counts = {}
_lock = threading.Lock()


def snapshot(device=None):
    result = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmLck:", "RssAnon:", "RssFile:")):
                    key, value, *_ = line.split()
                    result[key.rstrip(":") + "_bytes"] = int(value) * 1024
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(minflt=usage.ru_minflt, majflt=usage.ru_majflt)
        if device is not None:
            import torch
            free, total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            result.update(driver_free_bytes=free, driver_total_bytes=total,
                          torch_allocated_bytes=torch.cuda.memory_allocated(device),
                          torch_reserved_bytes=reserved,
                          driver_used_minus_reserved_bytes=total-free-reserved)
    except (OSError, RuntimeError) as error:
        result["snapshot_error"] = str(error)
    return result


@contextlib.contextmanager
def observe(stage, *, device=None, startup=False, **metadata):
    enabled = os.getenv("SGLANG_QWEN38_OBSERVE", "1") == "1"
    if not enabled:
        yield
        return
    import torch
    nvtx = os.getenv("SGLANG_QWEN38_OBSERVE_NVTX", "0") == "1"
    before = snapshot(device) if startup else None
    start = time.perf_counter()
    if nvtx:
        torch.cuda.nvtx.range_push("qwen38:" + stage)
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        if nvtx:
            torch.cuda.nvtx.range_pop()
        row = dict(stage=stage, host_seconds=time.perf_counter()-start,
                   pid=os.getpid(), metadata=metadata, error=error)
        if startup:
            row.update(before=before, after=snapshot(device))
        with _lock:
            count, total = _counts.get(stage, (0, 0.0))
            count, total = count+1, total+row["host_seconds"]
            _counts[stage] = count, total
        row.update(calls=count, cumulative_host_seconds=total)
        if startup or error or count == 1 or count % 256 == 0:
            logger.info("Qwen38Observe %s", json.dumps(row, sort_keys=True))
