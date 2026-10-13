# SPDX-License-Identifier: Apache-2.0
"""Opt-in startup-only CUDA allocator evidence; never used for performance runs.

Set SGLANG_QWEN38_STARTUP_MEMORY_DIR to an absolute writable output directory.
Samples do not empty the cache; explicit boundary pairs capture its actual effect.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import pickle
import time

logger = logging.getLogger(__name__)
_PROBES = {}


def get_startup_memory_probe(device):
    directory = os.getenv("SGLANG_QWEN38_STARTUP_MEMORY_DIR")
    if not directory:
        return None
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return None
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (os.getpid(), str(device), directory)
    if key not in _PROBES:
        _PROBES[key] = StartupMemoryProbe(device, directory)
    return _PROBES[key]


def memory_mark(probe, stage, **kwargs):
    if probe is not None:
        probe.mark(stage, **kwargs)


def storage_inventory(model):
    """Deduplicate underlying storage, retain aliases and unused-module names."""
    storages, non_cuda = {}, {"meta_tensors": 0}
    for kind, values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in values:
            if tensor.device.type == "meta":
                non_cuda["meta_tensors"] += 1
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages[key] = {
                    "device": str(tensor.device),
                    "pointer": storage.data_ptr(),
                    "bytes": storage.nbytes(),
                    "aliases": [],
                }
            storages[key]["aliases"].append(
                {
                    "name": name,
                    "kind": kind,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
    rows = list(storages.values())
    return {
        "storages": rows,
        "cuda_storage_bytes": sum(
            r["bytes"] for r in rows if r["device"].startswith("cuda")
        ),
        "cpu_storage_bytes": sum(
            r["bytes"] for r in rows if r["device"].startswith("cpu")
        ),
        "meta_tensors": non_cuda["meta_tensors"],
        "unused_name_candidates": [
            r
            for r in rows
            if any("mtp" in a["name"] or "visual" in a["name"] for a in r["aliases"])
        ],
    }


class StartupMemoryProbe:
    def __init__(self, device, directory):
        import torch

        directory = Path(directory)
        if not directory.is_absolute():
            raise ValueError("SGLANG_QWEN38_STARTUP_MEMORY_DIR must be absolute")
        directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.root = directory / f"startup-memory-pid{os.getpid()}-{time.time_ns()}"
        self.root.mkdir()
        self.sequence = 0
        self.previous = None
        self.history_status = "not_available"
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="all", max_entries=200000
            )
            self.history_status = "enabled (bounded 200000 entries)"
        except (AttributeError, TypeError) as exc:
            self.history_status = f"unavailable: {type(exc).__name__}: {exc}"
        self.events = self.root / "events.jsonl"
        self.mark(
            "probe.created",
            metadata={
                "allocator_history": self.history_status,
                "allocator_config": {
                    k: os.getenv(k)
                    for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
                },
                "scope": "startup diagnosis, synchronizes CUDA, snapshots excluded from benchmarks",
            },
        )

    def mark(self, stage, *, model=None, metadata=None):
        import torch

        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        stats = torch.cuda.memory_stats(self.device)
        keys = (
            "allocated_bytes.all.current",
            "active_bytes.all.current",
            "reserved_bytes.all.current",
            "inactive_split_bytes.all.current",
            "allocation.all.current",
            "active.all.current",
            "segment.all.current",
            "num_alloc_retries",
            "num_ooms",
        )
        values = {k: stats.get(k, 0) for k in keys}
        values.update(driver_free_bytes=free, driver_total_bytes=total)
        values["driver_used_minus_torch_reserved_bytes"] = (
            total - free - values["reserved_bytes.all.current"]
        )
        stem = f'{self.sequence:04d}-{stage.replace("/","_")}'
        row = {
            "sequence": self.sequence,
            "stage": stage,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "device": str(self.device),
            "device_index": getattr(self.device, "index", None),
            "values": values,
            "delta": (
                {k: v - self.previous[k] for k, v in values.items()}
                if self.previous
                else None
            ),
            "metadata": metadata or {},
            "snapshot": stem + ".pickle",
        }
        # _snapshot includes allocation stacks/history when supported. Keep a
        # portable JSON segment view and a pickle compatible with memory_viz.
        try:
            snapshot = torch.cuda.memory._snapshot()
        except AttributeError:
            snapshot = {"segments": torch.cuda.memory_snapshot()}
        with (self.root / row["snapshot"]).open("wb") as stream:
            pickle.dump(snapshot, stream)
        (self.root / (stem + ".segments.json")).write_text(
            json.dumps(snapshot.get("segments", []), default=str) + "\n"
        )
        if model is not None:
            inventory = storage_inventory(model)
            row["inventory"] = stem + ".inventory.json"
            row["model_cuda_storage_bytes"] = inventory["cuda_storage_bytes"]
            (self.root / row["inventory"]).write_text(
                json.dumps(inventory, indent=2) + "\n"
            )
        with self.events.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        logger.info(
            "Qwen38StartupMemory stage=%s free=%.2fMiB allocated=%.2fMiB active=%.2fMiB reserved=%.2fMiB inactive_split=%.2fMiB evidence=%s",
            stage,
            free / 2**20,
            values["allocated_bytes.all.current"] / 2**20,
            values["active_bytes.all.current"] / 2**20,
            values["reserved_bytes.all.current"] / 2**20,
            values["inactive_split_bytes.all.current"] / 2**20,
            self.events,
        )
        self.previous = values
        self.sequence += 1
