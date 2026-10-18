"""Read-only Python ownership/VMM/driver sidecars for PyTorch memory profiles."""

import collections
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def inventory():
    import torch

    objects = gc.get_objects()
    tensor_type = torch.Tensor
    roots = {}
    for obj in objects:
        cls = type(obj)
        if isinstance(cls.__module__, str) and cls.__module__.startswith("sglang.") and cls.__name__ in (
            "ModelRunner", "Scheduler", "KvVmmArena"
        ):
            roots[f"{cls.__name__}[{id(obj)}]"] = obj
    from sglang.srt.runtime_context import get_resources

    roots["runtime_resources"] = get_resources()
    for name, module in list(sys.modules.items()):
        if name.startswith(("flashinfer.", "sglang.")) and module is not None:
            roots[f"module.{name}"] = vars(module)
    # SGLang freezes startup objects, so gc.get_objects alone misses the model.
    # The existing MEM control request runs on Scheduler's stack at idle.
    # Do not materialize our own f_locals: it would retain this frame and its
    # gc.get_objects() list in a cycle while the server has cyclic GC frozen.
    # Inspect only callers, whose locals contain the scheduler control request.
    frame = sys._getframe(2)
    while frame is not None:
        for key, value in list(frame.f_locals.items()):
            cls = type(value)
            if isinstance(cls.__module__, str) and cls.__module__.startswith("sglang.") and cls.__name__ == "Scheduler":
                roots[f"stack.{key}.{cls.__name__}[{id(value)}]"] = value
        frame = frame.f_back
    del frame
    tensors, seen, arenas, skipped = {}, set(), [], []
    queue = collections.deque(roots.items())

    def add_tensor(path, tensor):
        if tensor.device.type == "meta":
            return
        try:
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
        except (RuntimeError, NotImplementedError) as exc:
            skipped.append(dict(owner=path, type=str(type(tensor)), reason=str(exc)))
            return
        key = (str(tensor.device), pointer, storage.nbytes())
        row = tensors.setdefault(key, dict(
            device=str(tensor.device), pointer=storage.data_ptr(),
            bytes=storage.nbytes(), aliases=[]
        ))
        row["aliases"].append(dict(
            name=path, shape=list(tensor.shape), dtype=str(tensor.dtype),
            offset=tensor.storage_offset(), numel=tensor.numel(),
            element_size=tensor.element_size(), tensor_id=id(tensor),
        ))

    while queue:
        path, obj = queue.popleft()
        if issubclass(type(obj), tensor_type):
            # Record every direct owner even when the Tensor itself is shared.
            add_tensor(path, obj)
            seen.add(id(obj))
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = obj.items()
        elif isinstance(obj, (list, tuple, collections.deque)):
            children = enumerate(obj)
        elif isinstance(type(obj).__module__, str) and type(obj).__module__.startswith(("sglang.", "torch.", "flashinfer.")):
            attrs = dict(getattr(obj, "__dict__", {}))
            for cls in type(obj).__mro__:
                slots = cls.__dict__.get("__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for key in slots:
                    if key not in ("__dict__", "__weakref__") and hasattr(obj, key):
                        attrs.setdefault(key, getattr(obj, key))
            children = attrs.items()
            if type(obj).__name__ == "KvVmmArena":
                arenas.append(dict(
                    owner=path, base=obj.base, reserved=obj.reserved,
                    backed=obj.backed_bytes, granularity=obj.granularity,
                    ranges=dict(obj._committed_by_offset),
                    pool_id=list(obj.pool.id), closed=obj._closed,
                ))
        else:
            continue
        for key, value in children:
            # Workspace caches use tuple/device/dtype keys. Never call arbitrary
            # repr(): a tensor key could synchronize or materialize GPU data.
            label = key if isinstance(key, (str, int)) else f"key[{type(key).__name__}:{id(key)}]"
            queue.append((f"{path}.{label}", value))
    # A C++ owner can hide a tensor from the runner object graph. Preserve that
    # distinction; do not silently declare Python reachability to be complete.
    for obj in objects:
        if issubclass(type(obj), tensor_type) and id(obj) not in seen:
            add_tensor(f"gc_unrooted.{type(obj).__name__}[{id(obj)}]", obj)
    return dict(storages=list(tensors.values()), arenas=arenas,
                visited_objects=len(seen), roots=list(roots), skipped=skipped)


def capture_sidecar(path):
    import torch

    torch.cuda.synchronize()
    data = inventory()
    free, total = torch.cuda.mem_get_info()
    data.update(pid=os.getpid(), time_ns=time.time_ns(),
                driver_free=free, driver_total=total,
                memory_stats=dict(torch.cuda.memory_stats()))
    data["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
         "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout
    Path(str(path) + ".owners.json").write_text(json.dumps(data) + "\n")


def install():
    import torch

    root = Path(os.environ["QWEN38_AUDIT_OUTPUT"])
    dump = torch.cuda.memory._dump_snapshot
    record = torch.cuda.memory._record_memory_history

    def wrapped_dump(filename="dump_snapshot.pickle"):
        try:
            capture_sidecar(filename)
        except Exception as exc:
            Path(str(filename) + ".owners-error.txt").write_text(repr(exc))
        return dump(filename)

    def wrapped_record(*args, **kwargs):
        # 74K/170K runs exceed the serving profiler's default 100K ring.
        # Keep complete alloc/free histories for these controlled captures.
        if kwargs.get("enabled", args[0] if args else "all") is not None:
            kwargs["max_entries"] = 2000000
        result = record(*args, **kwargs)
        if kwargs.get("enabled", args[0] if args else "all") is not None:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"start-{os.getpid()}-{time.time_ns()}.pickle"
            try:
                capture_sidecar(path)
            except Exception as exc:
                Path(str(path) + ".owners-error.txt").write_text(repr(exc))
            dump(str(path))
        return result

    torch.cuda.memory._dump_snapshot = wrapped_dump
    torch.cuda.memory._record_memory_history = wrapped_record
