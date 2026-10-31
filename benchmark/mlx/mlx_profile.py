# SPDX-License-Identifier: Apache-2.0
"""Bounded host instrumentation for MLX. Host spans are never GPU durations.

Importable by a serving replay as well as profile_glm_mlx.py. Hooks are scoped
to a single-threaded diagnostic run; concurrent inference must use a separate
process. Nothing reads array values, prompts, or weights into the trace.
"""

import collections
import contextlib
import functools
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path
from unittest.mock import patch


def arrays(value, mx, depth=0):
    if isinstance(value, mx.array):
        yield value
    elif hasattr(value, "named_modules"):
        return
    elif depth < 5:
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from arrays(item, mx, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                yield from arrays(item, mx, depth + 1)
        elif hasattr(value, "cache"):
            yield from arrays(value.cache, mx, depth + 1)


def describe(value, mx):
    # Record descriptors, never retain arrays and their computational graphs.
    return [
        dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes)
        for a in arrays(value, mx)
    ][:32]


class Recorder:
    def __init__(self, mx, mode="host", max_events=200000, graph_dir=None):
        self.mx = mx
        self.mode = mode
        self.max_events = max_events
        self.events = []
        self.dropped = 0
        self.origin = time.perf_counter_ns()
        self.stack = []
        self.owner = threading.get_ident()
        self.graph_dir = Path(graph_dir) if graph_dir else None
        self.graph_keys = set()
        if self.graph_dir:
            self.graph_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def span(self, name, category="host", **args):
        if threading.get_ident() != self.owner:
            raise RuntimeError("MLX diagnostic hooks require a single-threaded replay")
        start = time.perf_counter_ns()
        frame = [start, 0]
        self.stack.append(frame)
        try:
            yield args
        except BaseException as exc:
            args["exception"] = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter_ns() - start
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            event = dict(
                name=name,
                cat=category,
                ph="X",
                pid=os.getpid(),
                tid=self.owner,
                ts=(start - self.origin) / 1000,
                dur=elapsed / 1000,
                args=dict(args, self_us=max(0, elapsed - frame[1]) / 1000),
            )
            if len(self.events) < self.max_events:
                self.events.append(event)
            else:
                self.dropped += 1

    def wrap(self, name, original, *, category="host", synchronize=False, label=None):
        @functools.wraps(original)
        def call(*values, **kwargs):
            detail = dict(inputs=describe((values, kwargs), self.mx))
            if label and values:
                detail["module"] = label.get(id(values[0]), "unmapped")
            if synchronize and self.mode == "sync":
                with self.span(name + ".drain_before", "diagnostic_sync"):
                    self.mx.eval(list(arrays((values, kwargs), self.mx)))
                    self.mx.synchronize()
            with self.span(name, category, **detail) as event:
                result = original(*values, **kwargs)
                event["outputs"] = describe(result, self.mx)
                if self.graph_dir and category in ("module", "qmm", "fusion"):
                    key = (name, detail.get("module"), json.dumps(event["outputs"]))
                    if key not in self.graph_keys and len(self.graph_keys) < 64:
                        self.graph_keys.add(key)
                        outputs = list(arrays(result, self.mx))[:4]
                        if outputs:
                            filename = f"{len(self.graph_keys):02d}-{name.replace('.', '_')}.dot"
                            self.mx.export_to_dot(
                                str(self.graph_dir / filename), *outputs
                            )
                            event["graph"] = filename
                if category == "fusion":
                    event["returned_none"] = result is None
                if synchronize and self.mode == "sync":
                    with self.span(name + ".drain_after", "diagnostic_sync"):
                        self.mx.eval(list(arrays((result, values, kwargs), self.mx)))
                        self.mx.synchronize()
                return result

        return call

    @contextlib.contextmanager
    def instrument(self, model_module, model, serving_class=None):
        """Restores every hook even if the replay raises; no production flags."""
        mx = self.mx
        labels = {id(m): path for path, m in model.named_modules()}
        with contextlib.ExitStack() as stack:
            for cls_name, methods in {
                "MLP": ("__call__",),
                "MoE": ("__call__",),
                "ClampedSwitchGLU": ("__call__",),
                "KDA": ("__call__", "project_verify_rows"),
                "MLA": ("__call__", "_latent_matrices"),
                "DecoderLayer": ("__call__", "verify_ffn_rows", "_branch_output"),
                "MTPHead": ("__call__",),
                "Model": ("forward_with_hidden", "verify_mtp_rows"),
            }.items():
                cls = getattr(model_module, cls_name)
                for method in methods:
                    stack.enter_context(
                        patch.object(
                            cls,
                            method,
                            self.wrap(
                                cls_name + "." + method,
                                getattr(cls, method),
                                category="module",
                                label=labels,
                                synchronize=method
                                not in ("_branch_output", "_latent_matrices"),
                            ),
                        )
                    )
            for name in (
                "mlx_hc_collapse_norm",
                "mlx_kda_decode",
                "mlx_glm_router",
                "mlx_glm_down_combine",
                "mlx_affine_rows",
                "mlx_dense_rows",
                "_compiled_quantized_ffn",
                "_batch_quantized_ffn",
            ):
                if hasattr(model_module, name):
                    stack.enter_context(
                        patch.object(
                            model_module,
                            name,
                            self.wrap(
                                name,
                                getattr(model_module, name),
                                category="fusion",
                            ),
                        )
                    )
            for name in ("eval", "async_eval", "synchronize", "clear_cache"):
                stack.enter_context(
                    patch.object(
                        mx,
                        name,
                        self.wrap(
                            "mx." + name,
                            getattr(mx, name),
                            category="submission_wait",
                        ),
                    )
                )
            for name in ("quantized_matmul", "gather_qmm"):
                stack.enter_context(
                    patch.object(
                        mx,
                        name,
                        self.wrap(
                            "mx." + name,
                            getattr(mx, name),
                            category="qmm",
                        ),
                    )
                )
            if serving_class is not None:
                for name in ("prefill", "decode", "_flush_head", "_publish"):
                    wrapped = self.wrap(
                        "serving." + name,
                        getattr(serving_class, name),
                        category="serving",
                    )
                    if isinstance(serving_class.__dict__.get(name), staticmethod):
                        wrapped = staticmethod(wrapped)
                    stack.enter_context(patch.object(serving_class, name, wrapped))
            yield self

    def save(self, path):
        document = dict(
            traceEvents=sorted(self.events, key=lambda e: e["ts"]),
            displayTimeUnit="ms",
            mode=self.mode,
            dropped_events=self.dropped,
            evidence="CPU wall-clock spans; NOT GPU kernel timing",
        )
        Path(path).write_text(json.dumps(document) + "\n")

    def summary(self):
        return summarize(self.events)


def summarize(events):
    groups = collections.defaultdict(list)
    dispatch = collections.Counter()
    for event in events:
        groups[(event["cat"], event["name"], event["args"].get("module", ""))].append(
            event
        )
        if event["cat"] in ("fusion", "qmm"):
            dispatch[
                json.dumps(
                    dict(
                        op=event["name"],
                        inputs=event["args"].get("inputs", []),
                        outputs=event["args"].get("outputs", []),
                        returned_none=event["args"].get("returned_none"),
                    ),
                    sort_keys=True,
                )
            ] += 1
    rows = []
    for (category, name, module), group in groups.items():
        rows.append(
            dict(
                category=category,
                name=name,
                module=module,
                calls=len(group),
                inclusive_ms=sum(e["dur"] for e in group) / 1000,
                self_ms=sum(e["args"]["self_us"] for e in group) / 1000,
                median_ms=statistics.median(e["dur"] for e in group) / 1000,
            )
        )
    return dict(
        timings=sorted(rows, key=lambda r: r["self_ms"], reverse=True),
        dispatch=[
            dict(json.loads(key), calls=count) for key, count in dispatch.items()
        ],
    )


def source_manifest(root):
    files = [
        "benchmark/mlx/mlx_profile.py",
        "benchmark/mlx/profile_glm_mlx.py",
        "benchmark/mlx/mlx_metal_trace.py",
        "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
        "python/sglang/srt/hardware_backend/mlx/mtp_serving.py",
        "python/sglang/srt/hardware_backend/mlx/mtp.py",
    ]
    files += [
        str(p.relative_to(root))
        for p in (root / "python/sglang/kernels/ops").rglob("*mlx*.py")
    ]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}
