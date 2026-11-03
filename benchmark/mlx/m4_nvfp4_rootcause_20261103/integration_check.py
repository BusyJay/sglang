"""Exercise the exact original/patched ClampedSwitchGLU and NVFP4 classes.

AST isolation bypasses unavailable serving dependencies only; model methods,
native quantized projections and the new default branch execute on Metal.
"""

import ast
import json
import sys
import time
import types
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from common import OUT, ROOT, bank, routing


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


archive = ROOT / "benchmark/mlx/throughput_20261103/baseline/python/sglang/srt/hardware_backend/mlx/models"
ns = {"mx": mx, "nn": nn}
extract(archive / "glm5_fp4.py", {"NVFP4SwitchLinear"}, ns)
Projection = ns["NVFP4SwitchLinear"]
for name in ("sglang.srt", "sglang.srt.hardware_backend", "sglang.srt.hardware_backend.mlx",
             "sglang.srt.hardware_backend.mlx.models"):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
module = types.ModuleType("sglang.srt.hardware_backend.mlx.models.glm5_fp4")
module.NVFP4SwitchLinear = Projection
sys.modules[module.__name__] = module


def classes(path):
    namespace = {"mx": mx, "nn": nn, "SwitchGLU": SwitchGLU}
    extract(path, {"clamped_swiglu", "ClampedSwiGLU", "ClampedSwitchGLU"}, namespace)
    return namespace["ClampedSwitchGLU"]


Original = classes(archive / "glm5_next.py")
Patched = classes(OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py")
mx.random.seed(929)
mx.set_memory_limit(2 * 2**30)
mx.set_cache_limit(128 * 2**20)
ids, e = routing(8)
args = SimpleNamespace(hidden_size=4096, moe_intermediate_size=2048, n_routed_experts=e, swiglu_limit=7.0)
old, new = Original(args), Patched(args)
for name, n, k in (("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)):
    p = bank(e, n, k)
    layer = Projection(p.weight, p.scales, p.weight_scale_2)
    setattr(old, name, layer)
    setattr(new, name, layer)
records = []
for dtype in (mx.float32, mx.bfloat16):
    for count in (1, 4, 8, 9):
        route, _ = routing(count)
        x = mx.random.normal((1, count, 4096)).astype(dtype)
        a, b = old(x, route), new(x, route)
        mx.eval(a, b)
        assert mx.array_equal(a, b).item(), (count, dtype)
        rec = {"count": count, "dtype": str(dtype), "exact": True,
               "path": "native" if count in (1, 9) else "paired"}
        records.append(rec)
        print("INTEGRATION", rec, flush=True)
for count in (1, 4):
    route, _ = routing(count)
    x = mx.random.normal((1, count, 4096))
    scores = mx.full((1, count, 8, 1), 1 / 8)
    fns = {"baseline": mx.compile(lambda x: x + mx.sum(old(x, route) * scores, axis=-2)),
           "patched": mx.compile(lambda x: x + mx.sum(new(x, route) * scores, axis=-2))}
    final = None
    for mode in ("baseline", "patched", "patched", "baseline"):
        value = x
        mx.eval(fns[mode](value))
        raw = []
        for _ in range(300):
            t = time.perf_counter()
            value = fns[mode](value)
            mx.eval(value)
            raw.append((time.perf_counter() - t) * 1000)
        if final is None:
            final = value
        assert mx.array_equal(value, final).item(), (count, mode)
        rec = {"count": count, "dtype": "mlx.core.float32", "mode": mode,
               "steps": 300, "mean_ms": sum(raw) / len(raw), "raw_ms": raw, "final_exact": True}
        records.append(rec)
        (OUT / "integration_check.json").write_text(json.dumps(records, indent=2))
        print("INTEGRATION300", {k: v for k, v in rec.items() if k != "raw_ms"}, flush=True)
