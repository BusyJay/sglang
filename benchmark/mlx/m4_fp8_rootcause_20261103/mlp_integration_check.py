"""Exercise original and patched remote MLP classes with their FP8Linear.

AST isolation removes unavailable server imports. The actual MLP branch and
FP8Linear methods remain unchanged; checkpoint tensors are synthetic.
"""

import ast
import json
import statistics
import sys
import time
import types
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from common import OUT, ROOT, baseline


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


archived = ROOT / "benchmark/mlx/throughput_20261103/baseline/python/sglang/srt/hardware_backend/mlx/models"
namespace = {"nn": nn, "mx": mx,
             "mlx_fp8_dynamic_32": baseline.mlx_fp8_dynamic_32,
             "mlx_fp8_rows": baseline.mlx_fp8_rows}
FP8Linear = extract(archived / "glm5_fp4.py", "FP8Linear", namespace)
for name in ("sglang.srt", "sglang.srt.hardware_backend", "sglang.srt.hardware_backend.mlx",
             "sglang.srt.hardware_backend.mlx.models"):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
mod = types.ModuleType("sglang.srt.hardware_backend.mlx.models.glm5_fp4")
mod.FP8Linear = FP8Linear
sys.modules[mod.__name__] = mod


def clamped(gate, up, limit):
    return nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)


ns = {"nn": nn, "mx": mx, "clamped_swiglu": clamped}
Original = extract(archived / "glm5_next.py", "MLP", dict(ns))
Patched = extract(OUT / "integration_check/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py", "MLP", dict(ns))
mx.set_cache_limit(256 * 2**20)
mx.random.seed(855)
args = SimpleNamespace(hidden_size=4096, intermediate_size=2048, swiglu_limit=7.0)
old, new = Original(args), Patched(args)
for name, n, k in (("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)):
    weight = mx.random.randint(0, 126, (n, k), dtype=mx.uint8)
    scales = mx.full(((n+127)//128, k//128), .0001, mx.float32)
    mx.eval(weight, scales)
    layer = FP8Linear(weight, scales, block_size=128, activation_mode="dynamic32")
    setattr(old, name, layer)
    setattr(new, name, layer)
checks = []
for dtype in (mx.float32, mx.bfloat16):
    for count in (1, 4, 8):
        a = mx.random.normal((1, count, 4096)).astype(dtype)
        b = a
        for step in range(300):
            a = a + old(a)
            b = b + new(b)
            mx.eval(a, b)
        assert mx.array_equal(a, b).item(), (str(dtype), count)
        checks.append({"dtype": str(dtype), "count": count, "steps": 300, "exact": True})
        print("ACTUAL_MLP300_EXACT", checks[-1], flush=True)
for projection in (old.gate_proj, old.up_proj):
    projection.bias = mx.random.normal((2048,)) * .001
for dtype in (mx.float32, mx.bfloat16):
    x = mx.random.normal((1, 4, 4096)).astype(dtype)
    a, b = old(x), new(x)
    mx.eval(a, b)
    assert mx.array_equal(a, b).item(), (str(dtype), "bias")
    checks.append({"dtype": str(dtype), "count": 4, "bias": True, "exact": True})
(OUT / "mlp_integration_check.json").write_text(json.dumps(checks, indent=2))

# Time the exact integration patch with the ORIGINAL binary-search quantizer.
# This closes the gap between the optimized kernel macro and its runtime caller.
for projection in (old.gate_proj, old.up_proj):
    del projection.bias
trials = []
for count in (1, 4):
    x = mx.random.normal((1, count, 4096))
    functions = {
        "baseline": mx.compile(lambda x: x + old(x)),
        "patched": mx.compile(lambda x: x + new(x)),
    }
    final = None
    for mode in ("baseline", "patched", "patched", "baseline"):
        value = x
        mx.eval(functions[mode](value))
        raw = []
        for step in range(300):
            start = time.perf_counter()
            value = functions[mode](value)
            mx.eval(value)
            raw.append(time.perf_counter() - start)
        if final is None:
            final = value
        assert mx.array_equal(value, final).item(), (count, mode)
        row = {"count": count, "mode": mode, "mean_ms": statistics.mean(raw)*1000,
               "raw_seconds": raw, "final_300_exact": True}
        trials.append(row)
        print("ACTUAL_MLP_ABBA300", count, mode, row["mean_ms"], flush=True)
        (OUT / "actual_mlp_abba.json").write_text(json.dumps(trials, indent=2))
