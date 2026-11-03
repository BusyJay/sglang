"""Small synthetic native NVFP4 banks with the captured top-8 routing pattern."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exact = load("nv_exact_pair", OUT / "frozen/nv_exact_pair.py")
old = load("nv_shared", OUT / "frozen/nv_shared.py")
for name in ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.gemm", "sglang.kernels.ops.moe"):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
        sys.modules[name] = pkg
production = load("sglang.kernels.ops.gemm.mlx_nvfp4_gather", ROOT / "python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py")
production_work = load("sglang.kernels.ops.moe.mlx_expert_pairs", ROOT / "python/sglang/kernels/ops/moe/mlx_expert_pairs.py")
REAL_IDS = [
    [202, 10, 149, 205, 24, 121, 238, 187],
    [202, 24, 149, 10, 142, 276, 120, 2],
    [202, 24, 149, 10, 250, 115, 201, 254],
    [202, 24, 149, 267, 250, 115, 201, 254],
]


def routing(count=4, pattern="real"):
    if pattern == "none":
        rows = [list(range(i * 8, (i + 1) * 8)) for i in range(count)]
    elif pattern == "all":
        rows = [list(range(8)) for _ in range(count)]
    else:
        rows = [REAL_IDS[i % 4] for i in range(count)]
    mapping = {expert: i for i, expert in enumerate(sorted({v for row in rows for v in row}))}
    return mx.array([[[mapping[v] for v in row] for row in rows]], mx.uint32), len(mapping)


def bank(experts, n, k):
    # Valid unchanged NVFP4 codes and finite E4M3 scales. Only the expert bank
    # count is reduced to fit M4; no checkpoint is loaded or converted.
    weight = mx.random.randint(0, 256, (experts, n, k // 2), dtype=mx.uint8)
    scales = mx.random.randint(40, 56, (experts, n, k // 16), dtype=mx.uint8)
    global_scale = mx.full((experts,), 0.012 / k**0.5, mx.float32)
    mx.eval(weight, scales, global_scale)
    return SimpleNamespace(weight=weight, scales=scales, weight_scale_2=global_scale)


def native(x, p, ids):
    return mx.gather_qmm(
        x, p.weight.view(mx.uint32), p.scales, rhs_indices=ids,
        mode="nvfp4", group_size=16, bits=4, global_scale=p.weight_scale_2 * 2688,
    )


def old_project(x, p, ids, rps=2):
    return old.project(x, p.weight, p.scales, p.weight_scale_2, ids, rps)
