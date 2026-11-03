"""Final prior candidate and current warmup code, with import glue isolated."""

import importlib.util
import sys
import types
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT.parent / "m4_nvfp4_pair_20261103"))
spec = importlib.util.spec_from_file_location(
    "pair_fixture", OUT.parent / "m4_nvfp4_pair_20261103/common.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
bank, routing, native, load = fixture.bank, fixture.routing, fixture.native, fixture.load
for name in (
    "sglang.srt", "sglang.srt.hardware_backend", "sglang.srt.hardware_backend.mlx",
    "sglang.srt.hardware_backend.mlx.models", "sglang.kernels.ops.attention",
    "sglang.kernels.ops.attention.linear", "sglang.kernels.ops.layernorm",
    "sglang.kernels.ops.kvcache",
):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
    sys.modules[name] = package
warmup = load(
    "sglang.srt.hardware_backend.mlx.warmup",
    ROOT / "python/sglang/srt/hardware_backend/mlx/warmup.py",
)
