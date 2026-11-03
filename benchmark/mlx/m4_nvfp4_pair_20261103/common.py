"""Reuse previous native fixtures without rewriting their measurements."""

import importlib.util
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREV = OUT.parent / "m4_nvfp4_rootcause_20261103"
sys.path.insert(0, str(PREV))
spec = importlib.util.spec_from_file_location("previous_common", PREV / "common.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
bank, routing, native = fixtures.bank, fixtures.routing, fixtures.native
load = fixtures.load
rows = load("frozen_gather_rows", OUT / "frozen/mlx_nvfp4_gather.py")
work = load("frozen_pair_work", OUT / "frozen/mlx_expert_pairs.py")
oldpair = fixtures.exact
production = load("sglang.kernels.ops.gemm.mlx_nvfp4_swiglu", ROOT / "python/sglang/kernels/ops/gemm/mlx_nvfp4_swiglu.py")
