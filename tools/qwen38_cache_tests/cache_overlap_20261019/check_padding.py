"""Compare CUTLASS route padding against the original resident reference."""
import runpy
import sys
import torch
from sglang.srt.layers.moe import qwen38_prefill_routes as routes
original = routes.partition_prefill_routes

def partition(*args):
    ci, cw, ai, aw = original(*args)
    ci = torch.where(cw != 0, ci, -1)
    if ai is not None:
        ai = torch.where(aw != 0, ai, -1)
    return ci, cw, ai, aw
routes.partition_prefill_routes = partition
sys.argv = ['cached_w4a16_gpu_check.py', '--cache-overlap-only', '--iterations', '30', '--output', '/results/padding-parity.json']
runpy.run_path('tools/qwen38_cache_tests/cached_w4a16_gpu_check.py', run_name='__main__')
