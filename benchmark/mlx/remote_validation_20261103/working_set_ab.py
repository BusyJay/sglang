"""M5 full/compact NVFP4 banks, with identical checkpoint values and routes.

Four routed FFNs, two layers per compiled group, 300 dependent steps. RMS input
normalization bounds this diagnostic recurrence; it is not full-model inference
and changes no production equations or checkpoint arrays. Each arm uses the
same normalization, uniform routing scores and ordered sum. Endpoints must be
finite and bitwise equal. Frequent arm alternation limits temporal drift.
"""
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist

O = Path('benchmark/mlx/remote_validation_20261103')
C = Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(20 * 2**30)
mx.set_cache_limit(256 * 2**20)
inputs = mx.load(O / 'actual_ffn_operands.npz')
banks, ids, routes = {}, {}, {}
layers = (3, 4, 5, 6)
for layer in layers:
    a = mx.load(C / f'layer-{layer:02}.safetensors')
    route = inputs[f'c4_l{layer}_ids']
    mx.eval(route)
    keys = sorted(set(route.reshape(-1).tolist()))
    mapping = {v: i for i, v in enumerate(keys)}
    compact_route = mx.array([[[mapping[v] for v in row] for row in route[0].tolist()]], mx.uint32)
    ids[layer] = {'full': route, 'compact': compact_route}
    ct = Counter(route.reshape(-1).tolist())
    routes[layer] = dict(ids=route.tolist(), unique=len(keys), pair_fraction=sum(2 * (v // 2) for v in ct.values()) / route.size)
    banks[layer] = {}
    for kind in ('full', 'compact'):
        banks[layer][kind] = []
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            prefix = 'mlp.switch_mlp.' + projection + '.'
            values = [a[prefix + n] for n in ('weight', 'scales', 'weight_scale_2')]
            if kind == 'compact':
                values = [mx.take(v, mx.array(keys, mx.int32), axis=0) for v in values]
            mx.eval(values)
            banks[layer][kind].append(values)
    del a
print('WORKING_SET_READY', mx.get_active_memory() / 2**30, routes, flush=True)


def layer_fn(x, layer, kind, mode):
    x = mx.fast.rms_norm(x, None, 1e-5)
    route = ids[layer][kind]
    work = mlx_expert_pair_worklist(route) if mode == 'candidate' else None

    def proj(value, p):
        w, s, gs = p
        if mode == 'candidate':
            return mlx_nvfp4_gather_rows(value, w, s, gs * 2688, route, worklist=work)
        return mx.gather_qmm(value, w.view(mx.uint32), s, rhs_indices=route, mode='nvfp4', group_size=16, bits=4, global_scale=gs * 2688)

    gate, up = [proj(x[..., None, None, :], p) for p in banks[layer][kind][:2]]
    activated = nn.silu(mx.minimum(gate, 7.0)) * mx.clip(up, -7.0, 7.0)
    y = mx.sum(proj(activated, banks[layer][kind][2]).squeeze(-2) * (1 / route.shape[-1]), axis=-2)
    return x + y


fns = {}
for kind in ('full', 'compact'):
    for mode in ('baseline', 'candidate'):
        groups = []
        for start in (0, 2):
            def group(value, start=start, kind=kind, mode=mode):
                for layer in layers[start:start + 2]:
                    value = layer_fn(value, layer, kind, mode)
                return value
            groups.append(mx.compile(group))
        fns[(kind, mode)] = groups


def chain(x, key):
    for fn in fns[key]:
        x = fn(x)
    return x


x = inputs['c4_l3_x']
ref = chain(x, ('full', 'baseline'))
mx.eval(ref)
out = dict(device=mx.device_info(), mlx=mx.__version__, scope=__doc__, routes=routes, samples=[], resident_bytes=mx.get_active_memory(), env={k: v for k, v in os.environ.items() if k.startswith('MLX_')}, swap_before=subprocess.check_output(['sysctl', 'vm.swapusage'], text=True), source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py'))})
for key in fns:
    value = chain(x, key)
    mx.eval(value)
    assert mx.array_equal(value, ref).item(), key
    assert mx.all(mx.isfinite(value)).item(), key
# 30 blocks of 10 steps per arm = 300 dependent steps per arm, two rounds.
for repeat in range(2):
    values = {key: x for key in fns}
    raw = {key: [] for key in fns}
    for block in range(30):
        order = list(fns) if (block + repeat) % 2 == 0 else list(reversed(fns))
        for key in order:
            value = values[key]
            for step in range(10):
                t = time.perf_counter()
                value = chain(value, key)
                mx.eval(value)
                raw[key].append(time.perf_counter() - t)
            values[key] = value
    reference = values[('full', 'baseline')]
    for key in fns:
        finite = mx.all(mx.isfinite(values[key])).item()
        exact = mx.array_equal(values[key], reference).item()
        assert finite and exact, (repeat, key, finite, exact)
        row = dict(round=repeat, kind=key[0], mode=key[1], raw_seconds=raw[key], mean_ms=statistics.mean(raw[key]) * 1000, finite=finite, final_exact=exact)
        out['samples'].append(row)
        print('WORKSET300', repeat, key, round(row['mean_ms'], 4), finite, exact, flush=True)
    (O / 'working_set_ab.json').write_text(json.dumps(out, indent=2))
out['complete'] = True
out['peak_bytes'] = mx.get_peak_memory()
out['swap_after'] = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
(O / 'working_set_ab.json').write_text(json.dumps(out, indent=2))
