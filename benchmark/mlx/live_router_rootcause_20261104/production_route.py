"""M5 real single-layer MoE: captured/live route, order and submit boundaries.

Preserves checkpoint arrays and strict FP32 routing. This is a diagnostic,
not full-model throughput. No shared model state is changed.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

ROOT = Path('/Users/jay/workspace/sglang')
OUT = ROOT / 'benchmark/mlx/live_router_rootcause_20261104'
CACHE = Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
TARGET = Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=26)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--skip-bench', action='store_true')
    args = parser.parse_args()
    mx.set_memory_limit(12 * 2**30)
    mx.set_cache_limit(256 * 2**20)
    config = json.loads((TARGET / 'config.json').read_text())
    a = SimpleNamespace(**config.get('text_config', config))
    arrays = mx.load(CACHE / f'layer-{args.layer:02}.safetensors')
    bank = []
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        p = 'mlp.switch_mlp.' + name + '.'
        bank.append(tuple(arrays[p + n] for n in ('weight', 'scales', 'weight_scale_2')))
    shared = g.MLP(a, a.moe_intermediate_size * a.n_shared_experts)
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        p = 'mlp.shared_experts.' + name + '.'
        setattr(shared, name, FP8Linear(arrays[p + 'weight'], arrays[p + 'scales'], block_size=128, activation_mode='dynamic32'))
    router = arrays['mlp.gate.weight'].astype(mx.float32)
    bias = arrays['mlp.e_score_correction_bias']
    mx.eval(bank, shared.parameters(), router, bias)
    operands = mx.load(ROOT / 'benchmark/mlx/remote_validation_20261103/actual_ffn_operands.npz')
    x = operands[f'c4_l{args.layer}_x']
    archived_ids = operands[f'c4_l{args.layer}_ids']
    del operands, arrays

    def route(value):
        values=value.astype(mx.float32).reshape(-1,value.shape[-1])
        blocks=[]
        for start in range(0,values.shape[0],8):
            tile=values[start:start+8];count=tile.shape[0]
            if count<8:tile=mx.concatenate([tile,mx.zeros((8-count,tile.shape[1]),mx.float32)],0)
            blocks.append((tile @ router.T)[:count])
        scores=mx.concatenate(blocks,0).reshape(*value.shape[:-1],router.shape[0])
        return g._group_expert_select(scores, bias, a.num_experts_per_tok, a.n_group, a.topk_group, a.routed_scaling_factor, a.norm_topk_prob, a.scoring_func)

    captured_ids, captured_weights = route(x)
    mx.eval(captured_ids, captured_weights)
    print('LAYER_READY', args.layer, 'GiB', mx.get_active_memory()/2**30, 'route_matches_archive', mx.array_equal(captured_ids, archived_ids).item(), flush=True)

    def project(value, p, ids, mode, work):
        w, scales, gs = p
        if mode == 'native':
            return mx.gather_qmm(value, w.view(mx.uint32), scales, rhs_indices=ids, mode='nvfp4', group_size=16, bits=4, global_scale=gs*2688)
        k, n, count, picks = w.shape[-1]*2, w.shape[-2], ids.size, ids.shape[-1]
        return nv._kernel(False)(inputs=[value.reshape(-1,k), w, scales, (gs*2688).reshape(-1), ids.reshape(-1), mx.array([k,n,picks,count], mx.int32), *work], template=[('O',value.dtype),('RPS',1),('EXPERT',value.size//k==count),('SINGLES',True)], grid=(32,n,count), threadgroup=(32,2,1), output_shapes=[(count,n)], output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)

    def forward(value, ids, weights, mode, scope, order, placement):
        if scope.startswith('live'):
            ids, weights = route(value)
        if scope == 'live_x_after_router':
            value = mx.depends(value,[ids,weights])
        elif scope == 'live_x_after_ids':
            value=mx.depends(value,ids)
        elif scope == 'live_x_after_weights':
            value=mx.depends(value,weights)
        work = mlx_expert_pair_worklist(ids) if mode == 'reuse1' and placement == 'before_shared' else None
        shared_y = shared(value, batch_fusion=True)
        if mode == 'reuse1' and work is None:
            work = mlx_expert_pair_worklist(ids)
        if scope == 'live_experts_after_shared':
            ids = mx.depends(ids,shared_y)
            if mode == 'reuse1':work=tuple(mx.depends(list(work),shared_y))
        v = value[..., None, None, :]
        if order == 'up_first':
            up = project(v, bank[1], ids, mode, work)
            gate = project(v, bank[0], ids, mode, work)
        else:
            gate = project(v, bank[0], ids, mode, work)
            up = project(v, bank[1], ids, mode, work)
        act = nn.silu(mx.minimum(gate, a.swiglu_limit))*mx.clip(up,-a.swiglu_limit,a.swiglu_limit)
        down = project(act, bank[2], ids, mode, work).squeeze(-2)
        if scope == 'live_shared_after_experts':
            shared_y = shared(mx.depends(value,down), batch_fusion=True)
        return (down*weights[...,None]).sum(-2).astype(value.dtype) + shared_y

    specs = [(scope, mode, 'up_first', 'after_shared') for scope in ('captured','live','live_x_after_router','live_x_after_ids','live_x_after_weights') for mode in ('native','reuse1')]
    fs = {spec: mx.compile(lambda value,ids,weights,spec=spec: forward(value,ids,weights,spec[1],spec[0],spec[2],spec[3])) for spec in specs}
    data = dict(scope=__doc__, layer=args.layer, device=mx.device_info(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=[], samples=[], capture=[])
    ref = fs[specs[0]](x,captured_ids,captured_weights)
    for spec, fn in fs.items():
        y = fn(x,captured_ids,captured_weights)
        mx.eval(ref,y)
        exact = mx.array_equal(ref,y).item()
        assert exact, spec
        data['checks'].append(dict(spec=spec,exact=exact))
    for spec, fn in fs.items():
        for i in range(5):
            y = fn(x+mx.array(i*1e-8,mx.float32),captured_ids,captured_weights)
            mx.eval(y)
    raw = {spec: [] for spec in specs}
    for block in range(0 if args.skip_bench else (args.steps+9)//10):
        sequence = specs if block%2 == 0 else list(reversed(specs))
        for spec in sequence:
            for step in range(min(10,args.steps-block*10)):
                value = x+mx.array((block*10+step)*1e-8,mx.float32)
                mx.eval(value)
                t = time.perf_counter()
                y = fs[spec](value,captured_ids,captured_weights)
                mx.eval(y)
                raw[spec].append(time.perf_counter()-t)
    for spec, seconds in raw.items():
        if not seconds:
            continue
        row = dict(spec=spec,raw_seconds=seconds,mean_ms=statistics.mean(seconds)*1000,median_ms=statistics.median(seconds)*1000)
        data['samples'].append(row)
        print('LAYER300',args.layer,spec,row['mean_ms'],flush=True)
    if args.capture:
        # One forward per capture: preserve the real bank and producer dependency.
        # The driver may duplicate full bank buffers. Stop after a >6 GiB trace.
        for spec in specs[:4]:
            name=f'l{args.layer}_{spec[0]}_{spec[1]}_v2'
            path=OUT/(name+'.gputrace')
            assert not path.exists(),path
            mx.synchronize()
            mx.metal.start_capture(str(path))
            stream=mx.new_stream(mx.gpu)
            with mx.stream(stream):
                value=x+mx.array(0.00001,mx.float32)
                y=fs[spec](value,captured_ids,captured_weights)
                mx.eval(y)
                mx.synchronize(stream)
            mx.metal.stop_capture()
            size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            print('CAPTURE_DONE',name,size,flush=True)
            data['capture'].append(dict(spec=spec,path=str(path),bytes=size))
            if size > 6*2**30:
                break
    data['complete']=True
    data['active_bytes']=mx.get_active_memory()
    data['peak_bytes']=mx.get_peak_memory()
    (OUT/f'production_route_{args.layer}.json').write_text(json.dumps(data,indent=2))


if __name__ == '__main__':
    main()
