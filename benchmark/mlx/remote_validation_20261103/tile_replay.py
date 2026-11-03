"""M5 actual NVFP4 gate bank: route, layout, and output-row tile ablations.

Every arm projects identical activations through identical checkpoint values.
The archived M4 route is a diagnostic substitution only. This projection replay
is not full-model throughput. Timings alternate every ten calls, 300 per arm.
"""
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist

O = Path('benchmark/mlx/remote_validation_20261103')
C = Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(10 * 2**30)
mx.set_cache_limit(256 * 2**20)
inputs = mx.load(O / 'actual_ffn_operands.npz')
m4_ids = mx.array([[[202,10,149,205,24,121,238,187],[202,24,149,10,142,276,120,2],[202,24,149,10,250,115,201,254],[202,24,149,267,250,115,201,254]]], mx.uint32)
data = dict(scope=__doc__, device=mx.device_info(), mlx=mx.__version__, cases=[], source_sha256=hashlib.sha256(Path('python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py').read_bytes()).hexdigest())


def pair(value, w, s, gs, route, rps, work):
    count, picks = route.size, route.shape[-1]
    e, n, pk = w.shape
    k = pk * 2
    return nv._kernel(False)(inputs=[value.reshape(-1,k),w,s,gs.reshape(-1),route.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work], template=[('O',value.dtype),('RPS',rps),('EXPERT',value.size//k == count),('SINGLES',True)], grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*route.shape,1,n)


for layer in (3, 26):
    a = mx.load(C / f'layer-{layer:02}.safetensors')
    p = 'mlp.switch_mlp.gate_proj.'
    full = [a[p + n] for n in ('weight', 'scales', 'weight_scale_2')]
    mx.eval(full)
    del a
    x = inputs[f'c4_l{layer}_x'][...,None,None,:]
    for pattern, route in (('actual', inputs[f'c4_l{layer}_ids']), ('m4', m4_ids)):
        mx.eval(route)
        keys = sorted(set(route.reshape(-1).tolist()))
        remap = {v:i for i,v in enumerate(keys)}
        compact = [mx.take(v, mx.array(keys,mx.int32),axis=0) for v in full]
        compact_ids = mx.array([[[remap[v] for v in row] for row in route[0].tolist()]],mx.uint32)
        mx.eval(compact, compact_ids)
        ct = Counter(route.reshape(-1).tolist())
        row = dict(layer=layer,pattern=pattern,pair_fraction=sum(2*(v//2) for v in ct.values())/route.size, samples=[])
        fs = {}
        for kind, values, ids in (('full',full,route),('compact',compact,compact_ids)):
            w,s,glob=values
            gs=glob*2688
            work=mlx_expert_pair_worklist(ids)
            singles=(mx.arange(ids.size,dtype=mx.uint32),mx.ones((ids.size,),mx.uint32))
            mx.eval(gs,*work,*singles)
            fs[(kind,'native')]=mx.compile(lambda value,w=w,s=s,gs=gs,ids=ids:mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs))
            for rps in (1,2,4,8):
                fs[(kind,'pair'+str(rps))]=mx.compile(lambda value,w=w,s=s,gs=gs,ids=ids,rps=rps,work=work:pair(value,w,s,gs,ids,rps,work))
            fs[(kind,'singles4')]=mx.compile(lambda value,w=w,s=s,gs=gs,ids=ids,work=singles:pair(value,w,s,gs,ids,4,work))
        ref=fs[('full','native')](x)
        for name,fn in fs.items():
            y=fn(x);mx.eval(ref,y)
            assert mx.array_equal(ref,y).item(),(layer,pattern,name)
            assert mx.all(mx.isfinite(y)).item(),(layer,pattern,name)
        raw={key:[] for key in fs}
        for block in range(30):
            order=list(fs) if block%2==0 else list(reversed(fs))
            for key in order:
                for step in range(10):
                    value=x+mx.array((block*10+step)*1e-8,x.dtype)
                    mx.eval(value)
                    t=time.perf_counter();result=fs[key](value);mx.eval(result);raw[key].append(time.perf_counter()-t)
        for key in fs:
            sample=dict(kind=key[0],mode=key[1],raw_seconds=raw[key],mean_ms=statistics.mean(raw[key])*1000)
            row['samples'].append(sample)
        data['cases'].append(row)
        (O/'tile_replay.json').write_text(json.dumps(data,indent=2))
        print('TILE_REPLAY',layer,pattern,row['pair_fraction'],{str(k):round(statistics.mean(v)*1000,4) for k,v in raw.items()},flush=True)
        del fs,compact,raw
    del full
    mx.clear_cache()
data['complete']=True
(O/'tile_replay.json').write_text(json.dumps(data,indent=2))
