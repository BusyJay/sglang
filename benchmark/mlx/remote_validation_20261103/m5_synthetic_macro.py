"""Same four-layer synthetic NVFP4 macro used on M4, now on M5.

Matches M4 seed, packed-value distributions, top8 route, residual/score equation,
two-layer compiled grouping and 300 dependent steps. This is a synthetic macro,
not full-checkpoint throughput. All endpoints must remain finite and exact.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist


O=Path('benchmark/mlx/remote_validation_20261103')
mx.set_memory_limit(3*2**30);mx.set_cache_limit(128*2**20);mx.random.seed(999)
rows=[[202,10,149,205,24,121,238,187],[202,24,149,10,142,276,120,2],[202,24,149,10,250,115,201,254],[202,24,149,267,250,115,201,254]]
mapping={v:i for i,v in enumerate(sorted({v for row in rows for v in row}))}
ids=mx.array([[[mapping[v] for v in row] for row in rows]],mx.uint32);e=len(mapping)
banks=[]
for layer in range(4):
    values=[]
    for n,k in ((2048,4096),(2048,4096),(4096,2048)):
        w=mx.random.randint(0,256,(e,n,k//2),dtype=mx.uint8)
        s=mx.random.randint(40,56,(e,n,k//16),dtype=mx.uint8)
        gs=mx.full((e,),0.012/k**0.5,mx.float32)
        mx.eval(w,s,gs);values.append((w,s,gs))
    banks.append(values)
x=mx.random.normal((1,4,4096)).astype(mx.float32)
scores=mx.random.uniform(shape=(1,4,8,1));scores=scores/mx.sum(scores,axis=-2,keepdims=True)
mx.eval(x,scores,ids)

def layer(value,p,mode,ids,scores):
    work=mlx_expert_pair_worklist(ids) if mode=='candidate' else None
    def proj(value,p):
        w,s,gs=p
        if mode=='candidate':return nv.mlx_nvfp4_gather_rows(value,w,s,gs*2688,ids,worklist=work)
        return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs*2688)
    gate,up=[proj(value[...,None,None,:],p) for p in p[:2]]
    act=nn.silu(mx.minimum(gate,7.0))*mx.clip(up,-7.0,7.0)
    output=proj(act,p[2]).squeeze(-2)
    return value+mx.sum(output*scores,axis=-2).astype(value.dtype)
fns={}
for mode in ('baseline','candidate'):
    fns[mode]=[]
    for start in (0,2):
        def group(value,route,prob,start=start,mode=mode):
            for p in banks[start:start+2]:value=layer(value,p,mode,route,prob)
            return value
        fns[mode].append(mx.compile(group))
def chain(value,mode):
    for fn in fns[mode]:value=fn(value,ids,scores)
    return value
out=dict(device=mx.device_info(),scope=__doc__,trials=[],source_sha256=hashlib.sha256(Path('python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py').read_bytes()).hexdigest())
ref=chain(x,'baseline');new=chain(x,'candidate');mx.eval(ref,new);assert mx.array_equal(ref,new).item()
for repeat in range(4):
    row={};outputs={}
    for mode in list(fns) if repeat%2==0 else list(reversed(fns)):
        value=x;mx.eval(chain(x,mode));raw=[]
        for step in range(300):
            t=time.perf_counter();value=chain(value,mode);mx.eval(value);raw.append(time.perf_counter()-t)
        outputs[mode]=value
        row[mode]=dict(raw_seconds=raw,mean_ms=statistics.mean(raw)*1000,finite=mx.all(mx.isfinite(value)).item())
    for mode in fns:
        row[mode]['exact']=mx.array_equal(outputs[mode],outputs['baseline']).item()
        assert row[mode]['finite'] and row[mode]['exact'],(mode,row[mode])
    out['trials'].append(row);(O/'m5_synthetic_macro.json').write_text(json.dumps(out,indent=2))
    print('M5_SYNTHETIC300',repeat,{mode:row[mode]['mean_ms'] for mode in row},flush=True)
out['complete']=True;(O/'m5_synthetic_macro.json').write_text(json.dumps(out,indent=2))
