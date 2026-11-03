"""Native, projection pairing and fused SwiGLU: actual-width dependent chain."""
import argparse
import hashlib
import json
import statistics
import subprocess
import time

import mlx.core as mx
import mlx.nn as nn

from common import OUT, ROOT, bank, native, routing, rows, work, production
from pair_candidates import pair, fused

parser=argparse.ArgumentParser()
parser.add_argument('--count',type=int,default=1)
parser.add_argument('--layers',type=int,default=4)
parser.add_argument('--dtype',default='fp32',choices=('fp32','bf16'))
parser.add_argument('--rps',type=int,default=2)
parser.add_argument('--modes',default='pair,fused')
args=parser.parse_args()
mx.set_memory_limit(3*2**30);mx.set_cache_limit(128*2**20);mx.random.seed(120)
ids,e=routing(args.count)
banks=[[bank(e,n,k) for n,k in ((2048,4096),(2048,4096),(4096,2048))] for _ in range(args.layers)]
x=mx.random.normal((1,args.count,4096)).astype(mx.float32 if args.dtype=='fp32' else mx.bfloat16)
scores=mx.random.uniform(shape=(1,args.count,8,1));scores=scores/mx.sum(scores,axis=-2,keepdims=True)
mx.eval(x,scores)


def layer(x,p,mode,ids,scores):
    g,u,d=p;expanded=x[...,None,None,:]
    if mode=='production':
        activated=production.mlx_nvfp4_swiglu(expanded,g.weight,g.scales,g.weight_scale_2*2688,u.weight,u.scales,u.weight_scale_2*2688,ids,limit=7.0)
    elif mode=='fused':
        activated=fused(expanded,g,u,ids,rps=args.rps)
    else:
        if mode=='pair':gate,up=pair(expanded,g,u,ids,rps=args.rps)
        else:gate,up=native(expanded,g,ids),native(expanded,u,ids)
        activated=nn.silu(mx.minimum(gate,7.0))*mx.clip(up,-7.0,7.0)
    result=native(activated,d,ids)
    return x+mx.sum(result.squeeze(-2)*scores,axis=-2).astype(x.dtype)


modes=['native']+args.modes.split(',')
functions={}
for mode in modes:
    functions[mode]=[]
    for start in range(0,args.layers,2):
        def fn(value,ids,scores,start=start,mode=mode):
            for p in banks[start:start+2]:value=layer(value,p,mode,ids,scores)
            return value
        functions[mode].append(mx.compile(fn))


def chain(value,mode):
    for fn in functions[mode]:value=fn(value,ids,scores)
    return value


data={'args':vars(args),'device':mx.device_info(),'mlx':mx.__version__,'numerics':{},'first_ms':{},'trials':[],
      'scope':'M4 synthetic actual-width MoE chain; no full-model or HTTP throughput',
      'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},
      'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
path=OUT/f'macro_c{args.count}_{args.dtype}_r{args.rps}.json'
data['hashes']['production']=hashlib.sha256((ROOT/'python/sglang/kernels/ops/gemm/mlx_nvfp4_swiglu.py').read_bytes()).hexdigest()
ref=chain(x,'native');mx.eval(ref)
for mode in modes:
    t=time.perf_counter();y=chain(x,mode);mx.eval(y);data['first_ms'][mode]=(time.perf_counter()-t)*1000
    rec={'exact':mx.array_equal(y,ref).item(),'max_abs':mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item()}
    data['numerics'][mode]=rec;print('NUMERIC',mode,rec,flush=True);assert rec['exact'],mode
for repeat in range(4):
    row={};outputs={}
    for mode in modes if repeat%2==0 else reversed(modes):
        mx.eval(chain(x,mode));value=x;raw=[]
        for _ in range(300):
            t=time.perf_counter();value=chain(value,mode);mx.eval(value);raw.append((time.perf_counter()-t)*1000)
        row[mode]={'mean_ms':statistics.mean(raw),'raw_ms':raw};outputs[mode]=value
    for mode in modes:
        row[mode]['final_exact']=mx.array_equal(outputs[mode],outputs['native']).item()
        assert row[mode]['final_exact'] and mx.all(mx.isfinite(outputs[mode])).item(),mode
    data['trials'].append(row);path.write_text(json.dumps(data,indent=2));print('TRIAL',repeat,{m:row[m]['mean_ms'] for m in row},flush=True)
data['median_ms']={m:statistics.median(r[m]['mean_ms'] for r in data['trials']) for m in modes}
data['peak_bytes']=mx.get_peak_memory();data['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
path.write_text(json.dumps(data,indent=2));print('SUMMARY',data['median_ms'],flush=True)
