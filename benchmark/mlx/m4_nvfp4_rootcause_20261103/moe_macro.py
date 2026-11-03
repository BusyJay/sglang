"""Actual-width MoE projection/activation/combine chain, 300 dependent steps."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time

import mlx.core as mx
import mlx.nn as nn

from common import OUT,ROOT,bank,native,routing,production,production_work
from candidates import project
from paired_rows import project as pair,worklist

parser=argparse.ArgumentParser()
parser.add_argument("--count",type=int,default=4)
parser.add_argument("--layers",type=int,default=4)
parser.add_argument("--steps",type=int,default=300)
parser.add_argument("--pattern",default="real",choices=("real","none","all"))
parser.add_argument("--dtype",default="fp32",choices=("fp32","bf16"))
parser.add_argument("--modes",default="pair")
args=parser.parse_args()
mx.set_memory_limit(3*2**30);mx.set_cache_limit(128*2**20);mx.random.seed(999)
ids,e=routing(args.count,args.pattern)
banks=[]
for _ in range(args.layers):
    banks.append([bank(e,n,k) for n,k in ((2048,4096),(2048,4096),(4096,2048))])
x=mx.random.normal((1,args.count,4096)).astype(mx.float32 if args.dtype=="fp32" else mx.bfloat16)
scores=mx.random.uniform(shape=(1,args.count,8,1))
scores=scores/mx.sum(scores,axis=-2,keepdims=True)
mx.eval(x,scores,ids)


def layer(value,p,mode,ids,scores):
    gate,up,down=p
    projected=value[...,None,None,:]
    if mode in ("production","production_inline"):
        work=production_work.mlx_expert_pair_worklist(ids) if mode=="production" else None
        g,u=[production.mlx_nvfp4_gather_rows(projected,w.weight,w.scales,w.weight_scale_2*2688,ids,worklist=work) for w in (gate,up)]
    elif mode in ("pair","scoped","inline"):
        work=worklist(ids)
        opts={"rounding":"volatile" if mode=="pair" else "scoped", "inline_list":mode=="inline"}
        g,u=pair(projected,gate,ids,work=work,**opts),pair(projected,up,ids,work=work,**opts)
    elif mode=="list":
        g,u=[project(projected,w,ids,kind="list",static=False,math="safe",rounding="separate",rps=4) for w in (gate,up)]
    else:
        g,u=native(projected,gate,ids),native(projected,up,ids)
    activated=nn.silu(mx.minimum(g,7.0))*mx.clip(u,-7.0,7.0)
    if mode in ("production","production_inline"):
        result=production.mlx_nvfp4_gather_rows(activated,down.weight,down.scales,down.weight_scale_2*2688,ids,worklist=work)
    elif mode in ("pair","scoped","inline"):
        result=pair(activated,down,ids,work=work,**opts)
    elif mode=="list":
        result=project(activated,down,ids,kind="list",static=False,math="safe",rounding="separate",rps=4)
    else:
        result=native(activated,down,ids)
    # Same route weights and ordered reduction for both arms.
    combined=mx.sum(result.squeeze(-2)*scores,axis=-2).astype(value.dtype)
    return value+combined


modes=["native"]+args.modes.split(",")
compiled={}
for mode in modes:
    compiled[mode]=[]
    for start in range(0,args.layers,2):
        def block(value,route,prob,start=start,mode=mode):
            for p in banks[start:start+2]:value=layer(value,p,mode,route,prob)
            return value
        compiled[mode].append(mx.compile(block))


def chain(value,mode):
    for fn in compiled[mode]:value=fn(value,ids,scores)
    return value


data={"args":vars(args),"scope":"M4 synthetic actual-width MoE chain, compact banks, captured/remapped top-8 routing; no full-model or HTTP throughput",
      "device":mx.device_info(),"mlx":mx.__version__,"experts":e,"ids":ids.tolist(),
      "weight_bytes":sum(p.weight.nbytes+p.scales.nbytes+p.weight_scale_2.nbytes for ps in banks for p in ps),
      "first_ms":{},"numerics":{},"trials":[],"swap_before":subprocess.check_output(["sysctl","vm.swapusage"],text=True),
      "hashes":{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob("*.py")}}
path=OUT/f"moe_macro_{args.pattern}_c{args.count}_{args.dtype}_{args.layers}layers_{args.modes}.json"
for file in (ROOT/"python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py",ROOT/"python/sglang/kernels/ops/moe/mlx_expert_pairs.py"):
    data["hashes"][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
ref=chain(x,"native");mx.eval(ref)
for mode in modes:
    t=time.perf_counter();y=chain(x,mode);mx.eval(y);data["first_ms"][mode]=(time.perf_counter()-t)*1000
    data["numerics"][mode]={"one_step_exact":mx.array_equal(y,ref).item(),"max_abs":mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item()}
    print("NUMERIC",mode,data["numerics"][mode],flush=True)
    assert data["numerics"][mode]["one_step_exact"]
for repeat in range(4):
    row={};outputs={}
    for mode in modes if repeat%2==0 else reversed(modes):
        mx.eval(chain(x,mode));value=x;t=time.perf_counter()
        raw=[]
        for step in range(args.steps):
            t0=time.perf_counter();value=chain(value,mode);mx.eval(value);raw.append((time.perf_counter()-t0)*1000)
        elapsed=(time.perf_counter()-t)*1000/args.steps
        outputs[mode]=value
        row[mode]={"ms_per_step":elapsed,"raw_ms":raw,"finite":mx.all(mx.isfinite(value)).item()}
    for mode in modes:
        row[mode]["final_exact"]=mx.array_equal(outputs[mode],outputs["native"]).item()
        assert row[mode]["final_exact"] and row[mode]["finite"],(mode,row[mode])
    data["trials"].append(row);path.write_text(json.dumps(data,indent=2));print("TRIAL",repeat,{m:{k:v for k,v in row[m].items() if k!="raw_ms"} for m in row},flush=True)
data["median_ms"]={mode:statistics.median(row[mode]["ms_per_step"] for row in data["trials"]) for mode in modes}
data["peak_bytes"]=mx.get_peak_memory();data["swap_after"]=subprocess.check_output(["sysctl","vm.swapusage"],text=True)
path.write_text(json.dumps(data,indent=2));print("SUMMARY",data["median_ms"],data["peak_bytes"],flush=True)
