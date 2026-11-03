"""Native-parity two-row sharing including GPU worklist construction."""

import argparse
import json
import statistics
import time

import mlx.core as mx

from common import OUT,bank,native,routing
from candidates import project
from native_clone import project as clone
from paired_rows import project as pair,worklist

parser=argparse.ArgumentParser()
parser.add_argument("--pattern",default="real",choices=("real","none","all"))
parser.add_argument("--shape",default="gate",choices=("gate","down"))
parser.add_argument("--count",default=4,type=int)
parser.add_argument("--dtype",default="fp32",choices=("fp32","bf16"))
args=parser.parse_args()
mx.set_memory_limit(2*2**30);mx.set_cache_limit(128*2**20);mx.random.seed(914)
ids,e=routing(args.count,args.pattern)
n,k=(2048,4096) if args.shape=="gate" else (4096,2048)
p=bank(e,n,k)
shape=(1,args.count,1,1,k) if args.shape=="gate" else (1,args.count,8,1,k)
x=mx.random.normal(shape).astype(mx.float32 if args.dtype=="fp32" else mx.bfloat16)
work=worklist(ids);mx.eval(x,*work)
arms={"native":lambda:native(x,p,ids),"clone":lambda:clone(x,p,ids)}
arms["scoped4"]=lambda:pair(x,p,ids,4,rounding="scoped")
arms["inline4"]=lambda:pair(x,p,ids,4,rounding="scoped",inline_list=True)
for rps in (1,2,4,8):
    arms[f"pair{rps}"]=lambda rps=rps:pair(x,p,ids,rps)
    arms[f"prepair{rps}"]=lambda rps=rps:pair(x,p,ids,rps,work=work)
for static in (False,True):
    arms[f"list_exact{int(static)}"]=lambda static=static:project(x,p,ids,kind="list",static=static,math="safe",rounding="separate",rps=4)
data={"args":vars(args),"numerics":{},"trials":[],"first_ms":{},"device":mx.device_info()}
path=OUT/f"pair_micro_{args.shape}_{args.pattern}_c{args.count}_{args.dtype}.json"
ref=native(x,p,ids);mx.eval(ref)
for name,fn in arms.items():
    t=time.perf_counter();y=fn();mx.eval(y);data["first_ms"][name]=(time.perf_counter()-t)*1000
    rec={"exact":mx.array_equal(y,ref).item(),"max_abs":mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item()}
    data["numerics"][name]=rec;print("NUMERIC",name,rec,flush=True)
for j in range(4):
    row={}
    for name in list(arms) if j%2==0 else reversed(arms):
        fn=arms[name];mx.eval(fn());t=time.perf_counter()
        for _ in range(30):mx.eval(fn())
        row[name]=(time.perf_counter()-t)*1000/30
    data["trials"].append(row);path.write_text(json.dumps(data,indent=2));print("TRIAL",j,row,flush=True)
data["median_ms"]={name:statistics.median(row[name] for row in data["trials"]) for name in arms}
path.write_text(json.dumps(data,indent=2));print("SUMMARY",data["median_ms"],flush=True)
