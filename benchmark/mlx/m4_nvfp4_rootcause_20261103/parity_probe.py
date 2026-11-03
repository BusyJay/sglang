"""Find where custom arithmetic diverges from native before further tuning."""

import json

import mlx.core as mx

from common import OUT, bank, native, routing
from candidates import project
from native_clone import project as clone

mx.random.seed(955)
records = []
ids,e = routing(4)
for k in (512,1024,4096):
    p=bank(e,64,k)
    x=mx.random.normal((1,4,1,1,k))
    ref=native(x,p,ids)
    mx.eval(ref)
    arms={}
    for math in ("safe","fast","relaxed"):
        arms["clone_"+math]=lambda math=math:clone(x,p,ids,math=math)
        for fused in (False,True):
            arms[f"single_{math}_{int(fused)}"]=lambda math=math,fused=fused:project(x,p,ids,kind="single",static=True,math=math,fused=fused,rps=4)
    for rounding in ("separate", "contract_off"):
        arms[rounding]=lambda rounding=rounding:project(x,p,ids,kind="single",static=True,math="safe",rps=4,rounding=rounding)
    for name,fn in arms.items():
        y=fn();mx.eval(y)
        rec={"k":k,"name":name,"exact":mx.array_equal(y,ref).item(),"max_abs":mx.max(mx.abs(y-ref)).item(),"changed":mx.sum(y!=ref).item()}
        records.append(rec);print(rec,flush=True)
(OUT/"parity_probe.json").write_text(json.dumps(records,indent=2))
