"""Actual-width fused projection/activation geometry, alternating raw times."""
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from common import OUT,bank,native,routing
from pair_candidates import fused

mx.set_cache_limit(128*2**20);mx.set_memory_limit(2*2**30);mx.random.seed(161)
ids,e=routing(1);g,u=bank(e,2048,4096),bank(e,2048,4096)
x=mx.random.normal((1,1,1,1,4096));mx.eval(x)
def baseline():
    gate,up=native(x,g,ids),native(x,u,ids)
    return nn.silu(mx.minimum(gate,7.0))*mx.clip(up,-7.0,7.0)
arms={'native':baseline}
for rps in (1,2,4,8):
    for sgs in (2,4,8):arms[f'r{rps}s{sgs}']=lambda rps=rps,sgs=sgs:fused(x,g,u,ids,rps=rps,sgs=sgs)
data={'numerics':{},'trials':[]};ref=baseline();mx.eval(ref)
for name,fn in arms.items():
    y=fn();mx.eval(y);data['numerics'][name]=mx.array_equal(y,ref).item();assert data['numerics'][name],name
for repeat in range(4):
    row={}
    for name in list(arms) if repeat%2==0 else reversed(arms):
        fn=arms[name];mx.eval(fn());t=time.perf_counter()
        for _ in range(40):mx.eval(fn())
        row[name]=(time.perf_counter()-t)*1000/40
    data['trials'].append(row);print('TRIAL',repeat,row,flush=True)
data['median_ms']={m:statistics.median(r[m] for r in data['trials']) for m in arms}
(OUT/'geometry.json').write_text(json.dumps(data,indent=2));print('SUMMARY',data['median_ms'],flush=True)
