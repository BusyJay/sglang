"""Check native sigmoid rounding before fusing the activation epilogue."""
import json

import mlx.core as mx
import mlx.nn as nn

from common import OUT,bank,native,routing
from pair_candidates import fused,pair

mx.random.seed(144)
records=[]
for count in (1,4):
    ids,e=routing(count);g,u=bank(e,2048,4096),bank(e,2048,4096)
    for dtype in (mx.float32,mx.bfloat16):
        x=mx.random.normal((1,count,1,1,4096)).astype(dtype)
        gate,up=native(x,g,ids),native(x,u,ids)
        expected=nn.silu(mx.minimum(gate,7.0))*mx.clip(up,-7.0,7.0)
        for rps in (1,2,4):
            y=fused(x,g,u,ids,rps);mx.eval(y,expected)
            row={"count":count,"dtype":str(dtype),"rps":rps,"exact":mx.array_equal(y,expected).item(),"changed":mx.sum(y!=expected).item(),"max_abs":mx.max(mx.abs(y.astype(mx.float32)-expected.astype(mx.float32))).item()}
            records.append(row);print(row,flush=True)
(OUT/'activation_check.json').write_text(json.dumps(records,indent=2))
