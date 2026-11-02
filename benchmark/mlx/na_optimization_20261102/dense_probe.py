import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows
mx.random.seed(20261102);records=[];out=Path("benchmark/mlx/na_optimization_20261102/dense_probe.json")
for n,k in ((4096,8192),(24896,4096),(154880,4096)):
 w=mx.random.normal((n,k)).astype(mx.bfloat16)*0.01;mx.eval(w)
 for m in (1,3,4,8,16):
  x=mx.random.normal((1,m,k)).astype(mx.bfloat16);mx.eval(x)
  f=mx.compile(lambda x:mlx_dense_rows(x,w));g=mx.compile(lambda x:x@w.T)
  a=f(x);b=g(x);mx.eval(a,b);d=b.astype(mx.float32)-a.astype(mx.float32)
  r=dict(n=n,k=k,rows=m,rel_l2=mx.sqrt(mx.sum(d*d)/mx.sum(a.astype(mx.float32)**2)).item(),max_abs=mx.max(mx.abs(d)).item(),raw=[])
  for i in range(5):
   for name,fn in (("A",f),("B",g),("B",g),("A",f)):
    start=time.perf_counter()
    for j in range(10):mx.eval(fn(x))
    r["raw"].append(dict(group=i,mode=name,ms=(time.perf_counter()-start)*100))
  r["medians"]={name:statistics.median(z["ms"] for z in r["raw"] if z["mode"]==name) for name in ("A","B")}
  records.append(r);out.write_text(json.dumps(records,indent=2));print({a:b for a,b in r.items() if a!="raw"},flush=True)
 del w,x,a,b,f,g;mx.clear_cache()
