import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_swiglu_rows
from sglang.kernels.ops.gemm.mlx_qmm_swiglu import _nax_kernel
OUT=Path("benchmark/mlx/na_optimization_20261102")
mx.random.seed(20261102)
records=[]
for bits in (4,5,6):
 k,n=4096,2048
 ops=[]
 for _ in range(2):
  p=mx.quantize(mx.random.normal((n,k)).astype(mx.float16)*0.02,group_size=64,bits=bits);mx.eval(p);ops.append(p)
 for m in (3,4,8,16):
  x=mx.random.normal((1,m,k)).astype(mx.float16);mx.eval(x)
  work=mx.array([0,0,m],mx.uint32);count=mx.array([1],mx.uint32)
  def baseline(x):return mlx_affine_swiglu_rows(x,*ops,bits=bits,group_size=64)
  def candidate(x):
   return _nax_kernel()(inputs=[x.reshape(m,1,k),work,count,*[a[None] for p in ops for a in p],mx.array([0.],mx.float32)],template=[("T",x.dtype),("M",m),("N",n),("K",k),("GS",64),("BITS",bits),("HAS_LIMIT",False),("BM",32)],grid=(128*(n//64),1,1),threadgroup=(128,1,1),output_shapes=[(m,1,n)],output_dtypes=[x.dtype])[0].reshape(1,m,n)
  f,g=mx.compile(baseline),mx.compile(candidate)
  t=time.perf_counter();ref=f(x);mx.eval(ref);cold_a=time.perf_counter()-t
  t=time.perf_counter();got=g(x);mx.eval(got);cold_b=time.perf_counter()-t
  diff=got.astype(mx.float32)-ref.astype(mx.float32)
  row=dict(bits=bits,rows=m,max_abs=mx.max(mx.abs(diff)).item(),rel_l2=mx.sqrt(mx.sum(diff*diff)/mx.sum(ref.astype(mx.float32)**2)).item(),exact=mx.array_equal(ref,got).item(),cold_seconds=[cold_a,cold_b],raw=[])
  for _ in range(2):mx.eval(f(x),g(x))
  for group in range(5):
   for name,fn in (("A",f),("B",g),("B",g),("A",f)):
    start=time.perf_counter()
    for _ in range(20):mx.eval(fn(x))
    row["raw"].append(dict(group=group,mode=name,ms=(time.perf_counter()-start)*50))
  row["medians"]={name:statistics.median(r["ms"] for r in row["raw"] if r["mode"]==name) for name in ("A","B")}
  records.append(row);(OUT/"shared_gate_probe.json").write_text(json.dumps(records,indent=2));print({a:b for a,b in row.items() if a!="raw"},flush=True)
 del ops
 mx.synchronize();mx.clear_cache()
