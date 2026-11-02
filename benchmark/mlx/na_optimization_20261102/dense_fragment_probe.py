import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows
source='\nusing namespace mlx::steel;\nconst uint sg=simdgroup_index_in_threadgroup;\nconst uint col=threadgroup_position_in_grid.x*64+sg*32;\nconst uint part=threadgroup_position_in_grid.y;\nNAXTile<float,1,2> c;c.clear();\nfor(uint k=part*(K/SPLIT);k<(part+1)*(K/SPLIT);k+=32){\n NAXTile<T,1,2> a;NAXTile<T,2,2> b;\n a.load_safe(x+k,K,short2(32,M));\n b.load(w+size_t(col)*K+k,K);\n tile_matmad_nax(c,a,metal::bool_constant<false>{},b,metal::bool_constant<true>{});\n}\nfor(short j=0;j<2;++j){\n auto v=c.frag_at(0,j);\n for(short i=0;i<8;++i){auto p=BaseNAXFrag::get_coord(i);if(p.y<M)out[(size_t(part)*M+p.y)*N+col+j*16+p.x]=v[i];}\n}\n'
kernel=mx.fast.metal_kernel(name="sglang_dense_fragment_nax",input_names=["x","w"],output_names=["out"],source=source,header=Path("python/sglang/kernels/ops/gemm/mlx_qmm_swiglu_nax.metal").read_text())
mx.random.seed(20261102);records=[];out=Path("benchmark/mlx/na_optimization_20261102/dense_fragment_probe.json")
for n,k in ((4096,8192),(24896,4096),(154880,4096)):
 w=mx.random.normal((n,k)).astype(mx.bfloat16)*0.01;mx.eval(w)
 for m in (3,4,8):
  x=mx.random.normal((1,m,k)).astype(mx.bfloat16);mx.eval(x)
  f=mx.compile(lambda x:mlx_dense_rows(x,w));a=f(x);mx.eval(a)
  for split in (1,4,8,16):
   for sg in (2,):
    def candidate(x):
     v=kernel(inputs=[x,w],template=[("T",x.dtype),("M",m),("N",n),("K",k),("SPLIT",split),("SG",sg)],grid=(64*(n//64),split,1),threadgroup=(32*sg,1,1),output_shapes=[(split,m,n)],output_dtypes=[mx.float32])[0]
     return mx.sum(v,0).astype(x.dtype)[None]
    g=mx.compile(candidate);b=g(x);mx.eval(b);d=b.astype(mx.float32)-a.astype(mx.float32)
    r=dict(n=n,k=k,rows=m,split=split,sg=sg,rel_l2=mx.sqrt(mx.sum(d*d)/mx.sum(a.astype(mx.float32)**2)).item(),max_abs=mx.max(mx.abs(d)).item(),raw=[])
    for i in range(3):
     for name,fn in (("A",f),("B",g),("B",g),("A",f)):
      start=time.perf_counter()
      for j in range(10):mx.eval(fn(x))
      r["raw"].append(dict(group=i,mode=name,ms=(time.perf_counter()-start)*100))
    r["medians"]={name:statistics.median(z["ms"] for z in r["raw"] if z["mode"]==name) for name in ("A","B")}
    records.append(r);out.write_text(json.dumps(records,indent=2));print({a:b for a,b in r.items() if a!="raw"},flush=True)
 del w,x,a,b,f,g;mx.clear_cache()
