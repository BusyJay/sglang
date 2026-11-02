import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows
source="\nconst uint sg=simdgroup_index_in_threadgroup;\nconst uint col=(threadgroup_position_in_grid.x*SG+sg)*32;\nconst uint part=threadgroup_position_in_grid.y;\nconstexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,32,false,true,true,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\nmpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;\nauto a=op.template get_left_input_cooperative_tensor<T,T,float>();\nauto b=op.template get_right_input_cooperative_tensor<T,T,float>();\nauto c=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();\nfor(uint i=0;i<c.get_capacity();++i)c[i]=0;\nconstexpr uint STEP=K/SPLIT;\nfor(uint k=part*STEP;k<(part+1)*STEP;k+=32){\n for(uint i=0;i<a.get_capacity();++i){auto p=a.get_multidimensional_index(i);a[i]=p[1]<M?x[p[1]*K+k+p[0]]:T(0);}\n for(uint i=0;i<b.get_capacity();++i){auto p=b.get_multidimensional_index(i);b[i]=col+p[1]<N?w[size_t(col+p[1])*K+k+p[0]]:T(0);}\n op.run(a,b,c);\n}\nfor(uint i=0;i<c.get_capacity();++i){auto p=c.get_multidimensional_index(i);if(p[1]<M&&col+p[0]<N)out[(size_t(part)*M+p[1])*N+col+p[0]]=c[i];}\n"
kernel=mx.fast.metal_kernel(name="sglang_dense_direct_split_nax",input_names=["x","w"],output_names=["out"],source=source,header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n")
mx.random.seed(20261102);records=[];out=Path("benchmark/mlx/na_optimization_20261102/dense_direct_probe.json")
for n,k in ((4096,8192),(24896,4096),(154880,4096)):
 w=mx.random.normal((n,k)).astype(mx.bfloat16)*0.01;mx.eval(w)
 for m in (3,4,8):
  x=mx.random.normal((1,m,k)).astype(mx.bfloat16);mx.eval(x)
  f=mx.compile(lambda x:mlx_dense_rows(x,w));a=f(x);mx.eval(a)
  for split in (1,4,8,16):
   for sg in (1,4):
    def candidate(x):
     v=kernel(inputs=[x,w],template=[("T",x.dtype),("M",m),("N",n),("K",k),("SPLIT",split),("SG",sg)],grid=(32*sg*((n+32*sg-1)//(32*sg)),split,1),threadgroup=(32*sg,1,1),output_shapes=[(split,m,n)],output_dtypes=[mx.float32])[0]
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
