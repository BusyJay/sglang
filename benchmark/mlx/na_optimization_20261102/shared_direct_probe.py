import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_swiglu_rows
source="\nusing namespace metal;\nconst uint lane=thread_index_in_simdgroup;\nconst uint sg=simdgroup_index_in_threadgroup;\nconst uint col=(threadgroup_position_in_grid.x*SG+sg)*32;\nconst uint part=threadgroup_position_in_grid.y;\nconstexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,32,false,true,true,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\nmpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;\nauto a=op.template get_left_input_cooperative_tensor<T,T,float>();\nauto b=op.template get_right_input_cooperative_tensor<T,T,float>();\nauto g=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();\nauto u=g;\nfor(uint i=0;i<g.get_capacity();++i){g[i]=0;u[i]=0;}\nconstexpr uint STEP=K/SPLIT;\nfor(uint k=part*STEP;k<(part+1)*STEP;k+=32){\n for(uint i=0;i<a.get_capacity();++i){\n  auto c=a.get_multidimensional_index(i);a[i]=c[1]<M?x[c[1]*K+k+c[0]]:T(0);\n }\n for(uint i=0;i<b.get_capacity();++i){\n  auto c=b.get_multidimensional_index(i);\n  const uint r=col+c[1],kk=k+c[0],bit=kk*BITS;\n  const size_t byte=size_t(r)*(K*BITS/8)+bit/8;\n  uint q=((const device uchar*)gate_w)[byte];\n  if((bit%8)+BITS>8)q|=uint(((const device uchar*)gate_w)[byte+1])<<8;\n  q=(q>>(bit%8))&((1u<<BITS)-1);\n  const uint group=kk/GS;\n  b[i]=T(float(q)*float(gate_s[size_t(r)*(K/GS)+group])+float(gate_b[size_t(r)*(K/GS)+group]));\n }\n op.run(a,b,g);\n for(uint i=0;i<b.get_capacity();++i){\n  auto c=b.get_multidimensional_index(i);\n  const uint r=col+c[1],kk=k+c[0],bit=kk*BITS;\n  const size_t byte=size_t(r)*(K*BITS/8)+bit/8;\n  uint q=((const device uchar*)up_w)[byte];\n  if((bit%8)+BITS>8)q|=uint(((const device uchar*)up_w)[byte+1])<<8;\n  q=(q>>(bit%8))&((1u<<BITS)-1);\n  const uint group=kk/GS;\n  b[i]=T(float(q)*float(up_s[size_t(r)*(K/GS)+group])+float(up_b[size_t(r)*(K/GS)+group]));\n }\n op.run(a,b,u);\n}\nfor(uint i=0;i<g.get_capacity();++i){\n auto c=g.get_multidimensional_index(i);\n if(c[1]<M){const size_t pos=(size_t(part)*M+c[1])*N+col+c[0];gate[pos]=g[i];up[pos]=u[i];}\n}\n"
kernel=mx.fast.metal_kernel(name="sglang_shared_direct_split_nax",input_names=["x","gate_w","gate_s","gate_b","up_w","up_s","up_b"],output_names=["gate","up"],source=source,header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n")
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
  f=mx.compile(lambda x:mlx_affine_swiglu_rows(x,*ops,bits=bits,group_size=64))
  ref=f(x);mx.eval(ref)
  for split in (1,4,8,16):
   for sg in (1,4):
    def candidate(x):
     g,u=kernel(inputs=[x,*[a for p in ops for a in p]],template=[("T",x.dtype),("M",m),("N",n),("K",k),("GS",64),("BITS",bits),("SPLIT",split),("SG",sg)],grid=(32*sg*(n//(32*sg)),split,1),threadgroup=(32*sg,1,1),output_shapes=[(split,m,n)]*2,output_dtypes=[mx.float32]*2)
     gate=mx.sum(g,0).astype(x.dtype);up=mx.sum(u,0).astype(x.dtype)
     return (mx.sigmoid(gate)*gate*up).reshape(1,m,n)
    g=mx.compile(candidate);t=time.perf_counter();got=g(x);mx.eval(got);cold=time.perf_counter()-t
    diff=got.astype(mx.float32)-ref.astype(mx.float32)
    row=dict(bits=bits,rows=m,split=split,sg=sg,max_abs=mx.max(mx.abs(diff)).item(),rel_l2=mx.sqrt(mx.sum(diff*diff)/mx.sum(ref.astype(mx.float32)**2)).item(),cold=cold,raw=[])
    for _ in range(2):mx.eval(f(x),g(x))
    for group in range(3):
     for name,fn in (("A",f),("B",g),("B",g),("A",f)):
      start=time.perf_counter()
      for _ in range(10):mx.eval(fn(x))
      row["raw"].append(dict(group=group,mode=name,ms=(time.perf_counter()-start)*100))
    row["medians"]={name:statistics.median(r["ms"] for r in row["raw"] if r["mode"]==name) for name in ("A","B")}
    records.append(row);(OUT/"shared_direct_probe.json").write_text(json.dumps(records,indent=2));print({a:b for a,b in row.items() if a!="raw"},flush=True)
 del ops
 mx.synchronize();mx.clear_cache()
