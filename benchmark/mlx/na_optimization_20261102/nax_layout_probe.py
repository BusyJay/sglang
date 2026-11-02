import json,time,statistics
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_qmm_swiglu import _NAX_SOURCE,_expert_worklist,_nax_kernel
mx.random.seed(20261102)
out=Path("benchmark/mlx/na_optimization_20261102")
header=Path("python/sglang/kernels/ops/gemm/mlx_qmm_swiglu_nax.metal").read_text()
source=qsource=_NAX_SOURCE
conditions=[]
for name in ("gate_w","up_w"):
 for axis,stride in enumerate(("N*K*BITS/32","K*BITS/32","1")):
  conditions.append(f"{name}_strides[{axis}]=={stride}")
  source=source.replace(f"{name}_strides[{axis}]",f"({stride})")
for name in ("gate_s","gate_b","up_s","up_b"):
 for axis,stride in enumerate(("N*K/GS","K/GS","1")):
  conditions.append(f"{name}_strides[{axis}]=={stride}")
  source=source.replace(f"{name}_strides[{axis}]",f"({stride})")
shared="constexpr uint shared_pad=64+16/sizeof(T);\nthreadgroup T Gs[64*shared_pad],Us[64*shared_pad];\n"
source=source.replace(" threadgroup T Gs[64*pad],Us[64*pad];","")
qsource=qsource.replace(" threadgroup T Gs[64*pad],Us[64*pad];","")
source=shared+"if("+" && ".join(conditions)+"){\n"+source+"\n}else{\n"+qsource+"\n}"
kernel=mx.fast.metal_kernel(name="sglang_nax_layout_specialized",input_names=["x","work","work_count","gate_w","gate_s","gate_b","up_w","up_s","up_b","limit"],output_names=["out"],source=source,header=header,ensure_row_contiguous=False)
records=[]
for bits in (4,5,6):
 e,n,k=288,2048,4096
 ops=[]
 for _ in range(2):
  parts=[]
  for j in range(e):
   p=mx.quantize(mx.random.normal((1,n,k)).astype(mx.float16)*0.02,group_size=64,bits=bits);mx.eval(p);parts.append(p)
  operands=tuple(mx.concatenate([p[j] for p in parts],0) for j in range(3));mx.eval(operands);ops.append(operands)
 for m in (1152,2048):
  ids=mx.sort((mx.arange(m)%e).astype(mx.int32));x=mx.random.normal((m,1,k)).astype(mx.float16);work,count=_expert_worklist(ids,e,m,32);mx.eval(x,work,count)
  inputs=[x,work,count,*[a for p in ops for a in p],mx.array([0.],mx.float32)]
  kw=dict(template=[("T",x.dtype),("M",m),("N",n),("K",k),("GS",64),("BITS",bits),("HAS_LIMIT",False),("BM",32)],grid=(128*(n//64),(m+31)//32+e-1,1),threadgroup=(128,1,1),output_shapes=[(m,1,n)],output_dtypes=[x.dtype])
  f=mx.compile(lambda x:_nax_kernel()(inputs=[x,*inputs[1:]],**kw)[0]);g=mx.compile(lambda x:kernel(inputs=[x,*inputs[1:]],**kw)[0]);a=f(x);b=g(x);mx.eval(a,b)
  r=dict(bits=bits,rows=m,exact=mx.array_equal(a,b).item(),max_abs=mx.max(mx.abs(a-b)).item(),raw=[])
  for group in range(5):
   for name,fn in (("A",f),("B",g),("B",g),("A",f)):
    start=time.perf_counter()
    for j in range(5):mx.eval(fn(x))
    r["raw"].append(dict(group=group,mode=name,ms=(time.perf_counter()-start)*200))
  r["medians"]={name:statistics.median(z["ms"] for z in r["raw"] if z["mode"]==name) for name in ("A","B")}
  records.append(r);(out/"nax_layout_probe.json").write_text(json.dumps(records,indent=2));print({a:b for a,b in r.items() if a!="raw"},flush=True)
 del ops,inputs,a,b,x,f,g;mx.synchronize();mx.clear_cache()
