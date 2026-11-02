import json,time,types
from pathlib import Path
import mlx.core as mx
import importlib
op=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows');src=Path(op.__file__).read_text()
start=src.index('    uint lo = 0, hi = 126;');end=src.index('    out[row * K',start)
replacement="""    uint code;
    if (v < 0.015625f) {
        code = uint(rint(v * 512.0f));
    } else {
        uint bits = as_type<uint>(v);
        int exponent = int((bits >> 23) & 255) - 127;
        uint mantissa = bits & 0x7fffff;
        code = uint((exponent + 7) * 8) + (mantissa >> 20);
        uint tail = mantissa & 0xfffff;
        code += uint(tail > 0x80000 || (tail == 0x80000 && (code & 1)));
        code = min(code, 126u);
    }
"""
new=src[:start]+replacement+src[end:];new=new.replace('sglang_fp8_dynamic_32','sglang_fp8_dynamic_32_bits');mod=types.ModuleType('fast_quant');exec(compile(new,'fast_quant','exec'),mod.__dict__)
# all finite codes, exact midpoints and adjacent float32 numbers, zero signs and random values.
import numpy as np
codes=np.arange(127,dtype=np.uint8);mag=codes.astype(np.int32);e=mag>>3;mant=mag&7;v=np.where(e==0,mant*2**-9,(1+mant/8)*2.0**(e-7)).astype(np.float32);mid=(v[:-1]+v[1:])/2;inputs=np.concatenate([v,mid,np.nextafter(mid,np.float32(-np.inf)),np.nextafter(mid,np.float32(np.inf)),[0.0,-0.0,448.,449.]])
inputs=np.concatenate([inputs,-inputs]);inputs=np.pad(inputs,(0,(-len(inputs))%32));x=mx.array(inputs.reshape(-1,32));a=op.mlx_fp8_dynamic_32(x);b=mod.mlx_fp8_dynamic_32(x);mx.eval(a,b);assert all(mx.array_equal(i,j).item() for i,j in zip(a,b))
r=[]
for count,k in ((1,4096),(3,4096),(3,16384),(256,4096)):
 for dtype in (mx.bfloat16,mx.float32):
  x=(mx.random.normal((count,k))*50).astype(dtype);mx.eval(x);a=op.mlx_fp8_dynamic_32(x);b=mod.mlx_fp8_dynamic_32(x);mx.eval(a,b);assert all(mx.array_equal(i,j).item() for i,j in zip(a,b));trials=[]
  for trial in range(6):
   row={}
   for name,fn in ([('old',op.mlx_fp8_dynamic_32),('bits',mod.mlx_fp8_dynamic_32)] if trial%2==0 else [('bits',mod.mlx_fp8_dynamic_32),('old',op.mlx_fp8_dynamic_32)]):
    t=time.perf_counter()
    for _ in range(30):mx.eval(fn(x))
    row[name]=(time.perf_counter()-t)*1000/30
   trials.append(row)
  row={'count':count,'k':k,'dtype':str(dtype),'trials':trials};r.append(row);print(row,flush=True)
Path('benchmark/mlx/fp4_optimization_20261103/quant_bits.json').write_text(json.dumps(r,indent=2));Path('benchmark/mlx/fp4_optimization_20261103/quant_bits_candidate.py').write_text(new)
