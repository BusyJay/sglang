from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm import mlx_fp8_rows as op
import time,json,statistics
r=Path('benchmark/mlx/ar_optimization_20261103')
old=op._QUANT
start=old.index('    uint lo =');end=old.index('    out[row',start)
new=old[:start]+'''    uint code;
    if (v < 0.015625f) {
        code = uint(rint(v * 512.0f));
    } else {
        uint bits = as_type<uint>(v);
        code = ((bits + 0x7ffffu + ((bits >> 20) & 1u)) >> 20) - 960u;
    }
'''+old[end:]
k=mx.fast.metal_kernel(name='diagnostic_fp8_direct_round',input_names=['x'],output_names=['out','scales'],source=new,header=op._DECODE)
def direct(x):
 ksize=x.shape[-1];count=x.size//ksize
 return tuple(k(inputs=[x.reshape(count,ksize)],template=[('K',ksize)],grid=(count*32,ksize//32,1),threadgroup=(32,1,1),output_shapes=[(count,ksize),(count,ksize//32)],output_dtypes=[mx.uint8,mx.float32]))
if __name__=='__main__':
 result={'checks':[],'timing':[]}
 mx.random.seed(37)
 for dtype in (mx.float32,mx.bfloat16,mx.float16):
  # exact representable centers/midpoints and neighbors, random exponent coverage
  codes=mx.arange(127).astype(mx.float32);exp=mx.floor(codes/8);mant=codes%8;centers=mx.where(exp==0,mant/512,(1+mant/8)*mx.power(2.0,exp-7));mid=(centers[:-1]+centers[1:])/2
  values=mx.concatenate([centers,mid,mid*(1-2**-23),mid*(1+2**-23),mx.random.uniform(-448,448,(100000,)),mx.power(2.0,mx.random.uniform(-120,120,(100000,)))])
  # each group contains anchor448 to force scale1 for boundary checks
  values=values[:values.size//31*31].reshape(-1,31);x=mx.concatenate([values,mx.full((values.shape[0],1),448)],axis=1).astype(dtype)
  for xx in (x,-x,mx.random.normal((9,16384)).astype(dtype),mx.zeros((1,512),dtype)):
   a=direct(xx);b=op.mlx_fp8_dynamic_32(xx);mx.eval(a,b)
   row={'dtype':str(dtype),'shape':xx.shape,'codes_equal':mx.array_equal(a[0],b[0]).item(),'scales_equal':mx.array_equal(a[1].view(mx.uint32),b[1].view(mx.uint32)).item()};result['checks'].append(row);print('CHECK',row,flush=True)
   if not row['codes_equal'] or not row['scales_equal']:raise AssertionError(row)
 for size in (1536,2048,4096,16384):
  x=mx.random.normal((1,size));calls={'old':mx.compile(op.mlx_fp8_dynamic_32),'direct':mx.compile(direct)}
  for fn in calls.values():mx.eval(fn(x))
  for repeat in range(20):
   for label in (('old','direct') if repeat%2==0 else ('direct','old')):
    t=time.perf_counter();mx.eval(calls[label](x));result['timing'].append({'size':size,'arm':label,'s':time.perf_counter()-t})
 for size in (1536,2048,4096,16384):
  print('TIME',size,{arm:statistics.median(t['s'] for t in result['timing'] if t['size']==size and t['arm']==arm)*1e6 for arm in ('old','direct')},flush=True)
 (r/'quant_direct.json').write_text(json.dumps(result,indent=2))
