"""M5 actual checkpoint routed FFNs: fused-activation and reuse geometry.

Same activation, route, scale, clamped SwiGLU and outputs. 300 calls per mode,
interleaved every ten calls; synchronization is per FFN, not model throughput.
"""
import hashlib,json,statistics,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.gemm import mlx_nvfp4_swiglu as sw
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
from functools import cache
O=Path('benchmark/mlx/remote_validation_20261103');C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(10*2**30);mx.set_cache_limit(256*2**20);inputs=mx.load(O/'actual_ffn_operands.npz')
@cache
def static_kernel():
 source=nv._SOURCE.replace('const int K=dims[0],N=dims[1],PICKS=dims[2];','const int PICKS=dims[2];')
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_static_width',input_names=['x','weight','scales','global_scale','ids','dims','partners','counts'],output_names=['out'],source=source,header=nv._HEADER,compile_options={'math_mode':'safe'})
def reuse(value,p,ids,work,rps):
 w,s,gs=p;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 return static_kernel()(inputs=[value.reshape(-1,k),w,s,gs,ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',rps),('K',k),('N',n),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
def fused(value,g,u,ids,rps):
 w,s,gs=g;uw,us,ugs=u;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 return sw._kernel()(inputs=[value.reshape(-1,k),w,s,gs,uw,us,ugs,ids.reshape(-1),mx.array([k,n,picks],mx.int32),mx.array([7.0],mx.float32)],template=[('T',value.dtype),('RPS',rps),('K',k),('N',n),('EXPERT',value.size//k==count),('CLAMP',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
def native(value,p,ids):
 w,s,gs=p
 return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs)
out=dict(scope=__doc__,device=mx.device_info(),cases=[])
for layer in (26,35):
 a=mx.load(C/f'layer-{layer:02}.safetensors');bank=[]
 for name in ('gate_proj','up_proj','down_proj'):
  prefix='mlp.switch_mlp.'+name+'.';w,s,gs=[a[prefix+n] for n in ('weight','scales','weight_scale_2')];bank.append((w,s,gs*2688))
 mx.eval(bank);del a
 for count in (4,):
  x=inputs[f'c{count}_l{layer}_x'];ids=inputs[f'c{count}_l{layer}_ids'];work=mlx_expert_pair_worklist(ids);mx.eval(*work)
  def ffn(value,route,mode):
   ids=route
   work=mlx_expert_pair_worklist(ids) if mode.startswith('reuse') else None
   v=value[...,None,None,:];g,u,d=bank
   if mode.startswith('fused'):act=fused(v,g,u,ids,int(mode[-1]))
   else:
    g,u=[reuse(v,p,ids,work,int(mode[-1])) if mode.startswith('reuse') else native(v,p,ids) for p in (g,u)]
    act=nn.silu(mx.minimum(g,7.0))*mx.clip(u,-7.0,7.0)
   return (reuse(act,d,ids,work,int(mode[-1])) if mode.startswith('reuse') else native(act,d,ids)).squeeze(-2)
  modes=('native','reuse1','reuse2','reuse4')
  fs={mode:mx.compile(lambda value,route,mode=mode:ffn(value,route,mode)) for mode in modes}
  ref=fs['native'](x,ids)
  for mode,fn in fs.items():
   v=fn(x,ids);mx.eval(ref,v);assert mx.array_equal(ref,v).item(),(layer,count,mode);assert mx.all(mx.isfinite(v)).item()
  raw={mode:[] for mode in modes}
  for block in range(30):
   for mode in modes if block%2==0 else reversed(modes):
    for step in range(10):
     value=x+mx.array((block*10+step)*1e-8,x.dtype);mx.eval(value);t=time.perf_counter();v=fs[mode](value,ids);mx.eval(v);raw[mode].append(time.perf_counter()-t)
  row=dict(layer=layer,count=count,samples=[dict(mode=mode,raw_seconds=raw[mode],mean_ms=statistics.mean(raw[mode])*1000) for mode in modes]);out['cases'].append(row)
  (O/'ffn_static_width.json').write_text(json.dumps(out,indent=2));print('FFN_STATIC_WIDTH',layer,count,{m:round(statistics.mean(raw[m])*1000,4) for m in modes},flush=True)
 del bank,fs;mx.clear_cache()
out['complete']=True;(O/'ffn_static_width.json').write_text(json.dumps(out,indent=2))
