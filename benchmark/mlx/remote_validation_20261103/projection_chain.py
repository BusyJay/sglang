"""Attribute M5 reuse regression separately to gate, up and down operands."""
import hashlib,json,statistics,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
O=Path('benchmark/mlx/remote_validation_20261103');C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(8*2**30);mx.set_cache_limit(256*2**20);inputs=mx.load(O/'actual_ffn_operands.npz')
def reuse(value,p,ids,work,rps):
 w,s,gs=p;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 return nv._kernel(False)(inputs=[value.reshape(-1,k),w,s,gs,ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',rps),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
def native(value,p,ids):
 w,s,gs=p
 return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs)
out=dict(scope=__doc__,device=mx.device_info(),cases=[])
for layer in (3,26,35,41):
 a=mx.load(C/f'layer-{layer:02}.safetensors');bank=[]
 for name in ('gate_proj','up_proj','down_proj'):
  prefix='mlp.switch_mlp.'+name+'.';w,s,gs=[a[prefix+n] for n in ('weight','scales','weight_scale_2')];bank.append((w,s,gs*2688))
 mx.eval(bank);del a
 x=inputs[f'c4_l{layer}_x'][...,None,None,:];ids=inputs[f'c4_l{layer}_ids'];work=mlx_expert_pair_worklist(ids);mx.eval(*work)
 g,u=[native(x,p,ids) for p in bank[:2]];act=nn.silu(mx.minimum(g,7.0))*mx.clip(u,-7.0,7.0);mx.eval(act)
 for projection,p,value in zip(('gate','up','down'),bank,(x,x,act)):
  fs={'native':mx.compile(lambda value:native(value,p,ids))}
  for rps in (1,2,4,8):fs['reuse'+str(rps)]=mx.compile(lambda value,rps=rps:reuse(value,p,ids,work,rps))
  ref=fs['native'](value)
  for mode,fn in fs.items():
   v=fn(value);mx.eval(ref,v);assert mx.array_equal(ref,v).item(),(layer,projection,mode)
  raw={mode:[] for mode in fs}
  for block in range(30):
   for mode in list(fs) if block%2==0 else list(reversed(fs)):
    for step in range(10):
     inp=value+mx.array((block*10+step)*1e-8,value.dtype);mx.eval(inp);t=time.perf_counter();v=fs[mode](inp);mx.eval(v);raw[mode].append(time.perf_counter()-t)
  row=dict(layer=layer,projection=projection,shape=value.shape,samples=[dict(mode=mode,mean_ms=statistics.mean(raw[mode])*1000,raw_seconds=raw[mode]) for mode in fs]);out['cases'].append(row)
  print('PROJECTION_CHAIN',layer,projection,{m:round(statistics.mean(raw[m])*1000,4) for m in fs},flush=True)
  (O/'projection_chain.json').write_text(json.dumps(out,indent=2))
 del bank,fs;mx.clear_cache()
out['complete']=True;(O/'projection_chain.json').write_text(json.dumps(out,indent=2))
