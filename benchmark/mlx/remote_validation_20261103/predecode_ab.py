"""Actual shared FP8 FFN: lossless predecoded weights vs packed bytes."""
import hashlib,json,statistics,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from predecode_fp8 import decode,project
from sglang.kernels.ops.gemm import mlx_block_fp8 as fp
from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_dynamic_32,mlx_fp8_rows
O=Path('benchmark/mlx/remote_validation_20261103');C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(2*2**30);mx.set_cache_limit(128*2**20);inputs=mx.load(O/'actual_ffn_operands.npz');out=dict(scope=__doc__,device=mx.device_info(),cases=[],prep_seconds=[])
for layer in (3,26,41):
 a=mx.load(C/f'layer-{layer:02}.safetensors');p='mlp.shared_experts.'
 gw,gs,uw,us,dw,ds=[a[p+name] for name in ('gate_proj.weight','gate_proj.scales','up_proj.weight','up_proj.scales','down_proj.weight','down_proj.scales')]
 t=time.perf_counter();hw,hu=decode(gw),decode(uw);mx.eval(gw,gs,uw,us,dw,ds,hw,hu);out['prep_seconds'].append(time.perf_counter()-t);del a
 for count in (1,4):
  x=inputs[f'c{count}_l{layer}_x']
  def ffn(value,mode):
   q=mlx_fp8_dynamic_32(value)
   if mode=='separate':g,u=[mlx_fp8_rows(value,w,s,block_size=128,quantized_input=q) for w,s in ((gw,gs),(uw,us))]
   elif mode=='packed':g,u=fp.mlx_block_fp8_pair(value,gw,gs,uw,us,q)
   else:g,u=project(value,hw,gs,hu,us,q,rps=int(mode[-1]))
   act=nn.silu(mx.minimum(g,7.0))*mx.clip(u,-7.0,7.0)
   return mlx_fp8_rows(act,dw,ds,block_size=128,quantized_input=mlx_fp8_dynamic_32(act))
  fs={m:mx.compile(lambda value,m=m:ffn(value,m)) for m in ('separate','packed','half1','half2','half4')}
  ref=fs['separate'](x)
  for name,fn in fs.items():
   v=fn(x);mx.eval(v,ref);assert mx.array_equal(ref,v).item(),(layer,count,name);assert mx.all(mx.isfinite(v)).item()
  raw={m:[] for m in fs}
  for block in range(30):
   for name in list(fs) if block%2==0 else list(reversed(fs)):
    for j in range(10):
     value=x+mx.array((block*10+j)*1e-8,x.dtype);mx.eval(value);t=time.perf_counter();v=fs[name](value);mx.eval(v);raw[name].append(time.perf_counter()-t)
  row=dict(layer=layer,count=count,samples=[dict(mode=m,raw_seconds=raw[m],mean_ms=statistics.mean(raw[m])*1000) for m in fs]);out['cases'].append(row)
  (O/'predecode_ab.json').write_text(json.dumps(out,indent=2));print('PREDECODE_AB',layer,count,{m:round(statistics.mean(raw[m])*1000,4) for m in fs},'extraGiB',(hw.nbytes+hu.nbytes)/2**30,flush=True)
 del gw,gs,uw,us,dw,ds,hw,hu,fs;mx.clear_cache()
out['complete']=True;(O/'predecode_ab.json').write_text(json.dumps(out,indent=2))
