"""Actual cached layer, real routes, exact FP8 tiles and NVFP4 GPU diagnostics."""
import json,statistics,time,subprocess,os
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_block_fp8 import _pair
from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_rows,mlx_fp8_dynamic_32
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
O=Path('benchmark/mlx/remote_validation_20261103')
C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(8*2**30);mx.set_cache_limit(256*2**20)
a=mx.load(C/'layer-03.safetensors');mx.eval(list(a.values()));inputs=mx.load(O/'actual_ffn_operands.npz');data=dict(device=mx.device_info(),samples=[],resident=mx.get_active_memory())
x4=inputs['c4_l3_x'];x1=inputs['c1_l3_x']
gw,gs,uw,us=[a['mlp.shared_experts.'+n] for n in ('gate_proj.weight','gate_proj.scales','up_proj.weight','up_proj.scales')]
print('GEO_READY',mx.get_active_memory()/2**30,flush=True)
def bench(fn,x,n=100):
 raw=[];mx.eval(fn(x))
 for j in range(n):
  value=x+mx.array(j*1e-8,x.dtype);mx.eval(value);t=time.perf_counter();v=fn(value);mx.eval(v);raw.append(time.perf_counter()-t)
 return raw
for c,x in ((1,x1),(4,x4)):
 def sep(value):
  q=mlx_fp8_dynamic_32(value)
  return [mlx_fp8_rows(value,w,s,block_size=128,quantized_input=q) for w,s in ((gw,gs),(uw,us))]
 fns={'separate':mx.compile(sep)}
 for rps in (1,2,4):
  fns['pair'+str(rps)]=mx.compile(lambda value,r=rps:_pair(value,gw,gs,uw,us,mlx_fp8_dynamic_32(value),output_rows=r))
 ref=fns['separate'](x)
 for name in list(fns)+list(reversed(fns)):
  v=fns[name](x);mx.eval(ref,v);assert all(mx.array_equal(p,q).item() for p,q in zip(v,ref)),(c,name)
  raw=bench(fns[name],x);data['samples'].append(dict(count=c,mode=name,raw=raw));print('FP8GEO',c,name,round(statistics.mean(raw)*1000,4),flush=True)
ids=inputs['c4_l3_ids'];x=x4[...,None,None,:];prefix='mlp.switch_mlp.gate_proj.'
w,s,glob=[a[prefix+n] for n in ('weight','scales','weight_scale_2')];work=mlx_expert_pair_worklist(ids);mx.eval(*work)
def native(value):return mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=glob*2688)
def paired(value):return nv.mlx_nvfp4_gather_rows(value,w,s,glob*2688,ids,worklist=work)
fs={'native':mx.compile(native),'paired':mx.compile(paired)}
for mode in fs:mx.eval(fs[mode](x))
data['complete']=True;(O/'m5_geometry_timing_only.json').write_text(json.dumps(data,indent=2))
