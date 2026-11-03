"""M5 exact FFN macro boundary: runtime parameters and complete MoE.

All 42 checkpoint banks resident, captured actual c4 activations/ids. Each step
sweeps all layers with async submission and final sync. Not model throughput.
"""
import hashlib, importlib.util, json, statistics, subprocess, sys, time
from pathlib import Path
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
from sglang.kernels.ops.moe import mlx_expert_pairs as ep
from functools import cache
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/remote_validation_20261103'
s=importlib.util.spec_from_file_location('sglang.srt.hardware_backend.mlx.models.envelope_baseline',O/'baseline/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
budget=int(mx.device_info()['max_recommended_working_set_size']*.78)
mx.set_memory_limit(budget);mx.set_wired_limit(budget);mx.set_cache_limit(1024**3)
target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
m,_=load_fp4_model(target,enable_mtp=False);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.configure_eval_groups(decode=2,prefill=4,verification=2);m.prepare_for_inference();mx.eval(m.parameters());e=MlxDFlashServing(m,draft,3)
inputs=mx.load(O/'actual_ffn_operands.npz')
@cache
def strided_kernel():
 source=nv._SOURCE
 for key in ('pick','i'):
  source=source.replace('ids['+key+']','ids[('+key+'/dims[2])*ids_strides[ids_ndim-2]+'+key+'%dims[2]]')
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_strided_routes',input_names=['x','weight','scales','global_scale','ids','dims','partners','counts'],output_names=['out'],source=source,header=nv._HEADER,ensure_row_contiguous=False,compile_options={'math_mode':'safe'})
@cache
def strided_pairs_kernel():
 source=ep._SOURCE
 for key in ('pick','i'):
  source=source.replace('ids['+key+']','ids[('+key+'/dims[1])*ids_strides[ids_ndim-2]+'+key+'%dims[1]]')
 return mx.fast.metal_kernel(name='diagnostic_pairs_strided_routes',input_names=['ids','dims'],output_names=['partners','counts'],source=source,ensure_row_contiguous=False)
def strided_pairs(ids):
 return tuple(strided_pairs_kernel()(inputs=[ids,mx.array([ids.size,ids.shape[-1]],mx.uint32)],grid=(ids.size,1,1),threadgroup=(32,1,1),output_shapes=[(ids.size,)]*2,output_dtypes=[mx.uint32]*2))
class Reuse1(b.ClampedSwitchGLU):
 def __call__(self,x,ids,**kw):
  if not 1<x.shape[1]<=8:return b.ClampedSwitchGLU.__call__(self,x,ids,**kw)
  work=strided_pairs(ids) if getattr(self,'mode','reuse1')=='strided' else mlx_expert_pair_worklist(ids)
  def proj(value,p):
   if getattr(self,'mode','reuse1')=='gate_only' and p is self.down_proj or getattr(self,'mode','reuse1')=='down_only' and p is not self.down_proj:return p(value,ids)
   w,s,gs=p.weight,p.scales,p.weight_scale_2*2688;count,picks=ids.size,ids.shape[-1];_,n,pk=w.shape;k=pk*2
   return (strided_kernel() if getattr(self,'mode','reuse1')=='strided' else nv._kernel(False))(inputs=[value.reshape(-1,k),w,s,gs.reshape(-1),ids if getattr(self,'mode','reuse1')=='strided' else ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',1),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,n,count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
  gate,up=[proj(x[...,None,None,:],p) for p in (self.gate_proj,self.up_proj)]
  return proj(self.activation(up,gate),self.down_proj).squeeze(-2)
functions={};checks=[]
for scope in ('moe_runtime',):
 for mode in ('native','reuse1','strided'):
  rows=[]
  for i,l in enumerate(m.layers):
   if not isinstance(l.mlp,g.MoE):continue
   module=g._clone_graph_modules(l.mlp if scope=='moe_runtime' else l.mlp.switch_mlp)
   if scope=='moe_runtime':
    module.shared_experts.__class__=g.MLP;module.switch_mlp.__class__=b.ClampedSwitchGLU if mode=='native' else Reuse1
    module.switch_mlp.mode=mode
   else:module.__class__=b.ClampedSwitchGLU if mode=='native' else Reuse1
   x,ids=inputs[f'c4_l{i}_x'],inputs[f'c4_l{i}_ids'];parameters=module.parameters();router=l.mlp._router_weight
   def forward(value,route,weights,rw,module=module,scope=scope):
    if scope!='switch_closed':module.update(weights)
    if scope=='moe_runtime':
     module._router_weight=rw;module._router_sources=module._current_router_sources()
     return module(value,batch_fusion=True)
    return module(value,route)
   fn=mx.compile(forward)
   args=(x,ids,parameters if scope!='switch_closed' else {},router if scope=='moe_runtime' else None)
   y=fn(*args);mx.eval(y);assert mx.all(mx.isfinite(y)).item(),(scope,mode,i)
   if mode=='native':checks.append((scope,i,y))
   else:
    ref=next(v for sc,j,v in checks if sc==scope and j==i);assert mx.array_equal(ref,y).item(),(scope,mode,i)
   rows.append((fn,args))
  functions[(scope,mode)]=rows
print('ENVELOPE_READY',mx.get_active_memory()/2**30,flush=True)
data=dict(scope=__doc__,device=mx.device_info(),samples=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),resident_bytes=mx.get_active_memory(),swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True))
for scope in ('moe_runtime',):
 raw={mode:[] for mode in ('native','reuse1','strided')}
 for block in range(30):
  for mode in ('native','reuse1','strided') if block%2==0 else ('strided','reuse1','native'):
   for j in range(10):
    bump=mx.array((block*10+j)*1e-8,mx.float32);mx.eval(bump);t=time.perf_counter();ys=[]
    for fn,(x,ids,weights,rw) in functions[(scope,mode)]:
     y=fn(x+bump,ids,weights,rw);ys.append(y);mx.async_eval(y)
    mx.eval(ys);raw[mode].append(time.perf_counter()-t)
 for mode in raw:
  row=dict(scope=scope,mode=mode,raw_seconds=raw[mode],mean_ms=statistics.mean(raw[mode])*1000);data['samples'].append(row);print('ENVELOPE300',scope,mode,row['mean_ms'],flush=True)
 (O/'macro_strided.json').write_text(json.dumps(data,indent=2))
data['complete']=True;data['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);(O/'macro_strided.json').write_text(json.dumps(data,indent=2))
