"""M5 diagnostics: actual activations/routes, full banks, per-layer FFN ABBA.

Capture uses the eager target for observability; synchronized per-layer timings
are attribution, not serving throughput. No checkpoint arrays are changed.
"""
import copy, hashlib, importlib.util, json, os, statistics, subprocess, sys, time
from collections import Counter
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/remote_validation_20261103'
target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
s=importlib.util.spec_from_file_location('sglang.srt.hardware_backend.mlx.models.replay_baseline',O/'baseline/python/sglang/srt/hardware_backend/mlx/models/glm5_next.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
budget=int(mx.device_info()['max_recommended_working_set_size']*.78)
mx.set_memory_limit(budget);mx.set_wired_limit(budget);mx.set_cache_limit(1024**3)
data=dict(device=mx.device_info(),note=__doc__,cases=[],projection_cases=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True))
def save(): (O/'rootcause_replay.json').write_text(json.dumps(data,indent=2))
t=time.perf_counter();m,_=load_fp4_model(target,enable_mtp=False);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.configure_eval_groups(decode=2,prefill=4,verification=2);m.prepare_for_inference();mx.eval(m.parameters())
e=MlxDFlashServing(m,draft,3);tok=load_tokenizer(target);print('REPLAY_READY',time.perf_counter()-t,flush=True)
records={};recording=False
class ProbeSwitch(g.ClampedSwitchGLU):
 def __call__(self,x,indices,**kw):
  if recording:records[self.probe_id]=dict(x=x.view(x.dtype),ids=indices.view(indices.dtype))
  return b.ClampedSwitchGLU.__call__(self,x,indices,**kw)
for i,l in enumerate(m.layers):
 if isinstance(l.mlp,g.MoE):
  l.mlp.switch_mlp.probe_id=i;l.mlp.switch_mlp.__class__=ProbeSwitch
  l.mlp.shared_experts.__class__=b.MLP
 else:l.mlp.__class__=b.MLP
m.configure_fixed_groups(False)
prompt=tok.apply_chat_template([{'role':'user','content':'Explain why the sky is blue.'}],tokenize=True,add_generation_prompt=True)
cache=m.make_cache();h=e.prefill('p',mx.array([prompt],mx.int32),cache);mx.eval(h,[c.state for c in cache]);token=int(m.target_tokens(h[:,-1:]).item())
for step in range(32):
 v=e.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.state for c in cache]);token=int(v.item())
# Force a new actual draft chain at the consumed cache boundary.
e.states['p'].rows.clear();recording=True
v=e.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.state for c in cache]);recording=False
captured4=records;records={}
recording=True
v=m.forward_last_tokens(mx.array([[int(v.item())]],mx.int32),cache);mx.eval(v,[c.state for c in cache]);recording=False
captured1=records;records={}
e.clear();del cache,h,v
for l in m.layers:
 if isinstance(l.mlp,g.MoE):l.mlp.switch_mlp.__class__=g.ClampedSwitchGLU
print('CAPTURED',len(captured1),len(captured4),flush=True)
# Exact values/routes are portable and small; full banks stay resident.
arrays={}
for count,values in ((1,captured1),(4,captured4)):
 for i,c in values.items():
  mx.eval(c['x'],c['ids']);arrays[f'c{count}_l{i}_x']=c['x'];arrays[f'c{count}_l{i}_ids']=c['ids']
mx.savez(O/'actual_ffn_operands.npz',**arrays)

def bench(fn,x,n=12):
 mx.eval(fn(x));raw=[];last=None
 for j in range(n):
  inp=x+mx.array(j*1e-8,x.dtype)
  mx.eval(inp);t=time.perf_counter();last=fn(inp);mx.eval(last);raw.append(time.perf_counter()-t)
 return raw,last

for count,captured in ((1,captured1),(4,captured4)):
 for i,c in captured.items():
  sw=m.layers[i].mlp.switch_mlp;x,ids=c['x'],c['ids']
  classes={'baseline':b.ClampedSwitchGLU,'candidate':g.ClampedSwitchGLU}
  def project(value,cls):
   return cls.__call__(sw,value,ids)
  fns={name:mx.compile(lambda value,cls=cls:project(value,cls)) for name,cls in classes.items()}
  a,z=fns['baseline'](x),fns['candidate'](x);mx.eval(a,z);assert mx.array_equal(a,z).item(),(count,i)
  route=ids.reshape(-1).tolist();ct=Counter(route);pairs=sum(v//2 for v in ct.values())
  row=dict(count=count,layer=i,dtype=str(x.dtype),shape=x.shape,route=route,pair_fraction=2*pairs/len(route),unique_experts=len(ct),repeat_sizes=list(ct.values()),samples=[],weight_shapes={n:getattr(sw,n).weight.shape for n in ('gate_proj','up_proj','down_proj')})
  for mode in ('baseline','candidate','candidate','baseline'):
   raw,_=bench(fns[mode],x);row['samples'].append(dict(mode=mode,raw_seconds=raw,mean=statistics.mean(raw)))
  old=statistics.mean(v['mean'] for v in row['samples'] if v['mode']=='baseline');new=statistics.mean(v['mean'] for v in row['samples'] if v['mode']=='candidate');row.update(baseline_ms=old*1000,candidate_ms=new*1000,ratio=new/old)
  # Replay shared FP8 FFN with the same live activation and checkpoint.
  shared=m.layers[i].mlp.shared_experts
  fshared={name:mx.compile(lambda value,cls=cls:cls.__call__(shared,value,batch_fusion=True)) for name,cls in [('baseline',b.MLP),('candidate',g.MLP)]}
  sa,sb=[fn(x) for fn in fshared.values()];mx.eval(sa,sb);assert mx.array_equal(sa,sb).item(),('shared',count,i)
  row['shared_samples']=[]
  for mode in ('baseline','candidate','candidate','baseline'):
   raw,_=bench(fshared[mode],x);row['shared_samples'].append(dict(mode=mode,raw_seconds=raw,mean=statistics.mean(raw)))
  data['cases'].append(row);save();print('FFN_REPLAY',count,i,row['dtype'],round(old*1000,3),round(new*1000,3),'pair',round(row['pair_fraction'],3),flush=True)
  sw.__class__=g.ClampedSwitchGLU
# Geometry/memory and pairing controls on six spread-out actual layers.
keys=list(captured4);indices=[keys[k] for k in (0,len(keys)//5,2*len(keys)//5,3*len(keys)//5,4*len(keys)//5,len(keys)-1)]
for i in indices:
 c=captured4[i];sw=m.layers[i].mlp.switch_mlp;x=c['x'][...,None,None,:];ids=c['ids'];p=sw.gate_proj
 work=mlx_expert_pair_worklist(ids);mx.eval(*work)
 native=lambda value: p(value,ids)
 explicit=lambda value:mlx_nvfp4_gather_rows(value,p.weight,p.scales,p.weight_scale_2*2688,ids,worklist=work)
 inline=lambda value:mlx_nvfp4_gather_rows(value,p.weight,p.scales,p.weight_scale_2*2688,ids)
 # All-single worklist isolates paired-register/control costs with the same banks.
 singles=(mx.arange(ids.size,dtype=mx.uint32),mx.ones((ids.size,),mx.uint32))
 no_share=lambda value:mlx_nvfp4_gather_rows(value,p.weight,p.scales,p.weight_scale_2*2688,ids,worklist=singles)
 fns={name:mx.compile(fn) for name,fn in [('native',native),('explicit',explicit),('inline',inline),('no_share',no_share)]}
 row=dict(layer=i,samples=[],dtype=str(x.dtype))
 reference=fns['native'](x)
 for mode in ('native','explicit','inline','no_share','no_share','inline','explicit','native'):
  result=fns[mode](x);mx.eval(reference,result);assert mx.array_equal(reference,result).item(),(i,mode)
  raw,_=bench(fns[mode],x,n=30);row['samples'].append(dict(mode=mode,raw_seconds=raw,mean=statistics.mean(raw)))
 data['projection_cases'].append(row);save();print('PROJECT_REPLAY',i,{name:round(statistics.mean(v['mean'] for v in row['samples'] if v['mode']==name)*1000,3) for name in fns},flush=True)
data['complete']=True;data['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
