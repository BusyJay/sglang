import json,time,statistics
from pathlib import Path
import mlx.core as mx
import candidates as C
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.models import glm5_next as G
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
OUT=Path(__file__).parent;mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();e=MlxMTPServing(model,2)
for start in range(0,2050,256):h=e.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root)
samples=[];old=G.MoE.__call__
def capture(self,x,**kw):
 if x.shape[1]==3 and self.switch_mlp.gate_proj.bits==4 and len(samples)<4:samples.append((self,x))
 return old(self,x,**kw)
G.MoE.__call__=capture;mx.disable_compile();model.verify_mtp_rows(mx.array([[int(root.item()),10,11]]),[clone_cache(c) for c in base]);G.MoE.__call__=old;mx.enable_compile()
rows=[]
for index,(block,x) in enumerate(samples):
 sw=block.switch_mlp;ga=tuple(sw.gate_proj[k] for k in ('weight','scales','biases'));ua=tuple(sw.up_proj[k] for k in ('weight','scales','biases'));d=sw.down_proj
 ids,scores=G.mlx_glm_router(x,block._router_weight,block.e_score_correction_bias,top_k=8,scaling=block.args.routed_scaling_factor,renormalize=block.args.norm_topk_prob)
 local=x.astype(ga[1].dtype);sh=block.shared_experts(x,batch_fusion=True);mx.eval(local,sh,ids,scores);limit=sw.activation.limit
 def baseline(x,i,sc,sh):
  act=C.A.mlx_affine_routed_swiglu(x,ga,ua,i,bits=4,group_size=sw.gate_proj.group_size,limit=limit)
  return C.D.mlx_glm_down_combine(act,i,sc,d.weight,d.scales,d.biases,sh,bits=d.bits,group_size=d.group_size)
 def compact(x,i,sc,sh):
  meta=C.compact_group(i);act=C.compact_gate(x,ga,ua,i,meta,limit=limit,group_size=sw.gate_proj.group_size)
  return C.D.mlx_glm_down_combine(act,i,sc,d.weight,d.scales,d.biases,sh,bits=d.bits,group_size=d.group_size)
 def reuse(x,i,sc,sh):
  meta=C.D._group_picks(i,3);act=C.gate(x,ga,ua,i,kind='masks',grouped=meta,limit=limit,group_size=sw.gate_proj.group_size)
  return C.down(act,i,sc,d.weight,d.scales,d.biases,sh,bits=d.bits,group_size=d.group_size,kind='device',grouped=meta)
 fs={k:mx.compile(f) for k,f in [('baseline',baseline),('compact',compact),('reuse',reuse)]};args=(local,ids,scores,sh);ref=fs['baseline'](*args);mx.eval(ref)
 for name,f in fs.items():assert mx.array_equal(ref,f(*args)).item(),(index,name)
 raw=[]
 for group in range(3):
  for name in ('compact','reuse'):
   for arm in ('baseline',name,name,'baseline'):
    t=time.perf_counter()
    for _ in range(8):mx.eval(fs[arm](*args))
    raw.append(dict(group=group,arm=arm,ms=(time.perf_counter()-t)*1000/8))
 rows.append(dict(index=index,down_bits=d.bits,route=ids.tolist(),exact=True,raw=raw));(OUT/'transaction_probe.json').write_text(json.dumps(rows,indent=2));print(index,d.bits,{n:round(statistics.median(r['ms'] for r in raw if r['arm']==n),4) for n in fs},flush=True)
