import json,time,statistics,hashlib,gc,importlib,inspect
from pathlib import Path
import mlx.core as mx
import candidates as C
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing,clone_cache
from sglang.srt.hardware_backend.mlx.models import glm5_next as G
OUT=Path(__file__).parent
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model=load_mtp_model(Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'))
base=model.make_cache();engine=MlxMTPServing(model,2)
for start in range(0,2050,256):
 h=engine.prefill('base',(mx.arange(start,min(start+256,2050))%128)[None],base);mx.eval(h)
root=mx.argmax(model.lm_head(h[:,-1:]),-1);mx.eval(root)
# Capture3row real operands once; recorder absent during timing.
samples={};old_gate=G.mlx_affine_routed_swiglu;old_down=G.mlx_glm_down_combine
def gate_hook(*a,**kw):
 if kw['bits']==4 and a[0].shape[1]==3 and 'gate' not in samples:samples['gate']=(a,kw)
 return old_gate(*a,**kw)
def down_hook(*a,**kw):
 if a[0].shape[1]==3 and kw['bits'] not in samples:samples[kw['bits']]=(a,kw)
 return old_down(*a,**kw)
G.mlx_affine_routed_swiglu=gate_hook;G.mlx_glm_down_combine=down_hook
mx.disable_compile();scratch=[clone_cache(c) for c in base]
logits,hidden,snapshots=model.verify_mtp_rows(mx.array([[int(root.item()),10,11]]),scratch);mx.eval(logits,hidden)
G.mlx_affine_routed_swiglu=old_gate;G.mlx_glm_down_combine=old_down;mx.enable_compile()
report={'device':mx.device_info(),'records':[]}
def measure(label,fs,args,meta):
 ref=fs['baseline'](*args);mx.eval(ref)
 for name,fn in fs.items():
  result=fn(*args);mx.eval(result)
  assert mx.array_equal(ref,result).item(),(label,name,mx.max(mx.abs(ref-result)).item())
 raw=[]
 for group in range(3):
  for name in fs:
   if name=='baseline':continue
   for arm in ('baseline',name,name,'baseline'):
    mx.synchronize();t=time.perf_counter()
    for _ in range(6):mx.eval(fs[arm](*args))
    raw.append(dict(group=group,arm=arm,ms=(time.perf_counter()-t)*1000/6))
 row=dict(label=label,**meta,exact=True,raw=raw);report['records'].append(row)
 (OUT/'actual_probe.json').write_text(json.dumps(report,indent=2));print(label,meta,{n:round(statistics.median(r['ms'] for r in raw if r['arm']==n),4) for n in fs},flush=True)
args,kw=samples['gate'];x,ga,ua,ids=args[:4];grouped=C.D._group_picks(ids,3);mx.eval(grouped)
fs={'baseline':mx.compile(lambda x,g,u,i:old_gate(x,g,u,i,**kw))}
for kind in ('device','word','masks'):
 fs[kind]=mx.compile(lambda x,g,u,i,kind=kind:C.gate(x,g,u,i,kind=kind,grouped=grouped,group_size=kw['group_size'],limit=kw['limit']))
measure('gate',fs,(x,ga,ua,ids),dict(bits=4,group_size=kw['group_size'],limit=kw['limit'],route=ids.tolist()))
# Load/store arithmetic variants use exactly the same scheduler geometry.
for bits in (4,5,6):
 if bits not in samples:continue
 args,kw=samples[bits];a,i,sc,w,s,b,sh=args;grouped=C.D._group_picks(i,3);mx.eval(grouped)
 fs={'baseline':mx.compile(lambda a,i,sc,w,s,b,sh:old_down(a,i,sc,w,s,b,sh,**kw))}
 for kind in ('device','byte','threadgroup')+(('word','masks') if bits==4 else ()):
  fs[kind]=mx.compile(lambda a,i,sc,w,s,b,sh,kind=kind:C.down(a,i,sc,w,s,b,sh,bits=bits,kind=kind,grouped=grouped,group_size=kw['group_size']))
 measure('down',fs,args,dict(bits=bits,route=i.tolist()))
# Direct real gate/down transaction includes one shared metadata computation.
args,kw=samples['gate'];x,ga,ua,ids=args[:4]
layer=next(l for l in model.layers if isinstance(l.mlp,G.MoE) and l.mlp.switch_mlp.gate_proj.weight is ga[0]) if False else None
# Array handles captured in tracing can be views: use actual per-bit sample
# for combined cost with its own original x and route, never invent a match.
# Actual checkpoint vocabulary weights, draft1 and target3 hidden rows.
for count in (1,3):
 xx=hidden[:,:count];w=model.lm_head.weight
 fs={'baseline':mx.compile(lambda x,w:C.H.mlx_dense_argmax_rows(x,w))}
 for bm,rps,vec in ((4,4,False),(4,8,False),(8,4,True),(4,4,True)):
  fs[f'{bm}_{rps}_{vec}']=mx.compile(lambda x,w,bm=bm,rps=rps,vec=vec:C.head(x,w,bm=bm,rps=rps,vector=vec))
 measure('head',fs,(xx,w),dict(count=count))
# Retain review two parallel tiles, actual mHC projection weights.
hc=importlib.import_module('sglang.kernels.ops.layernorm.mlx_hc_mix')
source=(OUT.parent/'mechanism_push_review/mhc_parallel_tiles.metal').read_text()
kern=mx.fast.metal_kernel(name='review_parallel_real_mhc',input_names=['x','weight','eps'],output_names=['mix'],source=source)
def parallel(x,w):
 count=x.shape[1];k=4*x.shape[-1]
 return kern(inputs=[x,w,mx.array([1e-5],mx.float32)],template=[('K',k),('RMS_THREADS',min(k//4,1024)),('TILES',2)],grid=(1536,count,1),threadgroup=(512,1,1),output_shapes=[(1,count,24)],output_dtypes=[mx.float32])[0]
layer=model.layers[0];print('MHC candidates',[(k,v.shape) for k,v in layer.items() if isinstance(v,mx.array)],flush=True)
for key in ('hc_attn_fn','hc_ffn_fn'):
 if key not in layer:continue
 w=layer[key]
 for count in (1,3):
  xx=mx.repeat(hidden[:,:count,None,:],4,2)
  fs={'baseline':mx.compile(lambda x,w:hc.mlx_hc_rms_mix(x,w,eps=1e-5)),'parallel2':mx.compile(parallel)}
  measure('mhc',fs,(xx,w),dict(count=count,weight=key))
(OUT/'actual_probe.json').write_text(json.dumps(report,indent=2))
