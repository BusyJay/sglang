import json,time,hashlib,subprocess,collections
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_fp4 as f
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
root=Path('benchmark/mlx/fp4_optimization_20261103');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
mx.set_memory_limit(188*2**30);mx.set_wired_limit(188*2**30);mx.set_cache_limit(64*2**20)
r={'device':mx.device_info(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'samples':[],'micro':[],'attribution':[],'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
def save(): (root/'pilot.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
t=time.perf_counter();m,_=f.load_fp4_model(target,enable_mtp=True);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());r['load_s']=time.perf_counter()-t;print('LOADED',r['load_s'],mx.get_active_memory()/2**30,flush=True)
tok=load_tokenizer(target);engine=MlxMTPServing(m,2)
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
def generate(text,count=64):
 engine.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True);h=engine.prefill('p',mx.array([p],mx.int32),cache);mx.eval(h,[c.cache for c in cache]);token=int(m.target_tokens(h[:,-1:]).item());ids=[token];times=[]
 for i in range(count-1):
  engine.set_remaining('p',count-i-1);t=time.perf_counter();v=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.cache for c in cache]);token=int(v.item());times.append(time.perf_counter()-t);ids.append(token)
 row={'prompt':p,'ids':ids,'raw_s':times,'tok_s':len(times)/sum(times),'stats':engine.stats(),'active':mx.get_active_memory(),'peak':mx.get_peak_memory()};engine.remove('p');return row
for j,text in enumerate(texts):
 for trial in range(2):
  v=generate(text);v.update(text_id=j,trial=trial);r['samples'].append(v);save();print('BASE',j,trial,v['tok_s'],v['stats'],flush=True)
# Diagnostic serial drains attribute individual projection execution, destroying overlap.
labels={id(mod):name for name,mod in m.named_modules()};original=f.FP8Linear.project;events=[]
def prof(self,x):
 mx.eval(x);mx.synchronize();t=time.perf_counter();v=original(self,x);mx.eval(v);mx.synchronize();events.append({'module':labels.get(id(self),'unknown'),'block':self.block_size,'shape':list(x.shape),'ms':(time.perf_counter()-t)*1000});return v
f.FP8Linear.project=prof
try: generate(texts[0],10)
finally:f.FP8Linear.project=original
r['attribution']=events;summary=collections.defaultdict(float)
for e in events:summary['head' if e['module']=='lm_head' else 'mxfp8' if e['block']==32 else 'blockfp8']+=e['ms']
print('SYNC_PROJECTION_MS',dict(summary),'events',len(events),flush=True);save()
for name,p in [('q',m.model.layers[0].self_attn.q_proj),('head',m.lm_head),('shared',m.model.layers[3].mlp.shared_experts.gate_proj)]:
 for count in (1,3):
  x=mx.random.normal((count,p.weight.shape[-1])).astype(mx.bfloat16);mx.eval(x);reference=p(x)
  arms={'current':lambda:p(x)}
  if p.block_size==32:
   for tile in (1,2,4,8):
    def run(tile=tile):
     blocks=[]
     for start in range(0,count,tile):
      v=x[start:start+tile];n=v.shape[0]
      if n<tile:v=mx.concatenate([v,mx.zeros((tile-n,v.shape[-1]),v.dtype)],0)
      blocks.append(mx.quantized_matmul(v,p.weight.view(mx.uint32),p.scales,mode='mxfp8',group_size=32,bits=8)[:n])
     return mx.concatenate(blocks,0)
    arms['tile'+str(tile)]=run
   arms['simd']=lambda:f.mlx_fp8_rows(x,p.weight,p.scales)
  values={a:fn() for a,fn in arms.items()};mx.eval(values,reference);diff={a:float(mx.max(mx.abs(v.astype(mx.float32)-reference.astype(mx.float32))).item()) for a,v in values.items()};trials=[]
  for trial in range(4):
   row={}
   for a,fn in list(arms.items())[::1 if trial%2==0 else -1]:
    mx.eval(fn());t=time.perf_counter()
    for _ in range(4):mx.eval(fn())
    row[a]=(time.perf_counter()-t)*1000/4
   trials.append(row)
  v={'name':name,'shape':list(p.weight.shape),'count':count,'maxdiff':diff,'trials_ms':trials};r['micro'].append(v);save();print('MICRO',v,flush=True)
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
