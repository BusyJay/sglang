"""Actual M5 DFlash 300-token width A/B and separate synchronized diagnostics."""
import argparse,copy,hashlib,json,statistics,subprocess,time
from pathlib import Path
from collections import Counter
import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model,FP8Linear
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
ROOT=Path('/Users/jay/workspace/sglang');OUT=ROOT/'benchmark/mlx/dflash_verify_20261104'
TARGET=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
DRAFT=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
p=argparse.ArgumentParser();p.add_argument('--modes',default='reference,batch_fp8');p.add_argument('--rounds',type=int,default=2);p.add_argument('--prompts',type=int,default=3);p.add_argument('--tokens',type=int,default=300);args=p.parse_args()
mx.set_memory_limit(int(mx.device_info()['max_recommended_working_set_size']*.78));mx.set_cache_limit(2**30)
data={'device':mx.device_info(),'target':str(TARGET),'draft':str(DRAFT),'args':vars(args),'source_sha256':{},'samples':[],'diagnostics':[],'projection_checks':[],'swap_before':subprocess.check_output(['sysctl','vm.swapusage'],text=True)}
for file in [Path(__file__),Path(g.__file__),ROOT/'python/sglang/srt/hardware_backend/mlx/models/glm5_fp4.py']:
 data['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
def save(): (OUT/'fullmodel.json').write_text(json.dumps(data,indent=2))
t=time.perf_counter();model,_=load_fp4_model(TARGET,enable_mtp=False);model.configure_optimizations('affine_fused');model.configure_mtp_verification('ffn');model.configure_latent_projection_cache(True);model.configure_eval_groups(decode=2,prefill=4,verification=2);model.prepare_for_inference();mx.eval(model.parameters())
engine=MlxDFlashServing(model,DRAFT,3);tokenizer=load_tokenizer(TARGET)
print('MODEL_READY',time.perf_counter()-t,mx.get_active_memory()/2**30,flush=True)
original_project=g._project_rows;captured={};seen=Counter();capturing=False;mode='reference'
def project(projection,x):
 if isinstance(projection,FP8Linear) and x.shape[1]>1:
  seen[(projection.block_size,projection.activation_mode,tuple(projection.weight.shape),x.shape[1])]+=1
  if capturing and id(projection) not in captured: captured[id(projection)]=(projection,x)
  if mode=='batch_fp8': return projection(x)
 return original_project(projection,x)
g._project_rows=project
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。'][:args.prompts]
prompts=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=True,add_generation_prompt=True) for s in texts]
canonical={}
def select(new_mode,width):
 global mode
 mode=new_mode;engine.num_drafts=width;engine.clear();model.model._fixed_groups.clear()
 for layer in model.layers:
  for n in ('_fixed_mla_layer','_fixed_mla_attention'):
   owner=getattr(layer,n,None)
   if owner is not None:
    owner.graphs.clear()
    if hasattr(owner,'attention') and hasattr(owner.attention,'graphs'):owner.attention.graphs.clear()
 model.warmup_for_inference(verification_counts=(width+1,),capture_layer_ids=engine.capture_layer_ids,token_only=True)

def prefill(prompt):
 cache=model.make_cache()
 for start in range(0,len(prompt),256):
  hidden=engine.prefill('p',mx.array([prompt[start:start+256]],mx.int32),cache);mx.eval(hidden,[c.state for c in cache])
 return cache,int(model.target_tokens(hidden[:,-1:]).item())

# True verify graphs: independent same-prefix replays, phases synchronized only here.
model.configure_fixed_groups(False)
select('reference',7);cache,token=prefill(prompts[0]);input_tokens=mx.array([[token]*8],mx.int32);mx.eval(input_tokens)
capturing=True
result=model.verify_mtp_requests([input_tokens],[[clone_cache(c) for c in cache]],token_only=True,capture_layer_ids=engine.capture_layer_ids)
mx.eval(result);capturing=False
for projection,x in captured.values():
 ref=original_project(projection,x);bat=projection(x);mx.eval(ref,bat)
 diff=mx.abs(ref.astype(mx.float32)-bat.astype(mx.float32))
 row=dict(block=projection.block_size,activation_mode=projection.activation_mode,shape=list(projection.weight.shape),input_shape=list(x.shape),exact=bool(mx.array_equal(ref,bat).item()),max_abs=float(diff.max().item()))
 data['projection_checks'].append(row)
print('PROJECTION_CHECKS',len(captured),'exact',sum(r['exact'] for r in data['projection_checks']),'max',max((r['max_abs'] for r in data['projection_checks']),default=0),flush=True);save()
# Snapshot outputs retain graph references. Explicitly evaluate captured inputs now.
mx.eval([x for _,x in captured.values()]);del captured,result,cache
model.configure_fixed_groups(True)

modes=args.modes.split(',')
for round_id in range(args.rounds):
 configurations=[(m,w) for m in modes for w in (3,5,7)]
 if round_id%2:configurations.reverse()
 for current,width in configurations:
  select(current,width)
  for prompt_id,prompt in enumerate(prompts):
   engine.clear();before=copy.deepcopy(engine.stats());t=time.perf_counter();cache,token=prefill(prompt);prefill_seconds=time.perf_counter()-t
   ids=[token];raw=[];cycles=[];mx.reset_peak_memory()
   for step in range(args.tokens-1):
    engine.set_remaining('p',args.tokens-step-1);ct=engine.counters['verify_ct'];t=time.perf_counter()
    result=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(result,[c.state for c in cache]);token=int(result.item());dt=time.perf_counter()-t
    raw.append(dt);ids.append(token)
    if engine.counters['verify_ct']!=ct:cycles.append({'step':step,'seconds':dt,'offset':cache[0].offset})
   after=copy.deepcopy(engine.stats());exact=prompt_id not in canonical or ids==canonical[prompt_id];canonical.setdefault(prompt_id,ids)
   stats={k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_correct_drafts','num_accept_tokens','reject_ct','num_reuse_tokens')}
   row=dict(mode=current,num_drafts=width,round=round_id,prompt_id=prompt_id,input_tokens=len(prompt),prefill_seconds=prefill_seconds,tokens=ids,raw_seconds=raw,cycles=cycles,tok_s=len(raw)/sum(raw),steady_tok_s=len(raw[16:])/sum(raw[16:]),exact=exact,stats=stats,accept_length=stats['num_accept_tokens']/stats['verify_ct'],accept_rate=stats['num_correct_drafts']/max(1,stats['num_proposed_drafts']),correct_by_depth={k:v-before.get('correct_by_depth',{}).get(k,0) for k,v in after.get('correct_by_depth',{}).items()},eligible_by_depth={k:v-before.get('eligible_by_depth',{}).get(k,0) for k,v in after.get('eligible_by_depth',{}).items()},host_timings={k:v-before.get('timings',{}).get(k,0) for k,v in after.get('timings',{}).items()},peak_bytes=mx.get_peak_memory(),active_bytes=mx.get_active_memory())
   data['samples'].append(row);save();print('FULL300',round_id,current,width,prompt_id,round(row['tok_s'],3),'accept',round(row['accept_length'],3),'exact',exact,flush=True)
   engine.remove('p');del cache,result
   if not exact: raise RuntimeError('Token mismatch; inspect raw result before continuing')

# Stage times explain scaling but deliberately destroy overlap. No throughput claims.
for current in modes:
 for width in (3,5,7):
  select(current,width);engine.clear();cache,token=prefill(prompts[0]);chain=mx.array([[token]*(width+1)],mx.int32);mx.eval(chain)
  phase_rows=[];functions=[]
  def instrument(cls,label):
   original=cls.__call__
   def wrapped(self,*a,**kw):
    mx.synchronize();t=time.perf_counter();r=original(self,*a,**kw);mx.eval(r);phase_rows.append({'phase':label,'layers':len(self.layers) if hasattr(self,'layers') else 1,'seconds':time.perf_counter()-t});return r
   cls.__call__=wrapped;functions.append((cls,original))
  instrument(g._FixedLayerGroup,'kda_group');instrument(g._FixedMLALayer,'mla_layer')
  try:
   for replay in range(4):
    phase_rows=[];private=[clone_cache(c) for c in cache];t=time.perf_counter();r=model.verify_mtp_requests([chain],[private],token_only=True,capture_layer_ids=engine.capture_layer_ids);mx.eval(r)
    data['diagnostics'].append({'mode':current,'num_drafts':width,'replay':replay,'seconds':time.perf_counter()-t,'phases':phase_rows});print('SYNC_VERIFY',current,width,replay,round(data['diagnostics'][-1]['seconds']*1000,3),flush=True);save()
  finally:
   for cls,original in functions:cls.__call__=original
  del cache,private,r

data['projection_calls']=[{'key':list(k),'count':v} for k,v in seen.items()];data['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);data['complete']=True;save()
