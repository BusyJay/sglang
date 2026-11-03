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
def save(): (OUT/'multi_two_final.json').write_text(json.dumps(data,indent=2))
t=time.perf_counter();model,_=load_fp4_model(TARGET,enable_mtp=False);model.configure_optimizations('affine_fused');model.configure_mtp_verification('ffn');model.configure_latent_projection_cache(True);model.configure_eval_groups(decode=2,prefill=4,verification=2);model.prepare_for_inference();mx.eval(model.parameters())
mx.set_wired_limit(int(mx.device_info()['max_recommended_working_set_size']))
print('RUNNER_RESIDENCY',mx.device_info()['max_recommended_working_set_size'],flush=True)
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

from multi_singles_probe import MultiReuseSwitch
class SelectiveSwitch(MultiReuseSwitch):
 def __call__(self,x,indices):
  if x.shape[1]<8:return g.ClampedSwitchGLU.__call__(self,x,indices)
  return super().__call__(x,indices)
def arm_select(arm):
 for layer in model.layers:
  if isinstance(layer.mlp,g.MoE):
   sw=layer.mlp.switch_mlp;sw.__class__=g.ClampedSwitchGLU if arm=='reference' else SelectiveSwitch
   if arm!='reference':sw.probe_reuse=2;sw.probe_rps=4
 select('reference',7)
# Compare real full-model hidden/features and all immutable cache boundaries.
check=[];arm_select('reference');cache,token=prefill(prompts[0]);chain=mx.array([[token]*8],mx.int32)
for arm in ('reference','reuse4'):
 arm_select(arm);result=model.verify_mtp_requests([chain],[[clone_cache(c) for c in cache]],token_only=True,capture_layer_ids=engine.capture_layer_ids)[0]
 arrays=list(result[:2])+[result[3]]+[v for row in result[2] for c in row for v in c.cache if v is not None];mx.eval(arrays)
 check.append(arrays)
assert len(check[0])==len(check[1])
exact=all(mx.array_equal(a,b).item() for a,b in zip(*check));diffs=[float(mx.abs(a.astype(mx.float32)-b.astype(mx.float32)).max().item()) for a,b in zip(*check) if a.dtype!=mx.int32]
data['checks']={'arrays':len(check[0]),'exact':exact,'max_abs':max(diffs)};save();print('MULTI_FULL_NUMERIC',data['checks'],flush=True)
if not exact:raise RuntimeError('Inspect numerical differences before adopting')
del check,arrays,result,cache
canonical=None
for arm in ('reference','reuse4','reuse4','reference'):
 arm_select(arm);cache,token=prefill(prompts[0]);ids=[token];raw=[];before=copy.deepcopy(engine.stats());t0=time.perf_counter()
 for step in range(127):
  engine.set_remaining('p',127-step);ct=engine.counters['verify_ct'];t=time.perf_counter();result=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);token=int(result.item());dt=time.perf_counter()-t;ids.append(token);raw.append({'step':step,'seconds':dt,'verify':engine.counters['verify_ct']!=ct})
  if dt>.5 or time.perf_counter()-t0>8:break
 after=engine.stats();stats={k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_correct_drafts','num_accept_tokens','reject_ct')}
 if canonical is None:canonical=(ids,stats)
 row={'arm':arm,'tokens':ids,'raw':raw,'stats':stats,'exact_tokens':ids==canonical[0],'exact_stats':stats==canonical[1],'tok_s':len(raw)/sum(r['seconds'] for r in raw),'after16_tok_s':len(raw[16:])/sum(r['seconds'] for r in raw[16:]),'peak_bytes':mx.get_peak_memory()};data['samples'].append(row);save();print('MULTI_SERVING128',arm,row['tok_s'],row['after16_tok_s'],row['exact_tokens'],row['exact_stats'],flush=True)
 if len(raw)!=127 or not row['exact_tokens'] or not row['exact_stats']:raise RuntimeError('short serving comparison failed')
 engine.clear();del cache,result
print('MULTI_SERVING_DONE',flush=True)
