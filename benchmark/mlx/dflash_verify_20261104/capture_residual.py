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
def save(): (OUT/'runner_residency.json').write_text(json.dumps(data,indent=2))
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

model.configure_fixed_groups(False);engine.num_drafts=7
captures={};original=g.DecoderLayer.verify_ffn_rows
layer26=model.layers[26]
def capture(self,value,cache,**kw):
 if self is layer26:captures['residual']=value
 return original(self,value,cache,**kw)
g.DecoderLayer.verify_ffn_rows=capture
cache,token=prefill(prompts[0]);engine.set_remaining('p',40);out=engine.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(out,captures)
assert 'residual' in captures;mx.savez(OUT/'real_verify_residual.npz',**captures);print('REAL_CAPTURE',[(k,v.shape,str(v.dtype)) for k,v in captures.items()],flush=True)
