"""Alternating full-model arms with one identical resident target checkpoint."""
import argparse,copy,hashlib,json,os,subprocess,time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,default=128);p.add_argument('--repeats',type=int,default=1);p.add_argument('--output',default='benchmark/mlx/fp4_dflash_20261103/narrow_ab.json');a=p.parse_args()
target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
out=Path(a.output);r=dict(device=mx.device_info(),checkpoint=str(target),draft=str(draft),tokens=a.tokens,note='Current NVIDIA FP4 target, native MTP disabled; identical target and BF16 draft shared by all arms.',samples=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True),env={k:v for k,v in os.environ.items() if k.startswith('MLX_')})
def save():out.write_text(json.dumps(r,indent=2)+'\n')
mx.set_memory_limit(200*2**30);mx.set_wired_limit(200*2**30);mx.set_cache_limit(1024**3)
start=time.perf_counter();m,_=load_fp4_model(target,enable_mtp=False);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();m.configure_eval_groups(decode=2,prefill=4,verification=2);mx.eval(m.parameters());r['load_seconds']=time.perf_counter()-start;r['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
class AR:
 def clear(self):pass
 def remove(self,rid):pass
 def stats(self):return {}
 def set_remaining(self,rid,n):pass
 def prefill(self,rid,inputs,cache):return m.model(inputs,cache)
 def decode_batch(self,rids,inputs,caches,tokens,token_only=True):return m.forward_with_hidden(inputs,caches[0],token_only=token_only)[0]
print('TARGET_LOADED',mx.get_active_memory()/2**30,flush=True);engines={};d=MlxDFlashServing(m,draft,7)
for n in (3,):engines['dflash'+str(n)]=MlxDFlashServing(m,draft,n,drafter=d.drafter)
r['resident_bytes']=mx.get_active_memory();tok=load_tokenizer(target)
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
print('DRAFT_LOADED',mx.get_active_memory()/2**30,flush=True)

from sglang.srt.hardware_backend.mlx.models import glm5_next as g, glm5_fp4 as f
original=g.MLP.__call__
import sys
sys.path.insert(0,'benchmark/mlx/ar_optimization_20261103')
import native_nvfp4
original_nv=f.NVFP4SwitchLinear.__call__
def candidate_nv(self,x,indices,sorted_indices=False):
 if indices.shape[1]>1 and x.dtype in (mx.bfloat16,mx.float32) and self.weight.shape[-1]*2%512==0:
  return native_nvfp4.project(x,self.weight,self.scales,self.weight_scale_2,indices,rps=1 if self.weight.shape[-2]>=4096 else 2)
 return original_nv(self,x,indices,sorted_indices=sorted_indices)
for mode in ('baseline','candidate','candidate','baseline'):
 f.NVFP4SwitchLinear.__call__=candidate_nv if mode=='candidate' else original_nv
 m.model._fixed_groups.clear()
 for layer in m.layers:
  for name in ('_fixed_mla_layer','_fixed_mla_attention'):
   owner=getattr(layer,name,None)
   if owner is not None:owner.graphs.clear()
 for context in (32,):
  for repeat in range(a.repeats):
   for text_id,text in enumerate(texts):
    prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
    if context>len(prompt):
     pad=tok.encode('Background notes: '+('Machine learning uses data and algorithms. '*context))
     prompt=pad[:context-len(prompt)]+prompt
    order=list(engines);shift=(repeat*3+text_id)%len(order);order=order[shift:]+order[:shift]
    if repeat%2:order.reverse()
    for arm in order:
     e=engines[arm];e.clear();cache=m.make_cache();before=copy.deepcopy(e.stats());start=time.perf_counter()
     for i in range(0,len(prompt),256):h=e.prefill('p',mx.array([prompt[i:i+256]],mx.int32),cache);mx.eval(h,[c.cache for c in cache])
     token=int(m.target_tokens(h[:,-1:]).item());prefill=time.perf_counter()-start;ids=[token];raw=[];mx.reset_peak_memory()
     for i in range(a.tokens-1):
      e.set_remaining('p',a.tokens-i-1);start=time.perf_counter();v=e.decode_batch(['p'],mx.array([[token]],mx.int32),[cache],[token],token_only=True);mx.eval(v,[c.cache for c in cache]);token=int(v.item());raw.append(time.perf_counter()-start);ids.append(token)
     after=copy.deepcopy(e.stats());e.remove('p')
     delta={k:after.get(k,0)-before.get(k,0) for k in ('verify_ct','num_proposed_drafts','num_accept_tokens','num_correct_drafts','reject_ct','num_reuse_tokens')}
     if arm=='mtp2':delta['num_correct_drafts']=sum(after.get('correct_by_depth',{}).values())-sum(before.get('correct_by_depth',{}).values());delta['num_accept_tokens']=a.tokens-1
     row=dict(context=len(prompt),mode=mode,repeat=repeat,text_id=text_id,arm=arm,order=order,prefill_seconds=prefill,raw_seconds=raw,tok_s=len(raw)/sum(raw),steady_tok_s=len(raw[16:])/sum(raw[16:]),tokens=ids,hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),stats=delta,peak_bytes=mx.get_peak_memory(),active_bytes=mx.get_active_memory());r['samples'].append(row);save();print(mode,context,repeat,text_id,arm,round(row['tok_s'],3),delta,flush=True)
     del cache,h,v;mx.clear_cache()
 r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()