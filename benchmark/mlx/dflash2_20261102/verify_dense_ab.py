"""Alternating full-model arms with one identical resident target checkpoint."""
import argparse,copy,importlib.util,sys,hashlib,json,os,subprocess,time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models.glm5_next import Model,ModelArgs
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--tokens',type=int,default=256);p.add_argument('--repeats',type=int,default=2);p.add_argument('--output',default='benchmark/mlx/dflash2_20261102/compare_final.json');a=p.parse_args()
if a.baseline:
 spec=importlib.util.spec_from_file_location('dense_baseline','benchmark/mlx/dflash2_20261102/baseline_dense/mlx_dense_rows.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
 import sglang.srt.hardware_backend.mlx.models.glm5_next as glm
 glm.mlx_dense_rows=base.mlx_dense_rows

target=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
out=Path(a.output);r=dict(device=mx.device_info(),checkpoint=str(target),draft=str(draft),tokens=a.tokens,note='All arms share identical resident target without native MTP weights. Baseline full-chain expert grouping with four-row KDA dense tile vs full verification-row KDA tile; separate clean processes.',samples=[],swap_before=subprocess.check_output(['sysctl','vm.swapusage'],text=True),env={k:v for k,v in os.environ.items() if k.startswith('MLX_')})
def save():out.write_text(json.dumps(r,indent=2)+'\n')
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size']);mx.set_cache_limit(1024**3)
start=time.perf_counter();m,_=load_model(target,lazy=True,get_model_classes=lambda config:(lambda args:Model(args,enable_mtp=False),ModelArgs));m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());r['load_seconds']=time.perf_counter()-start
class AR:
 def clear(self):pass
 def remove(self,rid):pass
 def stats(self):return {}
 def set_remaining(self,rid,n):pass
 def prefill(self,rid,inputs,cache):return m.model(inputs,cache)
 def decode_batch(self,rids,inputs,caches,tokens,token_only=True):return m.forward_with_hidden(inputs,caches[0],token_only=token_only)[0]
assert m.mtp is None
r['native_mtp_loaded']=False;r['baseline']=a.baseline
engines={'ar':AR()};d=MlxDFlashServing(m,draft,7)
for n in (3,5,7):engines['dflash'+str(n)]=MlxDFlashServing(m,draft,n,drafter=d.drafter)
r['resident_bytes']=mx.get_active_memory();tok=load_tokenizer(target)
texts=['Explain why the sky is blue.','Write a Python function that computes the first n Fibonacci numbers, with an explanation.','请介绍机器学习中的过拟合，并给出避免过拟合的方法。']
for context in (32,2050):
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
    row=dict(context=len(prompt),repeat=repeat,text_id=text_id,arm=arm,order=order,prefill_seconds=prefill,raw_seconds=raw,tok_s=len(raw)/sum(raw),steady_tok_s=len(raw[16:])/sum(raw[16:]),tokens=ids,hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),stats=delta,peak_bytes=mx.get_peak_memory(),active_bytes=mx.get_active_memory());r['samples'].append(row);save();print(context,repeat,text_id,arm,round(row['tok_s'],3),delta,flush=True)
    del cache,h,v;mx.clear_cache()
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);save()
