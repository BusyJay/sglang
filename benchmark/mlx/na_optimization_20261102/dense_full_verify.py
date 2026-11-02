"""Target verification attribution; synchronized operator replays are diagnostic."""
import collections,contextlib,hashlib,json,time
from pathlib import Path
from unittest.mock import patch
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
root=Path('benchmark/mlx/na_optimization_20261102');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size']);mx.set_cache_limit(1024**3)
m,_=load_model(target,lazy=True,get_model_classes=lambda config:(g.Model,g.ModelArgs));m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();mx.eval(m.parameters());e=MlxDFlashServing(m,draft,7);tok=load_tokenizer(target)
def arrays(v):
 if isinstance(v,nn.Module):return []
 if isinstance(v,mx.array):return [v]
 if isinstance(v,(list,tuple)):return [a for x in v for a in arrays(x)]
 if isinstance(v,dict):return [a for x in v.values() for a in arrays(x)]
 if hasattr(v,'cache'):return arrays(v.cache)
 return []
def timed(fn):
 mx.synchronize();start=time.perf_counter();v=fn();mx.eval(arrays(v));mx.synchronize();return v,time.perf_counter()-start
assert m.mtp is None
r=dict(device=mx.device_info(),native_mtp_loaded=False,resident_bytes=mx.get_active_memory(),note='Same resident target, paired baseline/candidate full compiled verification; independent graph dictionaries, all outputs and per-row snapshots checked exactly; first two trials excluded from summary.',cases=[])

original=g.mlx_dense_rows
def candidate(x,w):
 if x.dtype==mx.bfloat16 and w.dtype==mx.bfloat16 and w.shape==(4096,8192) and x.shape[1]>=3:
  return x@w.T
 return original(x,w)
# Distinct graph owners isolate traced baseline/candidate operators. These
# diagnostics run alone; serving never mutates shared modules or graph stores.
stores={mode:{'groups':{},'mla':{i:{} for i,l in enumerate(m.layers) if not l.is_linear}} for mode in ('baseline','candidate')}
def activate(mode):
 m.model._fixed_groups=stores[mode]['groups']
 for i,graphs in stores[mode]['mla'].items():m.layers[i]._fixed_mla_layer.graphs=graphs
 g.mlx_dense_rows=original if mode=='baseline' else candidate
for ctx in (32,2050):
 p=tok.apply_chat_template([{'role':'user','content':'Write a Python function that computes the first n Fibonacci numbers, with an explanation.'}],tokenize=True,add_generation_prompt=True)
 if ctx>len(p):p=tok.encode('Background: '+('Machine learning uses data. '*ctx))[:ctx-len(p)]+p
 cache=m.make_cache();activate('candidate')
 for i in range(0,len(p),256):h=e.prefill('p',mx.array([p[i:i+256]],mx.int32),cache);mx.eval(h)
 mx.eval(e.states['p'].draft_state);anchor=int(m.target_tokens(h[:,-1:]).item());proposals=e.drafter.draft(mx.array([[anchor]],mx.int32),[e.states['p'].draft_state],m.model.embed_tokens,m.lm_head,7);mx.eval(proposals)
 for count in (3,4,8):
  chain=mx.concatenate([mx.array([[anchor]],mx.int32),proposals[:,:count-1]],1);mx.eval(chain);samples=[]
  for trial in range(7):
   order=('baseline','candidate') if trial%2==0 else ('candidate','baseline');results={};times={}
   for mode in order:
    activate(mode);private=[clone_cache(c) for c in cache]
    result,elapsed=timed(lambda:m.verify_mtp_rows(chain,private,token_only=True,capture_layer_ids=e.capture_layer_ids));times[mode]=elapsed;results[mode]=result;del private
   left,right=results['baseline'],results['candidate']
   assert mx.array_equal(left[0],right[0]).item(),(ctx,count,trial,'tokens')
   def difference(a,b):
    delta=a.astype(mx.float32)-b.astype(mx.float32)
    denom=mx.sum(b.astype(mx.float32)**2)
    return dict(max_abs=mx.max(mx.abs(delta)).item(),rel_l2=mx.sqrt(mx.sum(delta*delta)/mx.maximum(denom,1e-30)).item())
   errors=[difference(left[1],right[1])]
   for aa,bb in zip(left[2],right[2]):
    for a,b in zip(aa,bb):
     assert a.offset==b.offset
     for x,y in zip(a.cache,b.cache):
      assert (x is None)==(y is None)
      if x is not None:errors.append(difference(x,y))
   max_error=max(e['rel_l2'] for e in errors)
   samples.append(dict(trial=trial,order=order,seconds=times,max_state_rel_l2=max_error));del left,right,result,results
  r['cases'].append(dict(context=len(p),count=count,samples=samples,target_tokens_equal=True));(root/'na_dense_full_verify.json').write_text(json.dumps(r,indent=2));print(len(p),count, samples[-1]['seconds'],flush=True)
 e.remove('p');del cache,h;mx.clear_cache()
g.mlx_dense_rows=original
