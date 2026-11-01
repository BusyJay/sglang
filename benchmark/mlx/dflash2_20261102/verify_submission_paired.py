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
root=Path('benchmark/mlx/dflash2_20261102');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
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
r=dict(device=mx.device_info(),native_mtp_loaded=False,resident_bytes=mx.get_active_memory(),note='Same resident target, paired baseline/candidate full compiled verification; independent graph dictionaries, all outputs and per-row snapshots checked exactly; first two trials excluded; only verification submission group changed4/8/16/64; async counts and host durations recorded.',cases=[])

# Diagnostic A/B isolates submission boundaries, retaining all kernels.
modes={'group4':4,'group8':8,'group16':16,'group64':64}
stores={mode:{'groups':{},'mla':{i:{} for i,l in enumerate(m.layers) if not l.is_linear}} for mode in modes}
counts={mode:[] for mode in modes};active=[None];real_async=mx.async_eval
current_counts=[0,0.0]
def audited_async(*args,**kwargs):
 start=time.perf_counter();result=real_async(*args,**kwargs);current_counts[0]+=1;current_counts[1]+=time.perf_counter()-start;return result
mx.async_eval=audited_async
def activate(mode):
 active[0]=mode;current_counts[:]=[0,0.0]
 m.model._verification_eval_group_size=modes[mode]
 m.model._fixed_groups=stores[mode]['groups']
 for i,graphs in stores[mode]['mla'].items():m.layers[i]._fixed_mla_layer.graphs=graphs
for ctx in (32,2050):
 p=tok.apply_chat_template([{'role':'user','content':'Write a Python function that computes the first n Fibonacci numbers, with an explanation.'}],tokenize=True,add_generation_prompt=True)
 if ctx>len(p):p=tok.encode('Background: '+('Machine learning uses data. '*ctx))[:ctx-len(p)]+p
 cache=m.make_cache();activate('group4')
 for i in range(0,len(p),256):h=e.prefill('p',mx.array([p[i:i+256]],mx.int32),cache);mx.eval(h)
 mx.eval(e.states['p'].draft_state);anchor=int(m.target_tokens(h[:,-1:]).item());proposals=e.drafter.draft(mx.array([[anchor]],mx.int32),[e.states['p'].draft_state],m.model.embed_tokens,m.lm_head,7);mx.eval(proposals)
 for count in (4,6,8):
  chain=mx.concatenate([mx.array([[anchor]],mx.int32),proposals[:,:count-1]],1);mx.eval(chain);samples=[]
  for trial in range(9):
   order=list(modes);order=order if trial%2==0 else order[::-1];results={};times={};submits={}
   for mode in order:
    activate(mode);private=[clone_cache(c) for c in cache]
    result,elapsed=timed(lambda:m.verify_mtp_rows(chain,private,token_only=True,capture_layer_ids=e.capture_layer_ids));times[mode]=elapsed;submits[mode]=list(current_counts);results[mode]=result;del private
   for mode in list(modes)[1:]:
    left,right=results['group4'],results[mode]
    assert all(mx.array_equal(a,b).item() for a,b in zip(left[:2]+left[3:],right[:2]+right[3:])),(ctx,count,trial,'outputs')
    for aa,bb in zip(left[2],right[2]):
     for a,b in zip(aa,bb):
      assert a.offset==b.offset
      for x,y in zip(a.cache,b.cache):
       assert (x is None and y is None) or (x is not None and y is not None and mx.array_equal(x,y).item()),(ctx,count,trial,'snapshots')
   samples.append(dict(trial=trial,order=order,seconds=times,submissions=submits));del left,right,result,results
  r['cases'].append(dict(context=len(p),count=count,samples=samples,outputs_and_snapshots_exact=True));(root/'verify_submission_paired.json').write_text(json.dumps(r,indent=2));print(len(p),count, samples[-1]['seconds'],flush=True)
 e.remove('p');del cache,h;mx.clear_cache()
mx.async_eval=real_async
