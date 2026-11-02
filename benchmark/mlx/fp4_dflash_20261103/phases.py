"""Diagnostic only: synchronize phases to attribute full-model critical work."""
import argparse,json,subprocess,time
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model,load_tokenizer
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=3);args=parser.parse_args()
root=Path('/Users/jay/workspace/sglang/benchmark/mlx/fp4_dflash_20261103');target=Path('/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5');draft=Path('/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a')
mx.set_memory_limit(200*2**30);mx.set_wired_limit(200*2**30);mx.set_cache_limit(1024**3)
m,_=load_fp4_model(target,enable_mtp=False);m.configure_optimizations('affine_fused');m.configure_mtp_verification('ffn');m.configure_latent_projection_cache(True);m.prepare_for_inference();m.configure_eval_groups(decode=2,prefill=4,verification=2);mx.eval(m.parameters());e=MlxDFlashServing(m,draft,args.count)
def arrays(v):
 if isinstance(v,mx.array):return [v]
 if isinstance(v,dict):return [a for x in v.values() for a in arrays(x)]
 if isinstance(v,(list,tuple)):return [a for x in v for a in arrays(x)]
 return []
def measure(fn):
 mx.synchronize();start=time.perf_counter();v=fn();mx.eval(arrays(v));mx.synchronize();return v,time.perf_counter()-start
r={'diagnostic':'Synchronized phases; destroys overlap, not throughput A/B; head/sort/selector are isolated replays inside draft, not additive phases','cases':[],'device':mx.device_info()}
tok=load_tokenizer(target)
for context in (32,):
 p=tok.apply_chat_template([{'role':'user','content':'Explain why the sky is blue.'}],tokenize=True,add_generation_prompt=True)
 p=tok.encode('Background notes: '+('Machine learning uses data. '*(context//5)))[:context-len(p)]+p
 cache=m.make_cache()
 for i in range(0,len(p),256):h=e.prefill('p',mx.array([p[i:i+256]],mx.int32),cache);mx.eval(h)
 mx.eval(arrays(e.states['p'].draft_state));mx.synchronize()
 token=int(m.target_tokens(h[:,-1:]).item())
 for j in range(12):
  state=e.states['p'];_,consume_s=measure(lambda:(e._consume_features(state),state.draft_state))
  draft_h=[];unary=[]
  def head(x):draft_h.append(x);v=m.lm_head(x).astype(mx.float32);unary.append(v);return v
  proposals,draft_s=measure(lambda:e.drafter.draft(mx.array([[token]],mx.int32),[state.draft_state],m.model.embed_tokens,head,args.count))
  _,head_s=measure(lambda:m.lm_head(draft_h[0]));_,sort_s=measure(lambda:mx.argpartition(-unary[0],kth=15,axis=-1)[...,:16]);_,select_s=measure(lambda:e.drafter.candidate_selector.select(draft_h[0],unary[0],mx.array([[token]],mx.int32)))
  chain=mx.concatenate([mx.array([[token]],mx.int32),proposals],1);private=[clone_cache(c) for c in cache]
  result,verify_s=measure(lambda:m.verify_mtp_rows(chain,private,token_only=True,capture_layer_ids=e.capture_layer_ids))
  values,hidden,snaps,features=result
  n=1
  while n<chain.shape[1] and int(chain[0,n].item())==int(values[0,n-1].item()):n+=1
  for dst,src in zip(cache,snaps[n-1]):dst.cache=[v.view(v.dtype) if v is not None else None for v in src.cache];dst.offset=src.offset
  offset=state.offset;state.pending_features.append(features[:,:n]);state.pending_positions.append(mx.arange(offset,offset+n,dtype=mx.int32));state.offset+=n;token=int(values[0,n-1].item())
  e._publish(state,cache)
  r['cases'].append(dict(context=context,cycle=j,num_accept_tokens=n,consume_seconds=consume_s,draft_seconds=draft_s,head_seconds=head_s,sort_seconds=sort_s,selector_seconds=select_s,verify_seconds=verify_s,cache_length=state.draft_state['positions'].size));(root/f'phases_topk{args.count}.json').write_text(json.dumps(r,indent=2));print(context,j,n,consume_s,draft_s,sort_s,verify_s,flush=True)
  del result,values,hidden,snaps,features,private,src,dst,draft_h,unary,proposals,chain
 e.remove('p');del cache,h;mx.clear_cache()
