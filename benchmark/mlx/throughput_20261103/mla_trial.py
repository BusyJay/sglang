exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
from sglang.srt.hardware_backend.mlx.models import glm5_next as glm
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache
original=glm.DecoderLayer.__call__
def candidate(self,x,cache,*,shared_stream=None):
 fixed=getattr(self,'_fixed_mla_layer',None)
 if not self.is_linear and fixed is not None and x.shape[0]==1 and x.shape[1]==1 and shared_stream is None:
  return fixed(x,cache)[0]
 return original(self,x,cache,shared_stream=shared_stream)
# Compare layer outputs/cache states on the same genuine AR input once.
cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v,[c.cache for c in cache]);token=int(v.item())
x=m.model.embed_tokens(mx.array([[token]],mx.int32));x=mx.repeat(x[...,None,:],m.args.hc_mult,-2)
checks=[]
for i,layer in enumerate(m.layers):
 a=clone_cache(cache[i]);b=clone_cache(cache[i]);old=original(layer,x,a);new=candidate(layer,x,b);mx.eval(old,new,[a.cache,b.cache]);diff=mx.abs(old.astype(mx.float32)-new.astype(mx.float32))
 rec={'layer':i,'linear':layer.is_linear,'exact':mx.array_equal(old,new).item(),'max':mx.max(diff).item(),'offset_equal':a.offset==b.offset};checks.append(rec)
 for j,(aa,bb) in enumerate(zip(a.cache,b.cache)):
  if aa is None or bb is None:assert aa is bb,(i,j)
  else:assert mx.array_equal(aa,bb).item(),('cache',i,j,aa.shape,bb.shape)
 x=old
print('MLA_NUMERIC',checks,flush=True)
(root/'mla_numeric.json').write_text(json.dumps(checks,indent=2));del cache,h,v,x,a,b,old,new
samples=[]
for mode in ('baseline','compiled_mla','compiled_mla','baseline'):
 glm.DecoderLayer.__call__=candidate if mode=='compiled_mla' else original;m.configure_fixed_groups(True)
 for j in range(3):
  row=generate(j,'ar');row['mode']=mode;samples.append(row)
  (root/'mla_ab.json').write_text(json.dumps(samples,indent=2));print('MLA300',mode,j,row['tok_s'],row['prefill_seconds'],flush=True)
glm.DecoderLayer.__call__=original
for j in range(3):
 ss=[v for v in samples if v['text_id']==j];print('MLA_TOKENS',j,all(v['tokens']==ss[0]['tokens'] for v in ss),flush=True)
print('MLA_AB_COMPLETED',flush=True)
