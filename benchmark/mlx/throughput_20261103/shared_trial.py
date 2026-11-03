exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
from nv_shared import project
import statistics
original=fp4.NVFP4SwitchLinear.__call__
captured={}
def capture(self,x,ids,sorted_indices=False):
 captured.setdefault((tuple(self.weight.shape),str(x.dtype)),(self,x,ids))
 return original(self,x,ids,sorted_indices=sorted_indices)
fp4.NVFP4SwitchLinear.__call__=capture;m.configure_fixed_groups(False)
records=[]
for count in (1,4):
 captured.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
 h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v);token=int(v.item());captured.clear()
 v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache) if count==1 else m.verify_mtp_rows(mx.array([[token]*count],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids);mx.eval(v)
 for key,(p,x,ids) in captured.items():
  x=mx.contiguous(x);ids=mx.contiguous(ids);mx.eval(x,ids);a=original(p,x,ids);mx.eval(a)
  variants={'native':lambda:original(p,x,ids)}
  for rps in (1,2,4):variants[str(rps)]=lambda rps=rps:project(x,p.weight,p.scales,p.weight_scale_2,ids,rps)
  errors={};trials=[]
  for name,fn in variants.items():
   b=fn();mx.eval(b);errors[name]={'exact':mx.array_equal(a,b).item(),'max':mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item()}
  for j in range(7):
   ts={}
   for name in list(variants) if j%2==0 else list(reversed(variants)):
    fn=variants[name];mx.eval(fn());t=time.perf_counter()
    for _ in range(15):mx.eval(fn())
    ts[name]=(time.perf_counter()-t)/15
   trials.append(ts)
  row={'count':count,'key':key,'ids':ids.tolist(),'errors':errors,'trials':trials};records.append(row)
  print('SHARED_MICRO',count,key,errors,{k:statistics.median(t[k] for t in trials)*1000 for k in variants},flush=True)
 captured.clear();del cache,h,v;mx.clear_cache()
(root/'shared_trial.json').write_text(json.dumps(records,indent=2))
best={tuple(v['key'][0]):min((k for k in v['errors'] if k!='native'),key=lambda k:statistics.median(t[k] for t in v['trials'])) for v in records if v['count']==4}
def candidate(self,x,ids,sorted_indices=False):
 if ids.ndim==3 and 1<ids.shape[1]<=4:return project(x,self.weight,self.scales,self.weight_scale_2,ids,int(best[tuple(self.weight.shape)]))
 return original(self,x,ids,sorted_indices=sorted_indices)
samples=[]
for mode in ('baseline','shared','shared','baseline'):
 fp4.NVFP4SwitchLinear.__call__=candidate if mode=='shared' else original;m.configure_fixed_groups(True)
 for j in range(3):
  row=generate(j,'dflash3');row['mode']=mode;samples.append(row)
  (root/'shared_ab.json').write_text(json.dumps(samples,indent=2));print('SHARED300',mode,j,row['tok_s'],row['stats'],flush=True)
fp4.NVFP4SwitchLinear.__call__=original
for j in range(3):
 ss=[v for v in samples if v['text_id']==j];print('SHARED_TOKENS',j,all(v['tokens']==ss[0]['tokens'] for v in ss),flush=True)
print('SHARED_AB_COMPLETED',flush=True)
