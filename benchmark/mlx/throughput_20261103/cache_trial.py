exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
import statistics
from mxfp8_cached import decode,project
cache_weights={}
start=time.perf_counter()
for name,p in m.named_modules():
 if isinstance(p,fp4.FP8Linear) and p.block_size==32 and p.weight.shape==(4096,8192):
  w=decode(p.weight);mx.eval(w);cache_weights[id(p)]=w
print('EXACT_CACHE',len(cache_weights),time.perf_counter()-start,mx.get_active_memory()/2**30,flush=True)
original=fp4.FP8Linear.project
captured=[]
def capture(self,x):
 if id(self) in cache_weights and not captured:captured.append((self,x))
 return original(self,x)
fp4.FP8Linear.project=capture
m.configure_fixed_groups(False)
cache=m.make_cache();prompt=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
h=m.model(mx.array([prompt],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v);token=int(v.item());captured.clear()
v=m.verify_mtp_rows(mx.array([[token]*4],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids);mx.eval(v)
p,x=captured[0];x=mx.contiguous(x);mx.eval(x)
a=original(p,x);b=project(x,cache_weights[id(p)],p.scales);mx.eval(a,b)
print('CACHE_ERROR',mx.array_equal(a,b).item(),mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item(),flush=True)
assert mx.array_equal(a,b).item()
trials=[]
for j in range(9):
 ts={}
 for mode in ('native','cache') if j%2==0 else ('cache','native'):
  fn=lambda:original(p,x) if mode=='native' else project(x,cache_weights[id(p)],p.scales)
  mx.eval(fn());t=time.perf_counter()
  for _ in range(16):mx.eval(fn())
  ts[mode]=(time.perf_counter()-t)/16
 trials.append(ts)
print('CACHE_MICRO',{k:statistics.median(t[k] for t in trials)*1000 for k in ('native','cache')},flush=True)
def candidate(self,x):
 if id(self) in cache_weights and x.size//x.shape[-1]<=4:return project(x,cache_weights[id(self)],self.scales)
 return original(self,x)
del cache,h,v,captured,x,a,b
samples=[]
for mode in ('baseline','cache','cache','baseline'):
 fp4.FP8Linear.project=candidate if mode=='cache' else original;m.configure_fixed_groups(True)
 for j in range(3):
  for arm in ('ar','dflash3') if j%2==0 else ('dflash3','ar'):
   row=generate(j,arm);row['mode']=mode;samples.append(row)
   (root/'cache_ab.json').write_text(json.dumps({'samples':samples,'trials':trials},indent=2))
   print('CACHE300',mode,j,arm,row['tok_s'],row['stats'],flush=True)
fp4.FP8Linear.project=original
for j in range(3):
 ss=[v for v in samples if v['text_id']==j];assert all(v['tokens']==ss[0]['tokens'] for v in ss),j
print('CACHE_AB_COMPLETED_EXACT',flush=True)
