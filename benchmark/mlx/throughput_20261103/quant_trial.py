exec(open('benchmark/mlx/throughput_20261103/short_bottleneck.py').read().split('for j in range(3):')[0])
from fp8_quant_fused import project
import statistics
original=fp4.FP8Linear.project
captured={}
def capture(self,x):
 if self.block_size==128:captured.setdefault((tuple(self.weight.shape),str(x.dtype)),(self,x))
 return original(self,x)
fp4.FP8Linear.project=capture
m.configure_fixed_groups(False)
numerics=[]
for count in (1,4):
 captured.clear();cache=m.make_cache();p=tok.apply_chat_template([{'role':'user','content':texts[0]}],tokenize=True,add_generation_prompt=True)
 h=m.model(mx.array([p],mx.int32),cache);v=m.target_tokens(h[:,-1:]);mx.eval(v);token=int(v.item());captured.clear()
 v=m.forward_last_tokens(mx.array([[token]],mx.int32),cache) if count==1 else m.verify_mtp_rows(mx.array([[token]*count],mx.int32),cache,token_only=True,capture_layer_ids=e.capture_layer_ids);mx.eval(v)
 for key,(p,x) in captured.items():
  x=mx.contiguous(x);mx.eval(x);a=original(p,x);b=project(x,p.weight,p.scales);mx.eval(a,b)
  row={'count':count,'key':key,'exact':mx.array_equal(a,b).item(),'max':mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item()}
  numerics.append(row);print('QUANT_NUMERIC',row,flush=True)
  if not row['exact']:raise AssertionError(row)
 captured.clear();del cache,h,v;mx.clear_cache()
def candidate(self,x):
 if self.block_size==128 and x.size//x.shape[-1]<=4:return project(x,self.weight,self.scales)
 return original(self,x)
samples=[]
for mode in ('baseline','fused','fused','baseline'):
 fp4.FP8Linear.project=candidate if mode=='fused' else original;m.configure_fixed_groups(True)
 for j in range(3):
  for arm in ('ar','dflash3') if j%2==0 else ('dflash3','ar'):
   row=generate(j,arm);row['mode']=mode;samples.append(row)
   (root/'quant_ab.json').write_text(json.dumps({'samples':samples,'numerics':numerics},indent=2))
   print('QUANT300',mode,j,arm,row['tok_s'],row['stats'],flush=True)
fp4.FP8Linear.project=original
for j in range(3):
 ss=[v for v in samples if v['text_id']==j];assert all(v['tokens']==ss[0]['tokens'] for v in ss),j
print('QUANT_AB_COMPLETED_EXACT',flush=True)
