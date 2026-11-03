exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
from mxfp8_nax import fn
import statistics
r['micro']=[];seen=set()
for name,p in [('head',m.lm_head)]+list(m.named_modules()):
 if not isinstance(p,f.FP8Linear) or p.block_size!=32 or tuple(p.weight.shape) in seen:continue
 seen.add(tuple(p.weight.shape))
 for count in (1,4,12):
  x=mx.random.normal((count,p.weight.shape[1])).astype(mx.bfloat16);ref=f.mxfp8_project_rows(x,p.weight,p.scales);mx.eval(ref)
  for split in (1,2,4,8,16):
   call=mx.compile(lambda xx:fn(xx,p.weight,p.scales,split=split));times=[];v=call(x);mx.eval(v)
   for i in range(12):
    t=time.perf_counter();v=call(x);mx.eval(v);times.append(time.perf_counter()-t)
   row={'name':name,'shape':p.weight.shape,'count':count,'split':split,'times':times,'exact':mx.array_equal(v,ref).item(),'max_error':mx.max(mx.abs(v.astype(mx.float32)-ref.astype(mx.float32))).item()};r['micro'].append(row);print('MX_NAX',name,count,split,statistics.median(times[3:])*1000,row['exact'],row['max_error'],flush=True)
 (root/'mxfp8_nax_test.json').write_text(json.dumps(r,indent=2))
