exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
import statistics
from native_nvfp4 import project
sw=next(layer.mlp.switch_mlp for layer in m.layers if hasattr(layer.mlp,'switch_mlp'))
r['micro']=[]
for name,p in [('gate',sw.gate_proj),('down',sw.down_proj)]:
 for dtype in (mx.float32,mx.bfloat16):
  ids=mx.array([[1,5,17,48,103,131,176,247]],mx.uint32);k=p.weight.shape[-1]*2;x=mx.random.normal((1,1,1,k) if name=='gate' else (1,8,1,k)).astype(dtype);ref=p(x,ids);mx.eval(ref)
  for rps in (1,2,4,8):
   fn=mx.compile(lambda xx,ii:project(xx,p.weight,p.scales,p.weight_scale_2,ii,rps=rps));v=fn(x,ids);mx.eval(v);times=[]
   for i in range(15):
    t=time.perf_counter();v=fn(x,ids);mx.eval(v);times.append(time.perf_counter()-t)
   row={'name':name,'dtype':str(dtype),'rps':rps,'times':times,'exact':mx.array_equal(v,ref).item(),'max_error':mx.max(mx.abs(v.astype(mx.float32)-ref.astype(mx.float32))).item()};r['micro'].append(row);print('NV_MICRO',name,str(dtype),rps,statistics.median(times[3:])*1000,row['exact'],row['max_error'],flush=True)
(root/'nv_final.json').write_text(json.dumps(r,indent=2))

from mxfp8_single import mxfp8_single
original=f.NVFP4SwitchLinear.__call__
f.mxfp8_project_rows=lambda x,w,s:mxfp8_single(x,w,s,tilegroups=1)
exec(open('benchmark/mlx/ar_optimization_20261103/single_ab.py').read().split('# Clear graph owners recursively for valid dispatch A/B.\n')[1].split("for mode in (")[0])
def candidate(self,x,indices,sorted_indices=False):
 if x.size//x.shape[-1] <= indices.size:
  return project(x,self.weight,self.scales,self.weight_scale_2,indices,rps=1 if self.weight.shape[1]*2>=4096 else 2)
 return original(self,x,indices,sorted_indices)
for mode in ('baseline','narrow','narrow','baseline'):
 f.NVFP4SwitchLinear.__call__=original if mode=='baseline' else candidate;clear();generate(texts[0],16)
 for text in texts:
  v=generate(text,64);v['mode']=mode;r['samples'].append(v);print('NV_FULL',mode,text[:8],v['tok_s'],flush=True)
 (root/'nv_final.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
