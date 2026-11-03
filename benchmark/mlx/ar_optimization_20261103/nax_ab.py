exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
import statistics
from mxfp8_single import mxfp8_single
original=f.mxfp8_project_rows
def nax(x,w,s,tile=32):
 values=x.reshape(-1,x.shape[-1]);out=[]
 for start in range(0,values.shape[0],tile):
  v=values[start:start+tile];count=v.shape[0]
  if count<tile:v=mx.concatenate([v,mx.zeros((tile-count,v.shape[1]),v.dtype)],0)
  y=mx.quantized_matmul(v,w.view(mx.uint32),s,group_size=32,bits=8,mode='mxfp8');out.append(y[:count])
 return mx.concatenate(out,0).reshape(*x.shape[:-1],w.shape[0])
r['micro']=[]
projections=[('head',m.lm_head)]
for layer in m.layers:
 if layer.is_linear:projections += [('kda_qkv',layer.self_attn._fused_qkv),('kda_o',layer.self_attn.o_proj)];break
for name,p in projections:
 if p is None:continue
 for dtype in (mx.bfloat16,mx.float32):
  x=mx.random.normal((1,1,p.weight.shape[-1])).astype(dtype);ref=original(x,p.weight,p.scales);mx.eval(ref)
  for tile in (16,32,64):
   v=nax(x,p.weight,p.scales,tile);mx.eval(v);err=mx.abs(v.astype(mx.float32)-ref.astype(mx.float32));times=[]
   for i in range(15):
    t=time.perf_counter();v=nax(x,p.weight,p.scales,tile);mx.eval(v);times.append(time.perf_counter()-t)
   row={'name':name,'shape':p.weight.shape,'dtype':str(dtype),'tile':tile,'times':times,'max_error':mx.max(err).item(),'exact':mx.array_equal(v,ref).item()};r['micro'].append(row);print('NAX_MICRO',name,str(dtype),tile,statistics.median(times[3:])*1000,row['max_error'],row['exact'],flush=True)
def clear():
 seen=set()
 def walk(v):
  if id(v) in seen or isinstance(v,mx.array):return
  seen.add(id(v))
  if isinstance(v,dict):
   for c in list(v.values()):walk(c)
  elif isinstance(v,(tuple,list)):
   for c in v:walk(c)
  elif type(v).__module__.endswith('glm5_next') or hasattr(v,'named_modules'):
   if hasattr(v,'graphs') and isinstance(v.graphs,dict):v.graphs.clear()
   for c in list(vars(v).values()):walk(c)
 walk(m);m.configure_eval_groups(decode=2,prefill=4,verification=2)
for mode in ('baseline','nax32','nax16','single','baseline'):
 f.mxfp8_project_rows=(original if mode=='baseline' else lambda x,w,s:mxfp8_single(x,w,s,tilegroups=1)) if mode in ('baseline','single') else lambda x,w,s:nax(x,w,s,32 if mode=='nax32' else 16)
 clear();generate(texts[0],16);v=generate(texts[0],64);v['mode']=mode;r['samples'].append(v);print('NAX_FULL',mode,v['tok_s'],flush=True);(root/'nax_ab.json').write_text(json.dumps(r,indent=2))
