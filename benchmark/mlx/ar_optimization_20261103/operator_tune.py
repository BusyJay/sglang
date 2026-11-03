exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
import types,ast,statistics
from sglang.kernels.ops.gemm import mlx_fp8_rows as op
# Actual block-FP8 kernels: vary independent output row count without changing accumulation.
source=Path(op.__file__).read_text();tree=ast.parse(source);node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='mlx_fp8_rows');text=ast.get_source_segment(source,node)
variants={}
for rps in [1,2,4,8]:
 ns=dict(op.__dict__);snippet=text.replace('rps = 2 if block_size == 128 and n >= 4096 else 4 if n >= 4096 else 1','rps = '+str(rps));exec(snippet,ns);variants[rps]=ns['mlx_fp8_rows']
modules=[(name,p) for name,p in m.named_modules() if isinstance(p,f.FP8Linear) and p.block_size==128]
r['ops']=[];seen=set()
for name,p in modules:
 shape=tuple(p.weight.shape)
 if shape in seen:continue
 seen.add(shape);x=mx.random.normal((1,1,shape[1])).astype(mx.float32 if 'shared' in name else mx.bfloat16);q=op.mlx_fp8_dynamic_32(x);baseline=op.mlx_fp8_rows(x,p.weight,p.scales,block_size=128,quantized_input=q);mx.eval(baseline,q)
 for rps,fn in variants.items():
  out=fn(x,p.weight,p.scales,block_size=128,quantized_input=q);mx.eval(out);times=[]
  for i in range(18):
   t=time.perf_counter();out=fn(x,p.weight,p.scales,block_size=128,quantized_input=q);mx.eval(out);times.append(time.perf_counter()-t)
  row={'name':name,'shape':shape,'rps':rps,'dtype':str(x.dtype),'times':times,'exact':mx.array_equal(out,baseline).item(),'max_error':mx.max(mx.abs(out.astype(mx.float32)-baseline.astype(mx.float32))).item()};r['ops'].append(row);print('OP',name,shape,rps,statistics.median(times[3:])*1000,row['exact'],flush=True)
(root/'operator_tune.json').write_text(json.dumps(r,indent=2))
# full-model plans use per-shape best micro choice, then test opposite order.
best={}
for shape in seen:
 entries=[v for v in r['ops'] if tuple(v['shape'])==shape];best[shape]=min(entries,key=lambda v:statistics.median(v['times'][3:]))['rps']
def tuned(x,w,s,**kwargs):return variants[best.get(tuple(w.shape),2)](x,w,s,**kwargs)
original=f.mlx_fp8_rows
for mode in ['baseline','tuned','tuned','baseline']:
 f.mlx_fp8_rows=tuned if mode=='tuned' else original
 m.configure_fixed_groups(True)
 # Existing groups need a fresh trace for projection dispatch choices.
 for layer in m.layers:
  for owner in (getattr(layer,'_fixed_mla_attention',None),getattr(layer,'_fixed_mla_layer',None)):
   if owner is not None:owner.graphs.clear()
 m.configure_eval_groups(decode=2,prefill=4,verification=2)
 generate(texts[0],16);v=generate(texts[0],64);v['mode']=mode;r['samples'].append(v);print('FULL',mode,v['tok_s'],flush=True)
(root/'operator_tune.json').write_text(json.dumps(r,indent=2))
