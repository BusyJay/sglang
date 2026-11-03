exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
from block_hoist import fn
original=f.mlx_fp8_rows
r['micro']=[];seen=set()
for name,p in m.named_modules():
 if not isinstance(p,f.FP8Linear) or p.block_size!=128 or tuple(p.weight.shape) in seen:continue
 seen.add(tuple(p.weight.shape))
 for dtype in (mx.float32,mx.bfloat16):
  for count in (1,3,6):
   x=mx.random.normal((1,count,p.weight.shape[1])).astype(dtype);q=f.mlx_fp8_dynamic_32(x);a=fn(x,p.weight,p.scales,block_size=128,quantized_input=q);b=original(x,p.weight,p.scales,block_size=128,quantized_input=q);mx.eval(a,b);row={'name':name,'dtype':str(dtype),'count':count,'exact':mx.array_equal(a,b).item()};r['micro'].append(row);assert row['exact'],row
exec(open('benchmark/mlx/ar_optimization_20261103/single_ab.py').read().split('# Clear graph owners recursively for valid dispatch A/B.\n')[1].split('for mode in (')[0])
for mode in ('baseline','hoist','hoist','baseline'):
 f.mlx_fp8_rows=original if mode=='baseline' else fn;clear();generate(texts[0],16)
 for text in texts:
  v=generate(text,64);v['mode']=mode;r['samples'].append(v);print('HOIST_FULL',mode,text[:8],v['tok_s'],flush=True)
 (root/'block_hoist_ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
