exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
optimized=f.mxfp8_project_rows
# Frozen production reference, same checkpoint and four-row native rounding.
def baseline(x,w,s):
 values=x.reshape(-1,x.shape[-1]);out=[]
 for start in range(0,values.shape[0],4):
  tile=values[start:start+4];count=tile.shape[0]
  if count<4:tile=mx.concatenate([tile,mx.zeros((4-count,tile.shape[1]),tile.dtype)])
  out.append(mx.quantized_matmul(tile,w.view(mx.uint32),s,mode='mxfp8',group_size=32,bits=8)[:count])
 return mx.concatenate(out).reshape(*x.shape[:-1],w.shape[0])
exec(open('benchmark/mlx/ar_optimization_20261103/single_ab.py').read().split('# Clear graph owners recursively for valid dispatch A/B.\n')[1].split('for mode in (')[0])
r['micro']=[]
seen=set()
for name,p in [('head',m.lm_head)]+list(m.named_modules()):
 if not isinstance(p,f.FP8Linear) or p.block_size!=32 or tuple(p.weight.shape) in seen:continue
 seen.add(tuple(p.weight.shape))
 for dtype in (mx.float32,mx.bfloat16):
  for count in (1,3,6,9):
   x=mx.random.normal((1,count,p.weight.shape[1])).astype(dtype);a=optimized(x,p.weight,p.scales);b=baseline(x,p.weight,p.scales);mx.eval(a,b);row={'name':name,'shape':p.weight.shape,'dtype':str(dtype),'count':count,'exact':mx.array_equal(a,b).item(),'max_error':mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item()};r['micro'].append(row);assert row['exact'],row
for mode in ('baseline','optimized','optimized','baseline'):
 f.mxfp8_project_rows=baseline if mode=='baseline' else optimized;clear();generate(texts[0],16)
 for text in texts:
  v=generate(text,128);v['mode']=mode;r['samples'].append(v);print('FINAL_AR',mode,text[:8],v['tok_s'],'prefill',v['prefill_s'],flush=True)
 (root/'final_ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
r['swap_after']=subprocess.check_output(['sysctl','vm.swapusage'],text=True);(root/'final_ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
