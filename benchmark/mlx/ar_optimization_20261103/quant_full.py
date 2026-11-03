exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
from quant_direct import direct
original=f.mlx_fp8_dynamic_32
exec(open('benchmark/mlx/ar_optimization_20261103/single_ab.py').read().split('# Clear graph owners recursively for valid dispatch A/B.\n')[1].split('for mode in (')[0])
for mode in ('baseline','direct','direct','baseline'):
 f.mlx_fp8_dynamic_32=original if mode=='baseline' else direct;clear();generate(texts[0],16)
 for text in texts:
  v=generate(text,96);v['mode']=mode;r['samples'].append(v);print('QUANT_FULL',mode,text[:8],v['tok_s'],flush=True)
 (root/'quant_full.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
