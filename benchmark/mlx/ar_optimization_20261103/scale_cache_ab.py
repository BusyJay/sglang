exec(open('benchmark/mlx/ar_optimization_20261103/pilot.py').read().split('for group in (2,4,1,8,16,2):')[0])
for name,p in m.named_modules():
 if isinstance(p,f.NVFP4SwitchLinear):p['calibrated_scale']=p.weight_scale_2*2688;mx.eval(p['calibrated_scale'])
original=f.NVFP4SwitchLinear.__call__
def cached(self,x,indices,sorted_indices=False):
 return mx.gather_qmm(x,self.weight.view(mx.uint32),self.scales,rhs_indices=indices,mode='nvfp4',group_size=16,bits=4,global_scale=self.calibrated_scale,sorted_indices=sorted_indices)
exec(open('benchmark/mlx/ar_optimization_20261103/single_ab.py').read().split('# Clear graph owners recursively for valid dispatch A/B.\n')[1].split('for mode in (')[0])
for mode in ('baseline','cache','cache','baseline'):
 f.NVFP4SwitchLinear.__call__=original if mode=='baseline' else cached;clear();generate(texts[0],16)
 v=generate(texts[0],64);v['mode']=mode;r['samples'].append(v);print('SCALE_FULL',mode,v['tok_s'],flush=True)
 (root/'scale_cache_ab.json').write_text(json.dumps(r,indent=2))
