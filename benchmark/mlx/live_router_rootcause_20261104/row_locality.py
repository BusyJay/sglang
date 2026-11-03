"""M5 lossless row-tiled expert interleave with exact native FP4 arithmetic.

Same original-ID router and worklist. Row-tile addresses gather experts closer
at each N tile; no activation or checkpoint quantization changes. Both layouts
use the same custom dispatch and disabled input contiguity materialization.
"""
import argparse,gc,hashlib,json,statistics,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
# Reuse diagnostic checkpoint loader, replacing only main benchmark code.
from importlib.util import spec_from_file_location,module_from_spec
R=Path('/Users/jay/workspace/sglang');O=R/'benchmark/mlx/live_router_rootcause_20261104'
s=(O/'production_route.py').read_text()
a=s.index('    def project(');b=s.index('    def forward(',a)
s=s[:a]+'''    kernels={}
    for tile in (0,8,32,128):
        src=nv._SOURCE
        if tile:
            src=src.replace('(size_t(expert)*N+row+r)', '((size_t((row+r)/TILE)*EXPERTS+expert)*TILE+(row+r)%TILE)')
        kernels[tile]=mx.fast.metal_kernel(name='diagnostic_nvfp4_row_interleave_'+str(tile),input_names=['x','weight','scales','global_scale','ids','dims','partners','counts'],output_names=['out'],source=src,header=nv._HEADER,ensure_row_contiguous=False,compile_options={'math_mode':'safe'})
    current_tile=0
    def project(value,p,ids,mode,work):
        w,scales,gs=p;k=w.shape[-1]*2;n=w.shape[-2];count=ids.size;picks=ids.shape[-1]
        tile=mode
        return kernels[tile](inputs=[value.reshape(-1,k),w,scales,(gs*2688).reshape(-1),ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',1),('EXPERT',value.size//k==count),('SINGLES',True),('TILE',max(1,tile)),('EXPERTS',288)],grid=(32,n,count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)

'''+s[b:]
s=s.replace("    def forward(value, ids, weights, mode, scope, order, placement):", "    def forward(value, ids, weights, params, mode, scope):")
s=s.replace("        work = mlx_expert_pair_worklist(ids) if mode == 'reuse1' and placement == 'before_shared' else None", "        work=mlx_expert_pair_worklist(ids)")
s=s.replace("        if mode == 'reuse1' and work is None:\n            work = mlx_expert_pair_worklist(ids)", "")
a=s.index("        if order == 'up_first':");b=s.index("        act = ",a)
s=s[:a]+"        up=project(v,params[1],ids,mode,work);gate=project(v,params[0],ids,mode,work)\n"+s[b:]
s=s.replace("project(act, bank[2]", "project(act, params[2]")
a=s.index('    specs = ')
s=s[:a]+'''    data=dict(scope=__doc__,device=mx.device_info(),layer=args.layer,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=[],samples=[],repack=[])
    specs=[(tile,scope) for tile in (0,8,32,128) for scope in ('live','live_x_after_weights')]
    fns={spec:mx.compile(lambda value,params,spec=spec:forward(value,captured_ids,captured_weights,params,*spec)) for spec in specs}
    for tile in (8,32,128):
        start=time.perf_counter();other=[]
        for w,scale,gs in bank:
            E,N,KP=w.shape
            ww=mx.contiguous(w.reshape(E,N//tile,tile,KP).transpose(1,0,2,3)).reshape(w.shape)
            ss=mx.contiguous(scale.reshape(E,N//tile,tile,scale.shape[-1]).transpose(1,0,2,3)).reshape(scale.shape)
            other.append((ww,ss,gs))
        mx.eval(other);data['repack'].append(dict(tile=tile,seconds=time.perf_counter()-start,active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory()))
        for (w,scale,gs),(ww,ss,_) in zip(bank,other):
            E,N,KP=w.shape
            assert mx.array_equal(mx.contiguous(ww.reshape(N//tile,E,tile,KP).transpose(1,0,2,3)).reshape(w.shape),w).item()
            assert mx.array_equal(mx.contiguous(ss.reshape(N//tile,E,tile,scale.shape[-1]).transpose(1,0,2,3)).reshape(scale.shape),scale).item()
        raw={(kind,scope):[] for kind in (0,tile) for scope in ('live','live_x_after_weights')}
        for scope in ('live','live_x_after_weights'):
            ref=fns[(0,scope)](x,bank);y=fns[(tile,scope)](x,other);mx.eval(ref,y)
            exact=mx.array_equal(ref,y).item();assert exact,(tile,scope);data['checks'].append(dict(tile=tile,scope=scope,exact=exact,packed_bytes_restored=True))
            for _ in range(4):mx.eval(fns[(0,scope)](x,bank),fns[(tile,scope)](x,other))
        seq=list(raw)
        for block in range(30):
            for key in seq if block%2==0 else list(reversed(seq)):
                kind,scope=key;params=bank if kind==0 else other
                for j in range(10):
                    value=x+mx.array((block*10+j)*1e-8,x.dtype);mx.eval(value);start=time.perf_counter();y=fns[key](value,params);mx.eval(y);raw[key].append(time.perf_counter()-start)
        for key,seconds in raw.items():
            row=dict(tile=tile,mode=key,raw_seconds=seconds,mean_ms=statistics.mean(seconds)*1000);data['samples'].append(row);print('ROWLOCALITY300',args.layer,tile,key,row['mean_ms'],flush=True)
        (OUT/f'row_locality_{args.layer}.json').write_text(json.dumps(data,indent=2))
        del other,params,ww,ss,y,ref;__import__('gc').collect();mx.clear_cache()
    data['complete']=True;(OUT/f'row_locality_{args.layer}.json').write_text(json.dumps(data,indent=2))

if __name__=='__main__':main()
'''
s=s.replace("import statistics", "import statistics").replace("source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()", "source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(s.encode()).hexdigest()")
ns={'__name__':'__main__','__file__':str(Path(__file__)),'s':s}
exec(compile(s,str(Path(__file__)),'exec'),ns)
