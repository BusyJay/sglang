"""M5 exact register-table FP4 decoding ablation, actual full banks.

The eight FP4 magnitudes times two are packed in a scalar integer. Conversion
changes no values, scales, reduction order or storage. Compare native, original
paired and table-decoded paired kernels, with 300 alternating calls per arm.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from sglang.kernels.ops.gemm import mlx_nvfp4_gather as nv
from sglang.kernels.ops.moe.mlx_expert_pairs import mlx_expert_pair_worklist
O=Path('benchmark/mlx/remote_validation_20261103')
C=Path('/Users/jay/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8')
mx.set_memory_limit(10*2**30);mx.set_cache_limit(256*2**20)
inputs=mx.load(O/'actual_ffn_operands.npz')
new_header=r'''
inline float sglang_fp4_table(uchar code) {
    uint magnitude = (0xc8643210u >> (uint(code & 7) * 4)) & 15;
    uint bits = as_type<uint>(float(magnitude) * 0.5f);
    return as_type<float>(bits | ((uint(code) & 8) << 28));
}
'''
import re
source=re.sub(r'float\(\*\(thread fp4_e2m1\*\)\(&([a-z][0-9])\)\)',r'sglang_fp4_table(\1)',nv._SOURCE)
assert source!=nv._SOURCE and 'float(*(thread fp4_e2m1*)' not in source
fast=mx.fast.metal_kernel(name='diagnostic_nvfp4_table',input_names=['x','weight','scales','global_scale','ids','dims','partners','counts'],output_names=['out'],source=source,header=nv._HEADER+new_header,compile_options={'math_mode':'safe'})
def project(value,w,s,gs,ids,work,rps,table):
    count,picks=ids.size,ids.shape[-1];e,n,pk=w.shape;k=pk*2
    kernel=fast if table else nv._kernel(False)
    return kernel(inputs=[value.reshape(-1,k),w,s,gs.reshape(-1),ids.reshape(-1),mx.array([k,n,picks,count],mx.int32),*work],template=[('O',value.dtype),('RPS',rps),('EXPERT',value.size//k==count),('SINGLES',True)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
data=dict(scope=__doc__,device=mx.device_info(),cases=[],source_sha256=hashlib.sha256(Path('python/sglang/kernels/ops/gemm/mlx_nvfp4_gather.py').read_bytes()).hexdigest())
for layer in (3,26,41):
    a=mx.load(C/f'layer-{layer:02}.safetensors');prefix='mlp.switch_mlp.gate_proj.'
    w,s,glob=[a[prefix+n] for n in ('weight','scales','weight_scale_2')];mx.eval(w,s,glob);del a
    gs=glob*2688
    for count in (1,4):
        x=inputs[f'c{count}_l{layer}_x'][...,None,None,:];ids=inputs[f'c{count}_l{layer}_ids'];work=mlx_expert_pair_worklist(ids);mx.eval(gs,*work)
        fns={'native':mx.compile(lambda value:mx.gather_qmm(value,w.view(mx.uint32),s,rhs_indices=ids,mode='nvfp4',group_size=16,bits=4,global_scale=gs))}
        for rps in (1,2,4):
            for table in (False,True):
                fns[('table' if table else 'original')+str(rps)]=mx.compile(lambda value,rps=rps,table=table:project(value,w,s,gs,ids,work,rps,table))
        ref=fns['native'](x)
        for name,fn in fns.items():
            value=fn(x);mx.eval(ref,value);assert mx.array_equal(ref,value).item(),(layer,count,name)
            assert mx.all(mx.isfinite(value)).item(),(layer,count,name)
        raw={name:[] for name in fns}
        for block in range(30):
            for name in list(fns) if block%2==0 else list(reversed(fns)):
                for step in range(10):
                    value=x+mx.array((block*10+step)*1e-8,x.dtype);mx.eval(value)
                    t=time.perf_counter();result=fns[name](value);mx.eval(result);raw[name].append(time.perf_counter()-t)
        row=dict(layer=layer,count=count,samples=[dict(mode=name,mean_ms=statistics.mean(raw[name])*1000,raw_seconds=raw[name]) for name in fns]);data['cases'].append(row)
        (O/'decode_ab.json').write_text(json.dumps(data,indent=2));print('DECODE_AB',layer,count,{name:round(statistics.mean(raw[name])*1000,4) for name in fns},flush=True)
    del w,s,glob,fns;mx.clear_cache()
data['complete']=True;(O/'decode_ab.json').write_text(json.dumps(data,indent=2))
