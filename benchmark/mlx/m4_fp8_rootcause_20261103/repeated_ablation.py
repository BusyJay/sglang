"""Remove repeated scale calculation independently from repeated encoding."""

import json
import statistics
import time

import mlx.core as mx

from common import OUT, baseline, direct, load

repeated = load("quant_repeat_ablation", OUT / "repeated_quant.py")
source = repeated._SOURCE
old = """float maximum=simd_max(abs(value));
 float scale=maximum>0.0f ? exp2(ceil(log2(maximum/448.0f))) : 1.0f;"""
assert old in source
source = source.replace(old, "float scale=xscale[(start+t)*(K/32)+k/32];")
scale_kernel = mx.fast.metal_kernel(
    name="diagnostic_fp8_repeated_encode_only",
    input_names=["x", "weight", "scales", "block_scales", "xq", "xscale"],
    output_names=["out"], source=source, header=repeated._HEADER)


def encode_only(x, w, s, q):
    n,k=w.shape;count=x.size//k;rows=min(count,4);rps=2 if n>=4096 else 1
    return scale_kernel(inputs=[x,w,mx.zeros((1,),mx.uint8),s.reshape(-1),*q],
        template=[('O',x.dtype),('RPS',rps),('N',n),('K',k),('BLOCK',128),
                  ('MX',False),('A8',True),('COUNT',count),('ROWS',rows)],
        grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),
        threadgroup=(128,1,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0]


mx.set_cache_limit(256*2**20);mx.random.seed(711)
trials=[]
for n,k in ((2048,4096),(4096,2048)):
    w=mx.random.randint(0,126,(n,k),dtype=mx.uint8)
    s=mx.random.uniform(.001,.02,((n+127)//128,k//128));x=mx.random.normal((4,k))
    q=direct.mlx_fp8_dynamic_32(x);mx.eval(w,s,x,q)
    variants={
        'prequantized':lambda:baseline.mlx_fp8_rows(x,w,s,block_size=128,quantized_input=q),
        'repeat_encode':lambda:encode_only(x,w,s,q),
        'repeat_scale_encode':lambda:repeated.project(x,w,s)}
    expected=variants['prequantized']();mx.eval(expected)
    for name,fn in variants.items():
        result=fn();mx.eval(result);assert mx.array_equal(expected,result).item(),name
    for repeat in range(11):
        row={'n':n,'k':k,'repeat':repeat}
        names=list(variants)
        if repeat%2:names.reverse()
        for name in names:
            fn=variants[name];mx.eval(fn());start=time.perf_counter()
            for _ in range(20):mx.eval(fn())
            row[name]=(time.perf_counter()-start)/20
        trials.append(row)
    print('REPEAT_CAUSE',n,k,{name:statistics.median(t[name] for t in trials if t['n']==n)*1000
                              for name in variants},flush=True)
    (OUT/'repeated_ablation.json').write_text(json.dumps(trials,indent=2))
