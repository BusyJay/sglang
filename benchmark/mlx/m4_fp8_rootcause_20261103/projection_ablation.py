"""Isolate pairing, output geometry and E4M3 decode instructions."""

import json
import statistics
import time

import mlx.core as mx

from common import OUT, baseline, direct, paired

HALF_HEADER = r"""
inline float mlx_e4m3(uchar bits) {
    ushort v = bits & 127;
    ushort sign_bit = ushort((bits >> 7) & 1) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return float(converted * 256.0);
}
"""
decode_check = mx.fast.metal_kernel(
    name="diagnostic_fp8_decode_all",
    input_names=["codes"], output_names=["out"],
    source="out[thread_position_in_grid.x] = mlx_e4m3(codes[thread_position_in_grid.x]);",
    header=HALF_HEADER)
ref_check = mx.fast.metal_kernel(
    name="diagnostic_fp8_decode_ref",
    input_names=["codes"], output_names=["out"],
    source="out[thread_position_in_grid.x] = mlx_e4m3(codes[thread_position_in_grid.x]);",
    header=baseline._DECODE.replace("e4m3(", "mlx_e4m3("))
codes = mx.arange(256).astype(mx.uint8)
kw = dict(inputs=[codes], grid=(256,1,1), threadgroup=(256,1,1),
          output_shapes=[(256,)], output_dtypes=[mx.float32])
a = decode_check(**kw)[0]; b = ref_check(**kw)[0]
mx.eval(a,b)
assert mx.array_equal(mx.isnan(a), mx.isnan(b)).item()
assert mx.all((a == b) | (mx.isnan(a) & mx.isnan(b))).item()
assert mx.all((a.view(mx.uint32) == b.view(mx.uint32)) | mx.isnan(a)).item()
assert mx.array_equal(mx.sign(a),mx.sign(b),equal_nan=True).item()
print("DECODE256_EXACT", flush=True)

half_kernel = mx.fast.metal_kernel(
    name="diagnostic_fp8_pair_half_decode",
    input_names=["gate_weight", "gate_scales", "up_weight", "up_scales",
                 "xq", "xscale", "dims"], output_names=["gate", "up"],
    source=paired._SOURCE, header=HALF_HEADER)
branch_kernel = mx.fast.metal_kernel(
    name="diagnostic_fp8_pair_branch_decode",
    input_names=["gate_weight", "gate_scales", "up_weight", "up_scales",
                 "xq", "xscale", "dims"], output_names=["gate", "up"],
    source=paired._SOURCE, header=baseline._DECODE.replace("e4m3(", "mlx_e4m3("))


def half_pair(x, gw, gs, uw, us, quantized, rps=1, kernel=half_kernel):
    n,k=gw.shape;count=x.size//k;rows=min(count,4)
    return kernel(inputs=[gw,gs.reshape(-1),uw,us.reshape(-1),*quantized,
                               mx.array([k,n,count],mx.uint32)],
        template=[("O",x.dtype),("RPS",rps),("ROWS",rows)],
        grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),
        threadgroup=(128,1,1),output_shapes=[(count,n),(count,n)],
        output_dtypes=[x.dtype,x.dtype])


if __name__ == "__main__":
    mx.set_cache_limit(256*2**20);mx.random.seed(541)
    trials=[];checks=[]
    for n,k in ((2048,4096),(4096,2048),(4096,16384)):
        gw=mx.random.randint(0,255,(n,k),dtype=mx.uint8)
        uw=mx.random.randint(0,255,(n,k),dtype=mx.uint8)
        gw=mx.where((gw&127)==127,0,gw);uw=mx.where((uw&127)==127,0,uw)
        gs=mx.random.uniform(0.001,0.02,((n+127)//128,k//128))
        us=mx.random.uniform(0.001,0.02,gs.shape)
        mx.eval(gw,uw,gs,us)
        for count in (1,4):
            x=mx.random.normal((count,k));q=direct.mlx_fp8_dynamic_32(x);mx.eval(x,q)
            expected=(baseline.mlx_fp8_rows(x,gw,gs,block_size=128,quantized_input=q),
                      baseline.mlx_fp8_rows(x,uw,us,block_size=128,quantized_input=q))
            mx.eval(expected)
            variants={"separate":lambda: (
                baseline.mlx_fp8_rows(x,gw,gs,block_size=128,quantized_input=q),
                baseline.mlx_fp8_rows(x,uw,us,block_size=128,quantized_input=q))}
            for rps in (1,2,4):
                variants[f"pair{rps}"]=lambda rps=rps:half_pair(x,gw,gs,uw,us,q,rps,branch_kernel)
                variants[f"half{rps}"]=lambda rps=rps:half_pair(x,gw,gs,uw,us,q,rps)
            for name,fn in variants.items():
                result=fn();mx.eval(result)
                assert all(mx.array_equal(a,b).item() for a,b in zip(expected,result)),(n,k,count,name)
            checks.append({"n":n,"k":k,"count":count,"exact":True})
            for repeat in range(9):
                row={"n":n,"k":k,"count":count,"repeat":repeat}
                names=list(variants)
                if repeat%2:names.reverse()
                for name in names:
                    fn=variants[name];mx.eval(fn());start=time.perf_counter()
                    for _ in range(15):mx.eval(fn())
                    row[name]=(time.perf_counter()-start)/15
                trials.append(row)
            print("PROJECT_ABLATION",n,k,count,
                {name:statistics.median(r[name] for r in trials if (r['n'],r['k'],r['count'])==(n,k,count))*1000
                 for name in variants},flush=True)
            (OUT/'projection_ablation.json').write_text(json.dumps({"checks":checks,"trials":trials},indent=2))
