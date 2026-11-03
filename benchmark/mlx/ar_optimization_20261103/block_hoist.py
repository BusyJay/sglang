from sglang.kernels.ops.gemm import mlx_fp8_rows as op
import mlx.core as mx
from functools import cache
old=op._SOURCE;a=old.index('for (uint k = lane');b=old.index('\nfor (uint t = 0;',a)
source=old[:a]+r'''
for (uint block=0;block<K/128;++block) {
    float ws[RPS];
    for(uint r=0;r<RPS;++r)ws[r]=block_scales[((row+r)/128)*(K/128)+block];
    for(uint step=0;step<4;++step) {
        uint k=block*128+step*32+lane;
        float inputs[ROWS];
        for(uint t=0;t<ROWS;++t)inputs[t]=start+t<COUNT ? (A8 ? e4m3(xq[(start+t)*K+k])*xscale[(start+t)*(K/32)+k/32] : float(x[(start+t)*K+k])) : 0.0f;
        for(uint r=0;r<RPS;++r){
            if(row+r>=N)continue;
            float wv=e4m3(weight[size_t(row+r)*K+k])*ws[r];
            for(uint t=0;t<ROWS;++t)acc[t][r]+=inputs[t]*wv;
        }
    }
}
'''+old[b:]
@cache
def kernel():return mx.fast.metal_kernel(name='diagnostic_fp8_hoist_scale',input_names=['x','weight','scales','block_scales','xq','xscale'],output_names=['out'],source=source,header=op._DECODE)
def fn(x,w,s,*,block_size=32,quantized_input=None):
 if block_size!=128:return op.mlx_fp8_rows(x,w,s,block_size=block_size,quantized_input=quantized_input)
 n,k=w.shape;count=x.size//k;rps=2 if n>=4096 else 1;q,xs=quantized_input if quantized_input is not None else (mx.zeros((1,),mx.uint8),mx.zeros((1,),mx.float32))
 return kernel()(inputs=[x.reshape(count,k),w,mx.zeros((1,),mx.uint8),s.reshape(-1),q,xs],template=[('O',x.dtype),('RPS',rps),('N',n),('K',k),('BLOCK',128),('MX',False),('A8',quantized_input is not None),('COUNT',count),('ROWS',min(count,4))],grid=(((n+4*rps-1)//(4*rps))*128,1,(count+min(count,4)-1)//min(count,4)),threadgroup=(128,1,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
