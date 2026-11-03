"""Diagnostic exact FP32 decoded block weights; native lane order/dynamic32."""
from functools import cache
import importlib
import mlx.core as mx
base=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows')
_DECODE=r'''
const int K=dims[0],N=dims[1];
uint i=thread_position_in_grid.x;
if(i<uint(K)*uint(N)){
 uint row=i/K,k=i%K;
 out[i]=e4m3(weight[i])*scale[(row/128)*(K/128)+k/128];
}
'''
@cache
def decoder():
 return mx.fast.metal_kernel(name='diagnostic_fp8_f32_cache',input_names=['weight','scale','dims'],output_names=['out'],source=_DECODE,header=base._DECODE)
def decode(w,s):
 n,k=w.shape
 return decoder()(inputs=[w,s,mx.array([k,n],mx.int32)],grid=(w.size,1,1),threadgroup=(256,1,1),output_shapes=[w.shape],output_dtypes=[mx.float32])[0]
_SOURCE=base._SOURCE.replace('float wv=e4m3(weight[size_t(row+r)*K+k])*ws[r];','float wv=weight[size_t(row+r)*K+k];')
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_fp8_f32_shared',input_names=['x','weight','scales','block_scales','xq','xscale'],output_names=['out'],source=_SOURCE,header=base._DECODE)
def project(x,w,s,quantized):
 n,k=w.shape;count=x.size//k;rps=2 if n>=4096 else 1;rows=min(count,4);xq,xscale=quantized
 return kernel()(inputs=[x.reshape(count,k),w,mx.zeros((1,),mx.uint8),s.reshape(-1),xq,xscale],template=[('O',x.dtype),('RPS',rps),('N',n),('K',k),('BLOCK',128),('MX',False),('A8',True),('COUNT',count),('ROWS',rows)],grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),threadgroup=(128,1,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
