"""Diagnostic cache of exact E4M3 values; original UE8M0 group scales/order."""
from functools import cache
import importlib
import mlx.core as mx
base=importlib.import_module('sglang.kernels.ops.gemm.mlx_mxfp8_rows')
_DECODE='''
uint i=thread_position_in_grid.x;
if(i<COUNT)out[i]=half(sglang_e4m3(w[i]));
'''
@cache
def decoder():
 return mx.fast.metal_kernel(name='diagnostic_mxfp8_half_cache',input_names=['w'],output_names=['out'],header=base._HEADER,source=_DECODE)
def decode(w):
 return decoder()(inputs=[w],template=[('COUNT',w.size)],grid=(w.size,1,1),threadgroup=(256,1,1),output_shapes=[w.shape],output_dtypes=[mx.float16])[0]
_SOURCE=base._SOURCE.replace('const device uchar* wrow = (const device uchar*)weight + size_t(row) * K;','const device half* wrow = weight + size_t(row) * K;').replace('const device uchar* wg = wrow + k0;','const device half* wg = wrow + k0;')
import re
_SOURCE=re.sub(r'sglang_e4m3\((wg\[[^\]]+\])\)',r'float(\1)',_SOURCE)
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_mxfp8_half_shared',input_names=['x','weight','scales','dims'],output_names=['out'],source=_SOURCE,header=base._HEADER)
def project(x,w,s):
 n,k=w.shape;count=x.size//k;t=min(count,4)
 return kernel()(inputs=[x.reshape(count,k),w,s,mx.array([k,n,count],mx.int32)],template=[('T',x.dtype),('TILE',t),('KL',16)],grid=(32*((count+t-1)//t),2*((n+3)//4),1),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
