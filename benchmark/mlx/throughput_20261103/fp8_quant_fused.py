"""Diagnostic fuse exact dynamic32 quantization into block128 projection."""
from functools import cache
import importlib
import mlx.core as mx
base=importlib.import_module('sglang.kernels.ops.gemm.mlx_fp8_rows')
_HEADER=base._DECODE+r'''
inline uchar exact_encode(float input,float scale){
 float v=min(abs(input/scale),448.0f);
 uint code;
 if(v<0.015625f){code=uint(rint(v*512.0f));}
 else{
  uint bits=as_type<uint>(v);int exponent=int((bits>>23)&255)-127;
  uint mantissa=bits&0x7fffff;
  code=uint((exponent+7)*8)+(mantissa>>20);
  uint tail=mantissa&0xfffff;
  code+=uint(tail>0x80000||(tail==0x80000&&(code&1)));
  code=min(code,126u);
 }
 return uchar(code|(signbit(input)?128:0));
}
'''
_SOURCE=base._SOURCE
old='''for(uint t=0;t<ROWS;++t)inputs[t]=start+t<COUNT ? (A8 ? e4m3(xq[(start+t)*K+k])*xscale[(start+t)*(K/32)+k/32] : float(x[(start+t)*K+k])) : 0.0f;'''
new='''for(uint t=0;t<ROWS;++t){
 float value=start+t<COUNT ? float(x[(start+t)*K+k]) : 0.0f;
 float maximum=simd_max(abs(value));
 float scale=maximum>0.0f ? exp2(ceil(log2(maximum/448.0f))) : 1.0f;
 inputs[t]=e4m3(exact_encode(value,scale))*scale;
}'''
assert old in _SOURCE
_SOURCE=_SOURCE.replace(old,new)
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_fp8_quant_fused',input_names=['x','weight','scales','block_scales','xq','xscale'],output_names=['out'],source=_SOURCE,header=_HEADER)
def project(x,w,s):
 n,k=w.shape;count=x.size//k;rps=2 if n>=4096 else 1;rows=min(count,4)
 return kernel()(inputs=[x.reshape(count,k),w,mx.zeros((1,),mx.uint8),s.reshape(-1),mx.zeros((1,),mx.uint8),mx.zeros((1,),mx.float32)],template=[('O',x.dtype),('RPS',rps),('N',n),('K',k),('BLOCK',128),('MX',False),('A8',True),('COUNT',count),('ROWS',rows)],grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),threadgroup=(128,1,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
