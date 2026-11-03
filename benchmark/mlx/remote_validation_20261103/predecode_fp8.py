"""Lossless startup FP8 -> FP16 code expansion; original FP32 block scaling."""
import mlx.core as mx
from functools import cache
from sglang.kernels.ops.gemm import mlx_block_fp8 as fp
@cache
def decoder():
 return mx.fast.metal_kernel(name='diagnostic_fp8_exact_half_expand',input_names=['codes','dims'],output_names=['half_weight'],source='uint i=thread_position_in_grid.x;if(i<dims[0])half_weight[i]=half(mlx_e4m3(codes[i]));',header=fp._DECODE)
def decode(weight):
 return decoder()(inputs=[weight.reshape(-1),mx.array([weight.size],mx.uint32)],grid=(weight.size,1,1),threadgroup=(256,1,1),output_shapes=[weight.shape],output_dtypes=[mx.float16])[0]
@cache
def kernel():
 source=fp._SOURCE.replace('mlx_e4m3(gate_weight[size_t(row+r)*K+k])','float(gate_weight[size_t(row+r)*K+k])').replace('mlx_e4m3(up_weight[size_t(row+r)*K+k])','float(up_weight[size_t(row+r)*K+k])')
 return mx.fast.metal_kernel(name='diagnostic_exact_half_fp8_pair',input_names=['gate_weight','gate_scales','up_weight','up_scales','xq','xscale','dims'],output_names=['gate','up'],source=source,header=fp._DECODE)
def project(x,gw,gs,uw,us,q,rps=2):
 n,k=gw.shape;count=x.size//k;rows=min(count,4)
 return tuple(v.reshape(*x.shape[:-1],n) for v in kernel()(inputs=[gw,gs.reshape(-1),uw,us.reshape(-1),*q,mx.array([k,n,count],mx.uint32)],template=[('O',x.dtype),('RPS',rps),('ROWS',rows)],grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),threadgroup=(128,1,1),output_shapes=[(count,n)]*2,output_dtypes=[x.dtype]*2))
