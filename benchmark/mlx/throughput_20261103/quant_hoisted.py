"""Diagnostic fuse gate/up quantization; one block quant per shared input."""
from functools import cache
import mlx.core as mx
from fp8_quant_fused import _HEADER
_SOURCE=r'''
const int K=dims[0],N=dims[1],COUNT=dims[2];
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint row=(threadgroup_position_in_grid.x*4+sg)*RPS,start=threadgroup_position_in_grid.z*ROWS;
float g[ROWS][RPS]={0},u[ROWS][RPS]={0};
for(uint block=0;block<K/128;++block){
 float gs[RPS],us[RPS];
 for(uint r=0;r<RPS;++r){gs[r]=sgate[((row+r)/128)*(K/128)+block];us[r]=sup[((row+r)/128)*(K/128)+block];}
 for(uint step=0;step<4;++step){
  uint k=block*128+step*32+lane;
  float inputs[ROWS];
  for(uint t=0;t<ROWS;++t)inputs[t]=start+t<COUNT ? e4m3(xq[(start+t)*K+k])*xscale[(start+t)*(K/32)+k/32]:0.0f;
  for(uint r=0;r<RPS;++r){
   if(row+r>=N)continue;
   float gv=e4m3(wgate[size_t(row+r)*K+k])*gs[r],uv=e4m3(wup[size_t(row+r)*K+k])*us[r];
   for(uint t=0;t<ROWS;++t){g[t][r]+=inputs[t]*gv;u[t][r]+=inputs[t]*uv;}
  }
 }
}
for(uint t=0;t<ROWS;++t)for(uint r=0;r<RPS;++r){
 for(ushort off=16;off>=1;off>>=1){g[t][r]+=simd_shuffle_down(g[t][r],off);u[t][r]+=simd_shuffle_down(u[t][r],off);}
 if(lane==0&&start+t<COUNT&&row+r<N){gate[(start+t)*N+row+r]=O(g[t][r]);up[(start+t)*N+row+r]=O(u[t][r]);}
}
'''
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_fp8_gateup_shared',input_names=['wgate','sgate','wup','sup','xq','xscale','dims'],output_names=['gate','up'],source=_SOURCE,header=_HEADER)
def pair(x,g,u,quantized):
 n,k=g.weight.shape;count=x.size//k;rows=min(count,4);rps=1
 return kernel()(inputs=[g.weight,g.scales.reshape(-1),u.weight,u.scales.reshape(-1),*quantized,mx.array([k,n,count],mx.int32)],template=[('O',x.dtype),('RPS',rps),('ROWS',rows)],grid=(((n+4*rps-1)//(4*rps))*128,1,(count+rows-1)//rows),threadgroup=(128,1,1),output_shapes=[(count,n),(count,n)],output_dtypes=[x.dtype,x.dtype])
