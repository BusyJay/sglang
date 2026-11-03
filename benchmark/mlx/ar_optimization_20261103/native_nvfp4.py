# Diagnostic adaptation of Apple MLX MIT fp_qmv_fast_impl; retain native lane accumulation.
import mlx.core as mx
from functools import cache
_HEADER=r'''
inline float fp4(uchar v){half c=as_type<half>(ushort((v&7)<<9));c*=16384.0;return float((v&8)?-c:c);}
inline float fp8(uchar b){ushort v=b&127;ushort u=(v<<7)|(((v+1)>>7)<<14)|(ushort(b>>7)<<15);return float(as_type<half>(u)*256.0);}
'''
_SOURCE=r'''
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint pick=threadgroup_position_in_grid.z,token=pick/PICKS;
const uint expert=ids[pick];
const uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
float values[16],result[RPS]={0};
for(uint k=0;k<K;k+=512){
 for(uint i=0;i<16;++i)values[i]=float(x[(EXPERT?pick:token)*K+k+lane*16+i]);
 for(uint r=0;r<RPS;++r){
  uint rr=min(row+r,uint(N-1));
  const device ushort* w=(const device ushort*)(weight+(size_t(expert)*N+rr)*(K/2)+(k+lane*16)/2);
  float acc=0;
  for(uint i=0;i<4;++i){uint q=w[i];acc+=(values[4*i]*fp4(q)+values[4*i+1]*fp4(q>>4)+values[4*i+2]*fp4(q>>8)+values[4*i+3]*fp4(q>>12));}
  float scale=fp8(scales[(size_t(expert)*N+rr)*(K/16)+(k+lane*16)/16]);
  result[r]+=scale*acc;
 }
}
float global=global_scale[expert]/2688.0f;
for(uint r=0;r<RPS;++r){float sum=simd_sum(result[r]);if(lane==0&&row+r<N)out[pick*N+row+r]=O(sum*global);}
'''
@cache
def _kernel():return mx.fast.metal_kernel(name='diagnostic_nvfp4_native_rows',input_names=['x','weight','scales','global_scale','ids'],output_names=['out'],source=_SOURCE,header=_HEADER)
def project(x,w,s,g,ids,rps=4):
 e,n,pk=w.shape;k=pk*2;count=ids.size;picks=ids.shape[-1]
 return _kernel()(inputs=[x.reshape(-1,k),w,s,(g*2688).reshape(-1),ids.reshape(-1).astype(mx.uint32)],template=[('O',x.dtype),('K',k),('N',n),('RPS',rps),('PICKS',picks),('EXPERT',x.size//k==count)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*ids.shape,1,n)
