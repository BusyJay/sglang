"""Diagnostic native-order gather QMV sharing same-expert reads across tokens."""
from functools import cache
from nv_exact_pair import _HEADER
import mlx.core as mx
_SOURCE=r'''
const int K=dims[0],N=dims[1],PICKS=dims[2],COUNT=dims[3];
uint pick=threadgroup_position_in_grid.z,expert=ids[pick];
for(uint i=0;i<pick;++i)if(ids[i]==expert)return;
uint matches[ROWS],nmatch=0;
for(uint i=pick;i<COUNT;++i)if(ids[i]==expert)matches[nmatch++]=i;
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
float result[ROWS][RPS]={0};
for(int k=0;k<K;k+=512){
 float xv[ROWS][16];
 for(uint t=0;t<nmatch;++t){
  uint vector=EXPERT ? matches[t] : matches[t]/PICKS;
  for(int i=0;i<16;++i)xv[t][i]=float(x[size_t(vector)*K+k+lane*16+i]);
 }
 for(int r=0;r<RPS;++r){
  if(row+r>=N)continue;
  const device ushort* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
  uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
  float scale=float(*(thread fp8_e4m3*)(&s));
  float wv[16];
  for(int j=0;j<4;++j){ushort q=w[j];uchar a=q,b=q>>4,c=q>>8,d=q>>12;
   wv[4*j]=float(*(thread fp4_e2m1*)(&a));wv[4*j+1]=float(*(thread fp4_e2m1*)(&b));wv[4*j+2]=float(*(thread fp4_e2m1*)(&c));wv[4*j+3]=float(*(thread fp4_e2m1*)(&d));
  }
  for(uint t=0;t<nmatch;++t){
   float accum=0;
   for(int j=0;j<4;++j)accum+=(xv[t][4*j]*wv[4*j]+xv[t][4*j+1]*wv[4*j+1]+xv[t][4*j+2]*wv[4*j+2]+xv[t][4*j+3]*wv[4*j+3]);
   result[t][r]=fma(scale,accum,result[t][r]);
  }
 }
}
float global=global_scale[expert]/2688.0f;
for(uint t=0;t<nmatch;++t)for(int r=0;r<RPS;++r){
 float value=simd_sum(result[t][r]);
 if(lane==0&&row+r<N)out[matches[t]*N+row+r]=O(value*global);
}
'''
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_shared_expert',input_names=['x','weight','scales','global_scale','ids','dims'],output_names=['out'],source=_SOURCE,header=_HEADER)
def project(x,w,s,g,ids,rps=1):
 e,n,pk=w.shape;k=pk*2;count=ids.size;rows=ids.shape[1]
 return kernel()(inputs=[x.reshape(-1,k),w,s,(g*2688).reshape(-1),ids.reshape(-1).astype(mx.uint32),mx.array([k,n,ids.shape[-1],count],mx.int32)],template=[('O',x.dtype),('ROWS',rows),('RPS',rps),('EXPERT',x.size//k==count)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*ids.shape,1,n)
