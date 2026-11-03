"""Diagnostic native-order NVFP4 gate/up/SwiGLU with expert-pair reuse."""
import mlx.core as mx
from functools import cache
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _HEADER
_SOURCE=r'''
uint pick=threadgroup_position_in_grid.z,count=counts[pick];
if(count==0)return;
uint expert=ids[pick],partner=partners[pick];
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const int K=dims[0],N=dims[1],PICKS=dims[2];
uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
uint v0=pick/PICKS,v1=partner/PICKS;
float g0[RPS]={0},u0[RPS]={0},g1[RPS]={0},u1[RPS]={0};
for(int k=0;k<K;k+=512){
 float x0[16],x1[16];
 #pragma unroll
 for(int i=0;i<16;++i){x0[i]=float(x[size_t(v0)*K+k+lane*16+i]);x1[i]=count==2?float(x[size_t(v1)*K+k+lane*16+i]):0;}
 #pragma unroll
 for(int r=0;r<RPS;++r){
  size_t wi=(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2;
  size_t si=(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16;
  const device ushort* g=(const device ushort*)(wg+wi);
  const device ushort* u=(const device ushort*)(wu+wi);
  float ag0=0,au0=0,ag1=0,au1=0;
  #pragma unroll
  for(int j=0;j<4;++j){
   ushort gp=g[j],up=u[j];uchar gc0=gp,gc1=gp>>4,gc2=gp>>8,gc3=gp>>12,uc0=up,uc1=up>>4,uc2=up>>8,uc3=up>>12;
   float gw0=float(*(thread fp4_e2m1*)(&gc0)),gw1=float(*(thread fp4_e2m1*)(&gc1)),gw2=float(*(thread fp4_e2m1*)(&gc2)),gw3=float(*(thread fp4_e2m1*)(&gc3));
   float uw0=float(*(thread fp4_e2m1*)(&uc0)),uw1=float(*(thread fp4_e2m1*)(&uc1)),uw2=float(*(thread fp4_e2m1*)(&uc2)),uw3=float(*(thread fp4_e2m1*)(&uc3));
   ag0+=(x0[4*j]*gw0+x0[4*j+1]*gw1+x0[4*j+2]*gw2+x0[4*j+3]*gw3);
   au0+=(x0[4*j]*uw0+x0[4*j+1]*uw1+x0[4*j+2]*uw2+x0[4*j+3]*uw3);
   if(count==2){ag1+=(x1[4*j]*gw0+x1[4*j+1]*gw1+x1[4*j+2]*gw2+x1[4*j+3]*gw3);au1+=(x1[4*j]*uw0+x1[4*j+1]*uw1+x1[4*j+2]*uw2+x1[4*j+3]*uw3);}
  }
  uchar gs=sgate[si],us=sup[si];float sgval=float(*(thread fp8_e4m3*)(&gs)),suval=float(*(thread fp8_e4m3*)(&us));
  {
#pragma clang fp contract(off)
   g0[r]+=sgval*ag0;u0[r]+=suval*au0;g1[r]+=sgval*ag1;u1[r]+=suval*au1;
  }
 }
}
float gg=global_gate[expert]/2688.0f,gu=global_up[expert]/2688.0f;
#pragma unroll
for(int r=0;r<RPS;++r){
 float a=simd_sum(g0[r]),b=simd_sum(u0[r]),c=simd_sum(g1[r]),d=simd_sum(u1[r]);
 if(lane==0){out[size_t(pick)*N+row+r]=native_swiglu(T(a*gg),T(b*gu));if(count==2)out[size_t(partner)*N+row+r]=native_swiglu(T(c*gg),T(d*gu));}
}
'''
_HEADER_EXTRA=r'''
template<typename T> inline T native_swiglu(T gate,T up){
 gate=T(metal::min(float(gate),7.0f));up=T(metal::clamp(float(up),-7.0f,7.0f));
 auto fraction=1/(1+metal::precise::exp(metal::abs(gate)));
 T sigmoid=T(gate<T(0)?fraction:1-fraction);
 return T(T(gate*sigmoid)*up);
}
'''
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_fused_reuse',input_names=['x','wg','sgate','global_gate','wu','sup','global_up','ids','dims','partners','counts'],output_names=['out'],source=_SOURCE,header=_HEADER+_HEADER_EXTRA,compile_options={'math_mode':'safe'})
def fused_reuse(value,g,u,ids,work,rps):
 w,s,gs=g;uw,us,ugs=u;e,n,pk=w.shape;k=pk*2;count,picks=ids.size,ids.shape[-1]
 return kernel()(inputs=[value.reshape(-1,k),w,s,gs,uw,us,ugs,ids.reshape(-1),mx.array([k,n,picks],mx.int32),*work],template=[('T',value.dtype),('RPS',rps)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
