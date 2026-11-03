"""Diagnostic multi-pick weight reuse, preserving native per-vector qdot order."""
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _HEADER
from sglang.srt.hardware_backend.mlx.models import glm5_next as g

work=mx.fast.metal_kernel(name='sglang_nvfp4_multi_work',input_names=['ids','dims'],output_names=['members','counts'],source=r'''
uint pick=thread_position_in_grid.x,count=dims[0];if(pick>=count)return;
uint expert=ids[pick],prior=0;
for(uint j=0;j<pick;++j)prior+=ids[j]==expert;
if(prior%REUSE){counts[pick]=0;for(uint j=0;j<REUSE;++j)members[pick*REUSE+j]=pick;return;}
uint n=0;
for(uint j=pick;j<count&&n<REUSE;++j)if(ids[j]==expert)members[pick*REUSE+n++]=j;
counts[pick]=n;for(uint j=n;j<REUSE;++j)members[pick*REUSE+j]=pick;
''')
kernel=mx.fast.metal_kernel(name='sglang_nvfp4_multi_reuse',input_names=['act','weight','scales','global_scale','ids','members','counts','dims'],output_names=['out'],header=_HEADER,source=r'''
uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];if(!nmatch)return;
uint expert=ids[pick],lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const int K=dims[0],N=dims[1],TOPK=dims[2];
uint row=threadgroup_position_in_grid.y*2*RPS+sg*RPS;
float accum[REUSE][RPS]={0};
for(int k=0;k<K;k+=512){
 float values[REUSE][16];
 #pragma unroll
 for(uint v=0;v<REUSE;++v){
  if(v<nmatch){
   uint p=members[pick*REUSE+v],index=EXPERT?p:p/TOPK;
   #pragma unroll
   for(int j=0;j<16;++j)values[v][j]=float(act[size_t(index)*K+k+lane*16+j]);
  }
 }
 #pragma unroll
 for(uint r=0;r<RPS;++r){
  const device ushort* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
  uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
  float scale=float(*(thread fp8_e4m3*)(&s)),dots[REUSE]={0};
  #pragma unroll
  for(int j=0;j<4;++j){
   ushort bits=w[j];uchar c0=bits,c1=bits>>4,c2=bits>>8,c3=bits>>12;
   float w0=float(*(thread fp4_e2m1*)(&c0)),w1=float(*(thread fp4_e2m1*)(&c1)),w2=float(*(thread fp4_e2m1*)(&c2)),w3=float(*(thread fp4_e2m1*)(&c3));
   #pragma unroll
   for(uint v=0;v<REUSE;++v)if(v<nmatch)dots[v]+=(values[v][4*j]*w0+values[v][4*j+1]*w1+values[v][4*j+2]*w2+values[v][4*j+3]*w3);
  }
  #pragma unroll
  for(uint v=0;v<REUSE;++v)if(v<nmatch){
#pragma clang fp contract(off)
   accum[v][r]+=scale*dots[v];
  }
 }
}
float global=global_scale[expert]/2688.0f;
#pragma unroll
for(uint v=0;v<REUSE;++v){
 #pragma unroll
 for(uint r=0;r<RPS;++r){
  float sum=simd_sum(accum[v][r]);
  if(lane==0&&v<nmatch)out[size_t(members[pick*REUSE+v])*N+row+r]=T(sum*global);
 }
}
''')

class MultiReuseSwitch(g.ClampedSwitchGLU):
 def __call__(self,x,indices):
  ids=indices.astype(mx.uint32);count=ids.size;reuse=self.probe_reuse;rps=self.probe_rps
  members,counts=work(inputs=[ids.reshape(-1),mx.array([count],mx.uint32)],template=[('REUSE',reuse)],grid=(count,1,1),threadgroup=(32,1,1),output_shapes=[(count*reuse,),(count,)],output_dtypes=[mx.uint32,mx.uint32])
  def project(value,p):
   k,n=p.weight.shape[-1]*2,p.weight.shape[-2]
   return kernel(inputs=[value.reshape(-1,k),p.weight,p.scales,p.weight_scale_2*2688,ids.reshape(-1),members,counts,mx.array([k,n,ids.shape[-1]],mx.uint32)],template=[('T',value.dtype),('REUSE',reuse),('RPS',rps),('EXPERT',value.size//k==count)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n)],output_dtypes=[value.dtype])[0].reshape(*ids.shape,1,n)
  expanded=x[...,None,None,:]
  gate=project(expanded,self.gate_proj);up=project(expanded,self.up_proj)
  return project(self.activation(up,gate),self.down_proj).squeeze(-2)
