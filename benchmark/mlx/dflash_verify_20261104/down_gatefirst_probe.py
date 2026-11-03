"""Experimental NVFP4 down projection with explicit ordered routing combine."""
import inspect
import textwrap
import mlx.core as mx
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _HEADER
from sglang.srt.hardware_backend.mlx.models import glm5_next as g

_SOURCE = r"""
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint pick=sg%TOPK,tok=threadgroup_position_in_grid.y*TILE+sg/TOPK;
const uint row=threadgroup_position_in_grid.x*RPS;
const int K=dims[0],N=dims[1],COUNT=dims[2];
float acc[RPS]={0};
if(tok<COUNT){
 uint expert=ids[tok*TOPK+pick];
 const device T* xv=act+(size_t(tok)*TOPK+pick)*K;
 for(int k=0;k<K;k+=512){
  float values[16];
  #pragma unroll
  for(int j=0;j<16;++j)values[j]=float(xv[k+lane*16+j]);
  #pragma unroll
  for(int r=0;r<RPS;++r){
   const device ushort* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
   uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
   float scale=float(*(thread fp8_e4m3*)(&s)),dot=0;
   #pragma unroll
   for(int j=0;j<4;++j){
    ushort v=w[j];uchar c0=v,c1=v>>4,c2=v>>8,c3=v>>12;
    dot+=(values[4*j]*float(*(thread fp4_e2m1*)(&c0))+values[4*j+1]*float(*(thread fp4_e2m1*)(&c1))+values[4*j+2]*float(*(thread fp4_e2m1*)(&c2))+values[4*j+3]*float(*(thread fp4_e2m1*)(&c3)));
   }
   {
#pragma clang fp contract(off)
    acc[r]+=scale*dot;
   }
  }
 }
}
threadgroup float terms[TOPK*TILE*RPS];
#pragma unroll
for(int r=0;r<RPS;++r){
 float sum=simd_sum(acc[r]);
 if(lane==0&&tok<COUNT){
  uint expert=ids[tok*TOPK+pick];
  float v=float(T(sum*(global_scale[expert]/2688.0f)));
  volatile float product=v*scores[tok*TOPK+pick];
  terms[(sg/TOPK*TOPK+pick)*RPS+r]=product;
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(pick==0&&lane<RPS&&tok<COUNT){
 float total=0;
 for(uint j=0;j<TOPK;++j)total+=terms[(sg/TOPK*TOPK+j)*RPS+lane];
 out[size_t(tok)*N+row+lane]=R(total)+shared[size_t(tok)*N+row+lane];
}
"""
kernel=mx.fast.metal_kernel(name='sglang_nvfp4_down_combine_probe',input_names=['act','weight','scales','global_scale','ids','scores','shared','dims'],output_names=['out'],source=_SOURCE,header=_HEADER)

def fused_down(act,ids,scores,p,shared,rps,tile):
 k,n=p.weight.shape[-1]*2,p.weight.shape[-2];count=ids.shape[1];topk=ids.shape[-1]
 return kernel(inputs=[act.reshape(-1,k),p.weight,p.scales,p.weight_scale_2*2688,ids.reshape(-1).astype(mx.uint32),scores.reshape(-1).astype(mx.float32),shared,mx.array([k,n,count],mx.int32)],template=[('T',act.dtype),('R',shared.dtype),('TOPK',topk),('RPS',rps),('TILE',tile)],grid=(32*(n//rps),topk*tile*((count+tile-1)//tile),1),threadgroup=(32,topk*tile,1),output_shapes=[shared.shape],output_dtypes=[shared.dtype])[0]

src=textwrap.dedent(inspect.getsource(g.MoE.__call__))
old='routed = self.switch_mlp(x, indices)\n    routed = (routed * weights[..., None]).sum(-2).astype(x.dtype)\n    return routed + shared'
new='sw=self.switch_mlp;expanded=x[...,None,None,:]\n    gate=sw.gate_proj(expanded,indices);up=sw.up_proj(expanded,indices)\n    act=sw.activation(up,gate)\n    return fused_down(act,indices,weights,sw.down_proj,shared,self.probe_rps,self.probe_tile)'
assert old in src
namespace=dict(g.__dict__);namespace['fused_down']=fused_down;exec(src.replace(old,new),namespace)

class DownCombineMoE(g.MoE):
 __call__=namespace['__call__']
