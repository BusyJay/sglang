"""Bounded two-token expert groups avoid runtime accumulator arrays."""

from functools import cache

import mlx.core as mx

from common import exact
from native_clone import header

_LIST = r"""
uint pick=thread_position_in_grid.x,count=dims[0];
if(pick>=count)return;
uint expert=ids[pick],prior=0,next=pick;
for(uint i=0;i<pick;++i)prior+=ids[i]==expert;
if(prior&1){counts[pick]=0;partners[pick]=pick;return;}
for(uint i=pick+1;i<count;++i)if(ids[i]==expert){next=i;break;}
partners[pick]=next;counts[pick]=next==pick?1:2;
"""


@cache
def list_kernel():
    return mx.fast.metal_kernel(
        name="nvfp4_expert_pairs",input_names=["ids","dims"],
        output_names=["partners","counts"],source=_LIST,
    )


def worklist(ids):
    return list_kernel()(
        inputs=[ids.reshape(-1),mx.array([ids.size],mx.uint32)],
        grid=(ids.size,1,1),threadgroup=(32,1,1),
        output_shapes=[(ids.size,),(ids.size,)],output_dtypes=[mx.uint32,mx.uint32],
    )


_SOURCE=r"""
uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];
if(nmatch==0)return;
uint expert=ids[pick],partner=partners[pick];
const int K=dims[0],N=dims[1],PICKS=dims[2];
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
uint v0=EXPERT?pick:pick/PICKS,v1=EXPERT?partner:partner/PICKS;
if(SINGLES&&nmatch==1){
 fp_qmv_fast_impl<O,16,4,true,true,RPS>(
  (const device uint32_t*)(weight+size_t(expert)*N*(K/2)),
  scales+size_t(expert)*N*(K/16),global_scale+expert,
  x+size_t(v0)*K,out+size_t(pick)*N,
  dims[0],dims[1],uint3(0,threadgroup_position_in_grid.y,0),sg,lane);
 return;
}
float a[RPS]={0},b[RPS]={0};
for(int k=0;k<K;k+=512){
 float x0[16],x1[16];
 #pragma unroll
 for(int i=0;i<16;++i){
  x0[i]=float(x[size_t(v0)*K+k+lane*16+i]);
  x1[i]=float(x[size_t(v1)*K+k+lane*16+i]);
 }
 #pragma unroll
 for(int r=0;r<RPS;++r){
  if(row+r>=N)continue;
  const device ushort* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
  uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
  float scale=float(*(thread fp8_e4m3*)(&s)),d0=0,d1=0;
  #pragma unroll
  for(int j=0;j<4;++j){
   ushort packed=w[j];uchar c0=packed,c1=packed>>4,c2=packed>>8,c3=packed>>12;
   float w0=float(*(thread fp4_e2m1*)(&c0)),w1=float(*(thread fp4_e2m1*)(&c1));
   float w2=float(*(thread fp4_e2m1*)(&c2)),w3=float(*(thread fp4_e2m1*)(&c3));
   d0+=(x0[4*j]*w0+x0[4*j+1]*w1+x0[4*j+2]*w2+x0[4*j+3]*w3);
   d1+=(x1[4*j]*w0+x1[4*j+1]*w1+x1[4*j+2]*w2+x1[4*j+3]*w3);
  }
  volatile float scaled0=scale*d0,scaled1=scale*d1;
  a[r]+=scaled0;b[r]+=scaled1;
 }
}
float global=global_scale[expert]/2688.0f;
#pragma unroll
for(int r=0;r<RPS;++r){
 float y0=simd_sum(a[r]),y1=simd_sum(b[r]);
 if(lane==0&&row+r<N){
  out[size_t(pick)*N+row+r]=O(y0*global);
  if(nmatch==2)out[size_t(partner)*N+row+r]=O(y1*global);
 }
}
"""


@cache
def kernel(rounding="volatile", inline_list=False):
    source=_SOURCE
    if rounding=="scoped":
        source=source.replace("volatile float scaled0=scale*d0,scaled1=scale*d1;\n  a[r]+=scaled0;b[r]+=scaled1;", "{\n#pragma clang fp contract(off)\na[r]+=scale*d0;b[r]+=scale*d1;\n}")
    if inline_list:
        source=source.replace("uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];\nif(nmatch==0)return;\nuint expert=ids[pick],partner=partners[pick];",r"""
uint pick=threadgroup_position_in_grid.z,expert=ids[pick],prior=0,partner=pick;
for(uint i=0;i<pick;++i)prior+=ids[i]==expert;
if(prior&1)return;
for(uint i=pick+1;i<dims[3];++i)if(ids[i]==expert){partner=i;break;}
uint nmatch=partner==pick?1:2;
""")
    return mx.fast.metal_kernel(
        name="nvfp4_two_rows_"+rounding+str(int(inline_list)),input_names=["x","weight","scales","global_scale","ids","dims","partners","counts"],
        output_names=["out"],source=source,header=header,
        compile_options={"math_mode":"safe"},
    )


def project(x,p,ids,rps=4,work=None,singles=True,rounding="volatile",inline_list=False):
    _,n,pk=p.weight.shape;k=pk*2
    partners,counts=(ids,ids) if inline_list else (worklist(ids) if work is None else work)
    return kernel(rounding,inline_list)(
        inputs=[x.reshape(-1,k),p.weight,p.scales,p.weight_scale_2*2688,ids.reshape(-1),
                mx.array([k,n,ids.shape[-1],ids.size],mx.int32),partners,counts],
        template=[("O",x.dtype),("RPS",rps),("EXPERT",x.size//k==ids.size),("SINGLES",singles)],
        grid=(32,2*((n+2*rps-1)//(2*rps)),ids.size),threadgroup=(32,2,1),
        output_shapes=[(ids.size,n)],output_dtypes=[x.dtype],
    )[0].reshape(*ids.shape,1,n)
