# SPDX-License-Identifier: Apache-2.0
"""MPP tensor-view vocabulary projection with an in-kernel rounded top1."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
const uint sg=simdgroup_index_in_threadgroup;
const uint kg=sg/SN,ng=sg%SN;
const uint col=(threadgroup_position_in_grid.x*SN+ng)*BN;
using ext=metal::dextents<int32_t,2>;
using strides=metal::array<int32_t,2>;
using dt=metal::tensor<device T,ext,metal::tensor_inline>;
using tt=metal::tensor<threadgroup float,ext,metal::tensor_inline>;
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,BN,64,false,true,true,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
dt ta((device T*)x,ext(K,M),strides{1,K});
dt tb((device T*)w+size_t(col)*w_strides[0],ext(K,min(uint(BN),uint(N-col))),strides{int(w_strides[1]),int(w_strides[0])});
auto aa=ta.template slice<64,16>(0,0);
auto bb=tb.template slice<64,BN>(0,0);
auto c=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();
for(uint i=0;i<c.get_capacity();++i)c[i]=0;
for(uint it=0;it<(K+64*KS-1)/(64*KS);++it){
 const uint k=(it*KS+kg)*64;
 if(k>=K)continue;
 dt a((device T*)x+k,ext(min(uint(64),uint(K-k)),M),strides{1,K});
 dt b((device T*)w+size_t(col)*w_strides[0]+k*w_strides[1],ext(min(uint(64),uint(K-k)),min(uint(BN),uint(N-col))),strides{int(w_strides[1]),int(w_strides[0])});
 if(w_strides[1]==1){
  op.run(a,b,c);
 }else{
  auto reg=op.template get_right_input_cooperative_tensor<T,T,float>();
  for(uint i=0;i<reg.get_capacity();++i){
   auto pos=reg.get_multidimensional_index(i);
   const uint row=col+pos[1],kk=k+pos[0];
   reg[i]=row<N && kk<K ? w[size_t(row)*w_strides[0]+size_t(kk)*w_strides[1]] : T(0);
  }
  op.run(a,reg,c);
 }
}
threadgroup float partials[KS*SN*16*BN];
tt tmp(partials+(kg*SN+ng)*16*BN,ext(BN,16),strides{1,BN});
c.store(tmp);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(kg!=0)return;
for(uint group=1;group<KS;++group){
 tt v(partials+(group*SN+ng)*16*BN,ext(BN,16),strides{1,BN});
 auto d=c;d.load(v);
 for(uint i=0;i<c.get_capacity();++i)c[i]+=d[i];
}
c.store(tmp);
simdgroup_barrier(mem_flags::mem_threadgroup);
const uint lane=thread_index_in_simdgroup;
for(uint row=0;row<M;++row){
 float best=-INFINITY;uint token=N;bool nan=false;
 for(uint j=lane;j<BN&&col+j<N;j+=32){
  float v=float(T(partials[ng*16*BN+row*BN+j]));
  if((isnan(v)&&!nan)||(!nan&&v>best)||((v==best||(isnan(v)&&nan))&&col+j<token)){
   best=v;token=col+j;nan=isnan(v);
  }
 }
 bool anynan=simd_any(nan);
 float maxv=simd_max(best);
 uint id=simd_min((anynan?nan:best==maxv)?token:uint(N));
 if(lane==0){scores[size_t(row)*BLOCKS+threadgroup_position_in_grid.x*SN+ng]=T(anynan?NAN:maxv);ids[size_t(row)*BLOCKS+threadgroup_position_in_grid.x*SN+ng]=id;}
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_dense_argmax_nax",
        input_names=["x", "w"],
        output_names=["scores", "ids"],
        source=_SOURCE,
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        ensure_row_contiguous=False,
    )


def _mlx_dense_argmax_nax(x, weight):
    """Project up to sixteen rows without writing the vocabulary logits."""
    count, k = x.shape[1:]
    n = weight.shape[0]
    bn, sn, ks = 16, 1, 16
    blocks = (n + bn - 1) // bn
    scores, ids = _kernel()(
        inputs=[mx.contiguous(x), weight],
        template=[
            ("T", x.dtype),
            ("M", count),
            ("N", n),
            ("K", k),
            ("KS", ks),
            ("SN", sn),
            ("BN", bn),
            ("BLOCKS", blocks),
        ],
        grid=(32 * sn * ks * ((blocks + sn - 1) // sn), 1, 1),
        threadgroup=(32 * sn * ks, 1, 1),
        output_shapes=[(count, blocks)] * 2,
        output_dtypes=[x.dtype, mx.uint32],
    )
    return mx.take_along_axis(ids, mx.argmax(scores, -1)[:, None], 1).reshape(1, count)
