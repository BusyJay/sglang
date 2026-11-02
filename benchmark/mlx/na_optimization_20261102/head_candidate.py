# SPDX-License-Identifier: Apache-2.0
"""MPP tensor-view vocabulary projection with an in-kernel rounded top1."""
from functools import cache
import mlx.core as mx

_SOURCE = "\nconst uint sg=simdgroup_index_in_threadgroup;\nconst uint kg=sg/SN,ng=sg%SN;\nconst uint col=(threadgroup_position_in_grid.x*SN+ng)*BN;\nusing ext=metal::dextents<int32_t,2>;\nusing strides=metal::array<int32_t,2>;\nusing dt=metal::tensor<device T,ext,metal::tensor_inline>;\nusing tt=metal::tensor<threadgroup float,ext,metal::tensor_inline>;\nconstexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,BN,64,false,true,true,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\nmpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;\ndt ta((device T*)x,ext(K,M),strides{1,K});\ndt tb((device T*)w+size_t(col)*K,ext(K,min(uint(BN),uint(N-col))),strides{1,K});\nauto aa=ta.template slice<64,16>(0,0);\nauto bb=tb.template slice<64,BN>(0,0);\nauto c=op.template get_destination_cooperative_tensor<decltype(aa),decltype(bb),float>();\nfor(uint i=0;i<c.get_capacity();++i)c[i]=0;\nfor(uint it=0;it<(K+64*KS-1)/(64*KS);++it){\n const uint k=(it*KS+kg)*64;\n if(k>=K)continue;\n dt a((device T*)x+k,ext(min(uint(64),uint(K-k)),M),strides{1,K});\n dt b((device T*)w+size_t(col)*K+k,ext(min(uint(64),uint(K-k)),min(uint(BN),uint(N-col))),strides{1,K});\n op.run(a,b,c);\n}\nthreadgroup float partials[KS*SN*16*BN];\ntt tmp(partials+(kg*SN+ng)*16*BN,ext(BN,16),strides{1,BN});\nc.store(tmp);\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nif(kg!=0)return;\nfor(uint group=1;group<KS;++group){\n tt v(partials+(group*SN+ng)*16*BN,ext(BN,16),strides{1,BN});\n auto d=c;d.load(v);\n for(uint i=0;i<c.get_capacity();++i)c[i]+=d[i];\n}\nc.store(tmp);\nsimdgroup_barrier(mem_flags::mem_threadgroup);\nconst uint lane=thread_index_in_simdgroup;\nfor(uint row=0;row<M;++row){\n float best=-INFINITY;uint token=N;bool nan=false;\n for(uint j=lane;j<BN&&col+j<N;j+=32){\n  float v=float(T(partials[ng*16*BN+row*BN+j]));\n  if((isnan(v)&&!nan)||(!nan&&v>best)||((v==best||(isnan(v)&&nan))&&col+j<token)){\n   best=v;token=col+j;nan=isnan(v);\n  }\n }\n bool anynan=simd_any(nan);\n float maxv=simd_max(best);\n uint id=simd_min((anynan?nan:best==maxv)?token:uint(N));\n if(lane==0){scores[size_t(row)*BLOCKS+threadgroup_position_in_grid.x*SN+ng]=T(anynan?NAN:maxv);ids[size_t(row)*BLOCKS+threadgroup_position_in_grid.x*SN+ng]=id;}\n}\n"

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

def mlx_dense_argmax_nax(x, weight):
    """Project up to sixteen rows without writing the vocabulary logits."""
    count, k = x.shape[1:]
    n = weight.shape[0]
    bn, sn, ks = 16, 1, 16
    blocks = (n + bn - 1) // bn
    scores, ids = _kernel()(
        inputs=[mx.contiguous(x), mx.contiguous(weight)],
        template=[("T", x.dtype), ("M", count), ("N", n), ("K", k),
                  ("KS", ks), ("SN", sn), ("BN", bn), ("BLOCKS", blocks)],
        grid=(32 * sn * ks * ((blocks + sn - 1) // sn), 1, 1),
        threadgroup=(32 * sn * ks, 1, 1),
        output_shapes=[(count, blocks)] * 2,
        output_dtypes=[x.dtype, mx.uint32],
    )
    return mx.take_along_axis(ids, mx.argmax(scores, -1)[:, None], 1).reshape(1, count)
