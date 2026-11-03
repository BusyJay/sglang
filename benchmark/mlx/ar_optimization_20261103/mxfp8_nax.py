from functools import cache
import mlx.core as mx
_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline float e4m3(uchar code) {
    uint mag = uint(code) & 127;
    if (mag == 127) return NAN;
    uint exp = mag >> 3, mant = mag & 7;
    float v = exp == 0 ? float(mant) * 0.001953125f :
        as_type<float>(((exp + 120) << 23) | (mant << 20));
    return code & 128 ? -v : v;
}
// Installed SDK's dual-cooperative run asserts K == 16 || K == 32.
constant constexpr auto diagnostic_desc=mpp::tensor_ops::matmul2d_descriptor(
    16,32,32,false,true,false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
using diagnostic_op=mpp::tensor_ops::matmul2d<diagnostic_desc,metal::execution_simdgroup>;
using diagnostic_left=diagnostic_op::cooperative_tensor_left_input_t<bfloat,bfloat,float>;
using diagnostic_right=diagnostic_op::cooperative_tensor_right_input_t<bfloat,bfloat,float>;
"""

_SOURCE = r"""
const uint K=dims[0], N=dims[1], M=dims[2];
const uint kg=simdgroup_index_in_threadgroup;
const uint col=threadgroup_position_in_grid.x*32;
using ext=metal::dextents<int32_t,2>;
using strides=metal::array<int32_t,2>;
using tt=metal::tensor<threadgroup float,ext,metal::tensor_inline>;
diagnostic_op op;
auto a=op.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();
auto b=op.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
auto total=op.template get_destination_cooperative_tensor<diagnostic_left,diagnostic_right,float>();
for(uint i=0;i<total.get_capacity();++i)total[i]=0;
for(uint block=kg;block<K/32;block+=SPLIT){
    auto partial=total;
    for(uint i=0;i<partial.get_capacity();++i)partial[i]=0;
    const uint k=block*32;
    for(uint i=0;i<a.get_capacity();++i){
        auto p=a.get_multidimensional_index(i);uint kk=k+p[0],row=p[1];
        a[i]=row<M ? bfloat(x[size_t(row)*K+kk]) : bfloat(0);
    }
    for(uint i=0;i<b.get_capacity();++i){
        auto p=b.get_multidimensional_index(i);uint kk=k+p[0],row=col+p[1];
        b[i]=row<N ? bfloat(e4m3(w[size_t(row)*K+kk])*as_type<float>(uint(scale[size_t(row)*(K/32)+block])<<23)) : bfloat(0);
    }
    op.run(a,b,total);
}
threadgroup float scratch[SPLIT*16*32];
tt tmp(scratch+kg*16*32,ext(32,16),strides{1,32});
total.store(tmp);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(kg!=0)return;
for(uint group=1;group<SPLIT;++group){
    tt v(scratch+group*16*32,ext(32,16),strides{1,32});
    auto other=total;other.load(v);
    for(uint i=0;i<total.get_capacity();++i)total[i]+=other[i];
}
total.store(tmp);
simdgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=thread_index_in_simdgroup;i<M*32;i+=32){
    uint row=i/32, j=i%32;
    if(col+j<N)out[size_t(row)*N+col+j]=O(scratch[i]);
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="diagnostic_mxfp8_narrow_nax",
        input_names=["w", "scale", "x", "dims"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


def fn(x,w,scale,split=4):
 n,k=w.shape;count=x.size//k
 assert x.dtype==mx.bfloat16 and k%32==0 and count<=16
 return _kernel()(inputs=[w,scale,x.reshape(count,k),mx.array([k,n,count],mx.int32)],template=[('O',x.dtype),('SPLIT',split)],grid=(32*split*((n+31)//32),1,1),threadgroup=(32*split,1,1),output_shapes=[(count,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
