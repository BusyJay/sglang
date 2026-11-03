# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 gate/up projection with a rounded SwiGLU epilogue.

Shares activation loads across two independent packed expert matrices. No
weight concatenation or extra activation quantization. Gate/up round to the
input dtype before clamping, native sigmoid and the two activation products.
"""

from functools import cache

import mlx.core as mx

from .mlx_nvfp4_gather import _HEADER

_SOURCE = r"""
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const int K=dims[0],N=dims[1],PICKS=dims[2];
uint pick=threadgroup_position_in_grid.z;
uint expert=ids[pick],token=EXPERT?pick:pick/PICKS;
uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
const device T* xv=x+size_t(token)*K;
float rg[RPS]={0},ru[RPS]={0};
for(int k=0;k<K;k+=512){
 float values[16];
 #pragma unroll
 for(int i=0;i<16;++i)values[i]=float(xv[k+lane*16+i]);
 #pragma unroll
 for(int r=0;r<RPS;++r){
  if(row+r>=N)continue;
  size_t wi=(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2;
  size_t si=(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16;
  const device ushort* wgp=(const device ushort*)(wg+wi);
  const device ushort* wup=(const device ushort*)(wu+wi);
  float ag=0,au=0;
  #pragma unroll
  for(int i=0;i<4;++i){
   ushort g=wgp[i],u=wup[i];
   uchar g0=g,g1=g>>4,g2=g>>8,g3=g>>12;
   uchar u0=u,u1=u>>4,u2=u>>8,u3=u>>12;
   ag+=(values[4*i]*float(*(thread fp4_e2m1*)(&g0))+values[4*i+1]*float(*(thread fp4_e2m1*)(&g1))+values[4*i+2]*float(*(thread fp4_e2m1*)(&g2))+values[4*i+3]*float(*(thread fp4_e2m1*)(&g3)));
   au+=(values[4*i]*float(*(thread fp4_e2m1*)(&u0))+values[4*i+1]*float(*(thread fp4_e2m1*)(&u1))+values[4*i+2]*float(*(thread fp4_e2m1*)(&u2))+values[4*i+3]*float(*(thread fp4_e2m1*)(&u3)));
  }
  uchar gs=sgate[si],us=sup[si];
  {
#pragma clang fp contract(off)
  rg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;
  ru[r]+=float(*(thread fp8_e4m3*)(&us))*au;
  }
 }
}
float gg=global_gate[expert]/2688.0f,gu=global_up[expert]/2688.0f;
#pragma unroll
for(int r=0;r<RPS;++r){
 float g=simd_sum(rg[r]),u=simd_sum(ru[r]);
 if(lane==0&&row+r<N){
T gate=T(g*gg),up=T(u*gu);
if(CLAMP){
 float bound=float(T(limit[0]));
 gate=T(metal::min(float(gate),bound));
 up=T(metal::clamp(float(up),-bound,bound));
}
// Native Sigmoid's BF16 intermediate overloads are deliberate.
auto fraction=1/(1+metal::precise::exp(metal::abs(gate)));
T sigmoid=T(gate<T(0)?fraction:1-fraction);
T activated=T(gate*sigmoid);
out[size_t(pick)*N+row+r]=T(activated*up);
}
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_mlx_nvfp4_swiglu",
        input_names=[
            "x",
            "wg",
            "sgate",
            "global_gate",
            "wu",
            "sup",
            "global_up",
            "ids",
            "dims",
            "limit",
        ],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def mlx_nvfp4_swiglu(
    x,
    gate_weight,
    gate_scales,
    gate_global_scale,
    up_weight,
    up_scales,
    up_global_scale,
    indices,
    *,
    limit=None,
):
    """Project gathered native NVFP4 gate/up matrices and apply clamped SwiGLU.

    Packed weights are uint8 [experts,N,K/2], scales are /16 uint8 E4M3, and
    global factors use the MLX encoded convention (checkpoint factor *2688).
    K must be divisible by 512 and N by 8 for the native-order fast-QMV contract.
    x contains one vector per token or per pick; output is indices.shape+(1,N).
    indices must be uint32 and in range. No CPU synchronization is introduced.
    Gate/up outputs round separately to x.dtype, including BF16 sigmoid and
    intermediate products. The default model integration uses this for AR;
    multi-token expert weight sharing remains a separate mechanism.
    """
    if (
        gate_weight.ndim != 3
        or gate_weight.dtype != mx.uint8
        or up_weight.dtype != mx.uint8
        or up_weight.shape != gate_weight.shape
    ):
        raise ValueError("Expected equal packed uint8 [experts,N,K/2] banks")
    experts, n, packed_k = gate_weight.shape
    k = packed_k * 2
    if not experts or not n or not k or k % 512 or n % 8:
        raise ValueError("Expected nonempty fast-QMV widths K%512==0, N%8==0")
    if any(
        s.dtype != mx.uint8 or s.shape != (experts, n, k // 16)
        for s in (gate_scales, up_scales)
    ):
        raise ValueError("Expected /16 uint8 E4M3 scales")
    if any(
        s.dtype != mx.float32 or s.size != experts
        for s in (gate_global_scale, up_global_scale)
    ):
        raise ValueError("Expected encoded FP32 global scale per expert")
    if indices.dtype != mx.uint32 or indices.ndim < 1 or not indices.size:
        raise ValueError("Expected nonempty uint32 expert picks")
    count, picks = indices.size, indices.shape[-1]
    if (
        x.ndim < 1
        or x.dtype not in (mx.float32, mx.bfloat16)
        or x.shape[-1] != k
        or x.size // k not in (count, count // picks)
    ):
        raise ValueError("Expected FP32/BF16 token or per-pick activation vectors")
    if max(count * k, count * n) >= 2**32 or count >= 2**31:
        raise ValueError("Expected uint32 vector and int32 count indexing")
    return _kernel()(
        inputs=[
            x.reshape(-1, k),
            gate_weight,
            gate_scales,
            gate_global_scale.reshape(-1),
            up_weight,
            up_scales,
            up_global_scale.reshape(-1),
            indices.reshape(-1),
            mx.array([k, n, picks], mx.int32),
            mx.array([limit or 0], mx.float32),
        ],
        template=[
            ("T", x.dtype),
            ("RPS", 1),
            ("EXPERT", x.size // k == count),
            ("CLAMP", limit is not None),
        ],
        grid=(32, n, count),
        threadgroup=(32, 2, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*indices.shape, 1, n)
