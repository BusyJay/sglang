# SPDX-License-Identifier: Apache-2.0
"""Native-order NVFP4 gathered QMV with bounded two-vector weight reuse.

M4 validation covers FP32/BF16, /16 scales, K divisible by 512 and N by 8.
Global scales use MLX's encoded convention (checkpoint factor times 2688).
No checkpoint conversion, activation quantization or mutable model state.
"""

from functools import cache

import mlx.core as mx

# qdot and fast-QMV arithmetic are derived from MLX 0.32.4 (Apple Inc., MIT).
# Keep native half-intermediate FP4/E4M3 decoding and each block's rounding.
_HEADER = r"""


struct fp4_e2m1 {
  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  uint8_t bits;
};


struct fp8_e4m3 {
  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

"""

_SOURCE = r"""
uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];
if(nmatch==0)return;
uint expert=ids[pick],partner=partners[pick];
const int K=dims[0],N=dims[1],PICKS=dims[2];
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint row=threadgroup_position_in_grid.y*(2*RPS)+sg*RPS;
uint v0=EXPERT?pick:pick/PICKS,v1=EXPERT?partner:partner/PICKS;
if(SINGLES&&nmatch==1){
 float result[RPS]={0};
 for(int k=0;k<K;k+=512){
  float values[16];
  #pragma unroll
  for(int i=0;i<16;++i)values[i]=float(x[size_t(v0)*K+k+lane*16+i]);
  #pragma unroll
  for(int r=0;r<RPS;++r){
   const auto* w=(const device ushort*)(weight+(size_t(expert)*N+row+r)*(K/2)+(k+lane*16)/2);
   uchar s=scales[(size_t(expert)*N+row+r)*(K/16)+(k+lane*16)/16];
   float scale=float(*(thread fp8_e4m3*)(&s)),accum=0;
   #pragma unroll
   for(int j=0;j<4;++j){
    ushort packed=w[j];uchar c0=packed,c1=packed>>4,c2=packed>>8,c3=packed>>12;
    accum+=(values[4*j]*float(*(thread fp4_e2m1*)(&c0))+
            values[4*j+1]*float(*(thread fp4_e2m1*)(&c1))+
            values[4*j+2]*float(*(thread fp4_e2m1*)(&c2))+
            values[4*j+3]*float(*(thread fp4_e2m1*)(&c3)));
   }
   {
#pragma clang fp contract(off)
    result[r]+=scale*accum;
   }
  }
 }
 float global=global_scale[expert]/2688.0f;
 #pragma unroll
 for(int r=0;r<RPS;++r){
  float value=simd_sum(result[r]);
  if(lane==0)out[size_t(pick)*N+row+r]=O(value*global);
 }
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
  {
#pragma clang fp contract(off)
a[r]+=scale*d0;b[r]+=scale*d1;
}
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

_INLINE = r"""
uint pick=threadgroup_position_in_grid.z,expert=ids[pick],prior=0,partner=pick;
for(uint i=0;i<pick;++i)prior+=ids[i]==expert;
if(prior&1)return;
for(uint i=pick+1;i<dims[3];++i)if(ids[i]==expert){partner=i;break;}
uint nmatch=partner==pick?1:2;
"""


@cache
def _kernel(inline):
    source = _SOURCE
    if inline:
        source = source.replace(
            "uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];\n"
            "if(nmatch==0)return;\n"
            "uint expert=ids[pick],partner=partners[pick];",
            _INLINE,
        )
    return mx.fast.metal_kernel(
        name="sglang_nvfp4_two_rows_" + str(int(inline)),
        input_names=[
            "x", "weight", "scales", "global_scale", "ids", "dims",
            "partners", "counts",
        ],
        output_names=["out"],
        source=source,
        header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def mlx_nvfp4_gather_rows(x, weight, scales, global_scale, indices, *, worklist=None):
    """Gather native NVFP4 matrices, sharing each weight read across two picks.

    ``indices`` is uint32 with final dimension top-k. Every index must be in
    [0, num_experts). x holds either one vector per token, broadcast across
    top-k, or one vector per pick. Results retain indices.shape+(1,N).
    A supplied worklist must come from mlx_expert_pair_worklist for these same
    indices. It can be shared across gate/up/down; it is not cached globally.
    Without one, pairing occurs in this dispatch, with no CPU synchronization.
    This specialized QMV contract covers K%512==0 and N%8==0, as native fast
    gather-QMV does. It does not define a changed reduction for other widths.
    """
    if weight.ndim != 3 or weight.dtype != mx.uint8:
        raise ValueError("Expected packed uint8 [experts,N,K/2] weights")
    e, n, pk = weight.shape
    k = pk * 2
    if not e or not n or not k or k % 512 or n % 8:
        raise ValueError("Expected nonempty fast-QMV widths K%512==0, N%8==0")
    if scales.dtype != mx.uint8 or scales.shape != (e, n, k // 16):
        raise ValueError("Expected uint8 E4M3 scales for each 16 weights")
    if global_scale.dtype != mx.float32 or global_scale.size != e:
        raise ValueError("Expected encoded FP32 scale per expert")
    if (
        indices.dtype != mx.uint32
        or indices.ndim < 1
        or not indices.size
        or not indices.shape[-1]
    ):
        raise ValueError("Expected nonempty uint32 expert picks")
    count, picks = indices.size, indices.shape[-1]
    if (
        x.ndim < 1
        or x.dtype not in (mx.float32, mx.bfloat16)
        or x.shape[-1] != k
        or x.size // k not in (count, count // picks)
    ):
        raise ValueError("Expected FP32/BF16 token or per-pick vectors")
    if max(count * k, count * n) >= 2**32 or count >= 2**31:
        raise ValueError("Expected uint32 vector and int32 count indexing")
    if worklist is None:
        partners, counts = indices, indices
    else:
        partners, counts = worklist
        if any(v.dtype != mx.uint32 or v.shape != (count,) for v in worklist):
            raise ValueError("Expected explicit partner/count arrays for every pick")
    return _kernel(worklist is None)(
        inputs=[
            x.reshape(-1, k), weight, scales, global_scale.reshape(-1),
            indices.reshape(-1), mx.array([k, n, picks, count], mx.int32),
            partners, counts,
        ],
        template=[
            ("O", x.dtype), ("RPS", 4), ("EXPERT", x.size // k == count),
            ("SINGLES", True),
        ],
        grid=(32, 2 * ((n + 7) // 8), count),
        threadgroup=(32, 2, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*indices.shape, 1, n)
