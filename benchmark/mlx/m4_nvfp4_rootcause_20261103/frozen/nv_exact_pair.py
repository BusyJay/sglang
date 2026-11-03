"""Diagnostic: exact native FP4 qdot, gate/up pairing with separate operands."""
from functools import cache
import mlx.core as mx
_HEADER='\n\nconstant constexpr float F8E4M3_MAX = 448.0f;\nconstant constexpr float F4E2M1_MAX = 6.0f;\n\nstruct fp4_e2m1 {\n  fp4_e2m1(float x) thread {\n    if (metal::isnan(x)) {\n      bits = 0x7;\n      return;\n    }\n\n    const uint8_t sign_bit = (metal::signbit(x)) ? 0x8 : 0x0;\n    x = metal::abs(x);\n\n    if (x > 5.0f) {\n      bits = 0x7;\n    } else if (x >= 3.5f) {\n      bits = 0x6;\n    } else if (x > 2.5f) {\n      bits = 0x5;\n    } else if (x >= 1.75f) {\n      bits = 0x4;\n    } else if (x > 1.25f) {\n      bits = 0x3;\n    } else if (x >= 0.75f) {\n      bits = 0x2;\n    } else if (x > 0.25f) {\n      bits = 0x1;\n    } else {\n      bits = 0x0;\n    }\n    bits |= sign_bit;\n  }\n\n  operator float16_t() thread {\n    half converted = as_type<half>(ushort((bits & 7) << 9));\n    converted *= 16384.0;\n    return bits & 8 ? -converted : converted;\n  }\n\n  operator float() thread {\n    return static_cast<float>(this->operator float16_t());\n  }\n\n  operator bfloat16_t() thread {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }\n\n  uint8_t bits;\n};\n\n\nstruct fp8_e4m3 {\n  template <typename T>\n  fp8_e4m3(T f) thread {\n    // From PyTorch\n    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148\n    uint32_t fp8_max = 543 << 21;\n    uint32_t denorm_mask = 141 << 23;\n    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));\n    uint32_t sign = f_bits & 0x80000000;\n    f_bits ^= sign;\n    if (f_bits >= fp8_max) {\n      // Default behavior saturates to min/max\n      bits = 0x7E;\n    } else {\n      if (f_bits < (121 << 23)) {\n        f_bits = as_type<uint32_t>(\n            as_type<float>(f_bits) + as_type<float>(denorm_mask));\n        bits = static_cast<uint8_t>(f_bits - denorm_mask);\n      } else {\n        // resulting mantissa is odd\n        uint8_t mant_odd = (f_bits >> 20) & 1;\n        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;\n        f_bits += mant_odd;\n        bits = static_cast<uint8_t>(f_bits >> 20);\n      }\n    }\n    bits |= static_cast<uint8_t>(sign >> 24);\n  }\n\n  operator float16_t() thread {\n    uint16_t v = bits & 127;\n    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;\n    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;\n    half converted = as_type<half>(u);\n    return converted * 256.0;\n  }\n\n  operator bfloat16_t() thread {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }\n\n  operator float() thread {\n    return static_cast<float>(this->operator float16_t());\n  }\n\n  uint8_t bits;\n};\n\n'
_SOURCE=r'''
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const int K=dims[0],N=dims[1],PICKS=dims[2];
uint pick=threadgroup_position_in_grid.z;
uint expert=ids[pick],token=pick/PICKS;
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
  rg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;
  ru[r]+=float(*(thread fp8_e4m3*)(&us))*au;
 }
}
float gg=global_gate[expert]/2688.0f,gu=global_up[expert]/2688.0f;
#pragma unroll
for(int r=0;r<RPS;++r){
 float g=simd_sum(rg[r]),u=simd_sum(ru[r]);
 if(lane==0&&row+r<N){gate[pick*N+row+r]=T(g*gg);up[pick*N+row+r]=T(u*gu);}
}
'''
@cache
def kernel():
 return mx.fast.metal_kernel(name='diagnostic_nvfp4_exact_pair',input_names=['x','wg','sgate','global_gate','wu','sup','global_up','ids','dims'],output_names=['gate','up'],source=_SOURCE,header=_HEADER)
def pair(x,g,u,ids,rps=4):
 e,n,pk=g.weight.shape;k=pk*2;count=ids.size
 vals=kernel()(inputs=[x.reshape(-1,k),g.weight,g.scales,(g.weight_scale_2*2688).reshape(-1),u.weight,u.scales,(u.weight_scale_2*2688).reshape(-1),ids.reshape(-1).astype(mx.uint32),mx.array([k,n,ids.shape[-1]],mx.int32)],template=[('T',x.dtype),('RPS',rps)],grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),output_shapes=[(count,n),(count,n)],output_dtypes=[x.dtype,x.dtype])
 return tuple(v.reshape(*ids.shape,1,n) for v in vals)
