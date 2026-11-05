# SPDX-License-Identifier: Apache-2.0
"""Grouped NVFP4 prefill with native scaled E4M3 activation tensor operands."""

from functools import cache

import mlx.core as mx

from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _HEADER

_GROUPS = r"""
uint expert = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint count = 0;
for (uint start = 0; start < dims[0]; start += 32) {
    uint i = start + lane;
    uint match = i < dims[0] && ids[i] == expert;
    uint rank = simd_prefix_exclusive_sum(match);
    if (match) owners[expert * dims[0] + count + rank] = i;
    count += simd_sum(match);
}
if (!lane) counts[expert] = count;
"""
_SOURCE = r"""
const uint expert = threadgroup_position_in_grid.y;
const uint count = counts[expert];
if (count<=4) return;
const uint K = dims[0], N = dims[1], PICKS = dims[2], TOTAL = dims[3];
const uint sg = simdgroup_index_in_threadgroup;
const uint lid = thread_position_in_threadgroup.x;
const uint n0 = threadgroup_position_in_grid.x * 128;
threadgroup uchar values[8*32], block_scales[8];
threadgroup half weight_tile[128*32];
using scales_t = tensor_blockwise<tensor_plane_scales,
    threadgroup metal_fp8_ue8m0_format,32,1>;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    128,8,32,false,true,false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroups<4>> op;
tensor<threadgroup half,dextents<int,2>,tensor_inline> a(weight_tile,dextents<int,2>(32,128));
tensor<threadgroup metal_fp8_e4m3_format,dextents<int,2>,tensor_inline,scales_t> b(
    values,dextents<int,2>(32,8),scales_t(block_scales));
auto c = op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>,decltype(b),float>();
for(uint start = 0; start < count; start += 8) {
    for(uint i = 0; i < c.get_capacity(); ++i)c[i] = 0.0f;
    for(uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = lid; i < 8*32; i += 128) {
            uint row = start + i / 32;
            uint pick = row < count ? owners[expert*TOTAL+row] : 0;
            uint input_row = PER_PICK ? pick : pick / PICKS;
            values[i] = row < count ? xq[size_t(input_row)*K+k+i%32] : 0;
        }
        if(lid < 8) {
            uint row = start + lid;
            uint pick = row < count ? owners[expert*TOTAL+row] : 0;
            uint input_row = PER_PICK ? pick : pick / PICKS;
            block_scales[lid] = row < count ? xscale[input_row*(K/32)+k/32] : 127;
        }
        for(uint i = lid; i < 128*8; i += 128) {
            uint n = n0 + i/8, col = k + (i%8)*4;
            const uint dest=(i/8)*32+(i%8)*4;
            if(n<N) {
                const ushort codes=*((const device ushort*)(
                    weight+(size_t(expert)*N+n)*(K/2)+col/2));
                uchar sb=scales[(size_t(expert)*N+n)*(K/16)+col/16];
                float scale=float(*(thread fp8_e4m3*)(&sb));
                for(uint j=0;j<4;++j) {
                    uchar code=uchar((codes>>(j*4))&15);
                    weight_tile[dest+j]=half(float(*(thread fp4_e2m1*)(&code))*scale);
                }
            } else for(uint j=0;j<4;++j)weight_tile[dest+j]=half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a,b,c);
    }
    float global = global_scale[expert]/2688.0f;
    for(uint i = 0; i < c.get_capacity(); ++i) {
        auto coord = c.get_multidimensional_index(i);
        uint row = start+coord[0], n=n0+coord[1];
        if(row<count && n<N) {
            uint pick = owners[expert*TOTAL+row];
            out[size_t(pick)*N+n] = O(c[i]*global);
        }
    }
}
"""


_SMALL = r"""
const uint expert=threadgroup_position_in_grid.y;
const uint count=counts[expert];
if(!count || count>4)return;
const uint K=dims[0],N=dims[1],PICKS=dims[2],TOTAL=dims[3];
const uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const uint n0=threadgroup_position_in_grid.x*32+sg*8;
float acc[4][8]={{0}};
for(uint k=0;k<K;k+=512) {
 float xv[4][16];
 for(uint r=0;r<count;++r) {
  uint pick=owners[expert*TOTAL+r],input_row=PER_PICK?pick:pick/PICKS;
  for(uint j=0;j<16;++j) {
   uint col=k+lane*16+j;
   uchar code=xq[size_t(input_row)*K+col];
   uchar sb=xscale[input_row*(K/32)+col/32];
   float scale=as_type<float>(sb==0?0x400000u:uint(sb)<<23);
   xv[r][j]=float(*(thread fp8_e4m3*)(&code))*scale;
  }
 }
 for(uint n=0;n<8;++n) {
  if(n0+n>=N)continue;
  const device ushort* w=(const device ushort*)(
   weight+(size_t(expert)*N+n0+n)*(K/2)+(k+lane*16)/2);
  uchar sb=scales[(size_t(expert)*N+n0+n)*(K/16)+(k+lane*16)/16];
  float scale=float(*(thread fp8_e4m3*)(&sb));
  float dot[4]={0};
  for(uint j=0;j<4;++j) {
   ushort packed=w[j];
   uchar c0=packed,c1=packed>>4,c2=packed>>8,c3=packed>>12;
   float v0=float(*(thread fp4_e2m1*)(&c0)),v1=float(*(thread fp4_e2m1*)(&c1));
   float v2=float(*(thread fp4_e2m1*)(&c2)),v3=float(*(thread fp4_e2m1*)(&c3));
   for(uint r=0;r<count;++r) dot[r]+=(xv[r][4*j]*v0+xv[r][4*j+1]*v1+xv[r][4*j+2]*v2+xv[r][4*j+3]*v3);
  }
  for(uint r=0;r<count;++r)acc[r][n]+=scale*dot[r];
 }
}
float global=global_scale[expert]/2688.0f;
for(uint r=0;r<count;++r)for(uint n=0;n<8;++n) {
 float value=simd_sum(acc[r][n]);
 if(!lane && n0+n<N)out[size_t(owners[expert*TOTAL+r])*N+n0+n]=O(value*global);
}
"""


def _source():
    return (
        "if(threadgroup_position_in_grid.x < dims[4]) {\n"
        + _SMALL
        + "\nreturn;\n}\n"
        + _SOURCE.replace(
            "threadgroup_position_in_grid.x * 128",
            "(threadgroup_position_in_grid.x-dims[4]) * 128",
        )
    )


def mlx_nvfp4_prefill_gate_up_a8(x, gate, up, indices, groups):
    """Share one activation quantization without duplicating checkpoint storage."""
    from sglang.kernels.ops.quantization.mlx_fp8_cache import mlx_fp8_cache_pack

    packed = mlx_fp8_cache_pack(x.reshape(-1, x.shape[-1]))
    return tuple(
        mlx_nvfp4_prefill_a8(x, *values, indices, groups, packed=packed)
        for values in (gate, up)
    )


@cache
def _kernels():
    return (
        mx.fast.metal_kernel(
            name="sglang_mlx_prefill_expert_owners",
            input_names=["ids", "dims"],
            output_names=["owners", "counts"],
            source=_GROUPS,
        ),
        mx.fast.metal_kernel(
            name="sglang_mlx_nvfp4_prefill_a8",
            input_names=[
                "weight",
                "scales",
                "global_scale",
                "xq",
                "xscale",
                "owners",
                "counts",
                "dims",
            ],
            output_names=["out"],
            source=_source(),
            header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            + _HEADER,
        ),
    )


def mlx_nvfp4_prefill_groups(indices, num_experts):
    """Return GPU expert owners once, reusable by gate/up/down projections."""
    if (
        indices.ndim < 1
        or indices.dtype != mx.uint32
        or not indices.size
        or indices.size >= 2**32
        or num_experts <= 0
    ):
        raise ValueError("Expected nonempty uint32 expert indices")
    return tuple(
        _kernels()[0](
            inputs=[indices.reshape(-1), mx.array([indices.size], mx.uint32)],
            grid=(num_experts * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(num_experts, indices.size), (num_experts,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
    )


def mlx_nvfp4_prefill_a8(
    x, weight, scales, global_scale, indices, groups, *, packed=None
):
    """W4A8 with unchanged NVFP4 checkpoint, native MXFP8 activation blocks.

    Residual and output dtypes are retained (BF16 or FP32). Token and per-pick
    activations use the same explicit expert owner table. Shapes are runtime shader parameters.
    """
    from sglang.kernels.ops.quantization.mlx_fp8_cache import mlx_fp8_cache_pack

    if weight.ndim != 3 or indices.ndim < 1 or not indices.size or x.ndim < 1:
        raise ValueError("Expected nonempty grouped NVFP4 matrices and indices")
    e, n, pk = weight.shape
    k = pk * 2
    count, picks = indices.size, indices.shape[-1]
    if not e or not n or not k or not picks:
        raise ValueError("Expected nonempty NVFP4 projection dimensions")
    per_pick = x.size // k == count
    if (
        not e
        or not n
        or not k
        or not picks
        or count >= 2**32
        or len(groups) != 2
        or x.dtype not in (mx.bfloat16, mx.float32)
        or x.shape[-1] != k
        or k % 512
        or n % 16
        or weight.dtype != mx.uint8
        or scales.shape != (e, n, k // 16)
        or scales.dtype != mx.uint8
        or global_scale.dtype != mx.float32
        or global_scale.size != e
        or indices.dtype != mx.uint32
        or x.size // k not in (count, count // picks)
        or groups[0].shape != (e, count)
        or groups[1].shape != (e,)
        or any(v.dtype != mx.uint32 for v in groups)
    ):
        raise ValueError(
            f"Invalid grouped NVFP4 prefill operands: x={x.shape}/{x.dtype}, w={weight.shape}/{weight.dtype}, scales={scales.shape}/{scales.dtype}, global={global_scale.shape}/{global_scale.dtype}, ids={indices.shape}/{indices.dtype}, groups={[v.shape for v in groups]}"
        )
    if packed is None:
        packed = mlx_fp8_cache_pack(x.reshape(-1, k))
    if packed.dtype != mx.uint8 or packed.shape != (x.size // k, k + k // 32):
        raise ValueError("Expected packed MXFP8 activation rows matching x")
    codes = mx.contiguous(packed[:, :k])
    activation_scales = mx.contiguous(packed[:, k:])
    out = _kernels()[1](
        inputs=[
            weight,
            scales,
            global_scale.reshape(-1),
            codes,
            activation_scales,
            *groups,
            mx.array([k, n, picks, count, (n + 31) // 32], mx.uint32),
        ],
        template=[("O", x.dtype), ("PER_PICK", per_pick)],
        grid=((((n + 31) // 32) + ((n + 127) // 128)) * 128, e, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*indices.shape, 1, n)
