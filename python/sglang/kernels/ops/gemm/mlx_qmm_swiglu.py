# SPDX-License-Identifier: Apache-2.0
"""Expert-bucketed QMM with paired gate/up and rounded SwiGLU epilogue.

The SIMD QMM reduction and packed loaders follow Apple MLX 0.32.4. All expert
metadata is read on-device; no per-expert host sorting or dispatch is needed.
"""

from functools import cache
from pathlib import Path

import mlx.core as mx

_SOURCE = r"""
 const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
 uint task=threadgroup_position_in_grid.y;
 uint expert=0,begin=0,end=0;
 for(;expert<E;++expert){
  uint lo=begin,hi=M;
  while(lo<hi){uint mid=(lo+hi)/2;if(ids[mid]<=int(expert))lo=mid+1;else hi=mid;}
  end=lo;
  const uint tiles=(end-begin+BM-1)/BM;
  if(task<tiles)break;
  task-=tiles;begin=end;
 }
 if(expert==E)return;
 const uint row=begin+task*BM;
 const uint col=threadgroup_position_in_grid.x*32;
 constexpr uint pad=32+16/sizeof(T);
 threadgroup T Xs[BM*pad],Gs[32*pad],Us[32*pad];
 using mma_t=mlx::steel::BlockMMA<T,T,BM,32,32,1,2,false,true,pad,pad>;
 using xl_t=mlx::steel::BlockLoader<T,BM,32,pad,1,64>;
 using wl_t=QuantizedBlockLoader<T,32,32,pad,1,64,GS,BITS>;
 const uint nr=min(uint(BM),end-row);
 const size_t base=col;
 const auto gw=(const device uint8_t*)gate_w+size_t(expert)*gate_w_strides[0]*4+base*gate_w_strides[1]*4;
 const auto uw=(const device uint8_t*)up_w+size_t(expert)*up_w_strides[0]*4+base*up_w_strides[1]*4;
 xl_t loader_x(x+size_t(row)*K,K,Xs,sg,lane);
 wl_t loader_g(gw,gate_s+size_t(expert)*gate_s_strides[0]+base*gate_s_strides[1],gate_b+size_t(expert)*gate_b_strides[0]+base*gate_b_strides[1],K,Gs,sg,lane,gate_w_strides[1]*4,gate_w_strides[2],gate_s_strides[1],gate_b_strides[1],gate_s_strides[2],gate_b_strides[2]);
 wl_t loader_u(uw,up_s+size_t(expert)*up_s_strides[0]+base*up_s_strides[1],up_b+size_t(expert)*up_b_strides[0]+base*up_b_strides[1],K,Us,sg,lane,up_w_strides[1]*4,up_w_strides[2],up_s_strides[1],up_b_strides[1],up_s_strides[2],up_b_strides[2]);
 mma_t g(sg,lane),u(sg,lane);
 for(uint k=0;k<K;k+=32){
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(nr==BM)loader_x.load_unsafe();else loader_x.load_safe(short2(32,nr));
  loader_g.load_unsafe();loader_u.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  g.mma(Xs,Gs);u.mma(Xs,Us);
  loader_x.next();loader_g.next();loader_u.next();
 }
 for(short i=0;i<decltype(g)::TM;++i){
  for(short j=0;j<decltype(g)::TN;++j){
   const auto gv=g.Ctile.frag_at(i,j),uv=u.Ctile.frag_at(i,j);
   const uint r=row+g.sm+i*8;
   const uint c=col+g.sn+j*16;
   if(r>=end)continue;
   for(short k=0;k<2;++k){
    T gate=T(gv[k]),up=T(uv[k]);
    if(HAS_LIMIT){gate=T(min(float(gate),float(T(limit[0]))));up=T(clamp(float(up),float(T(-limit[0])),float(T(limit[0]))));}
    auto e=1/(1+metal::precise::exp(metal::abs(gate)));
    const T sigmoid=(gate<0)?e:1-e;
    const T silu=T(gate*sigmoid);
    out[size_t(r)*N+c+k]=T(silu*up);
   }
  }
 }
"""

_NAX_SOURCE = r"""
 using namespace mlx::steel;
 const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
 const uint expert=threadgroup_position_in_grid.z;
 uint lo=0,hi=M;
 while(lo<hi){uint mid=(lo+hi)/2;if(ids[mid]<int(expert))lo=mid+1;else hi=mid;}
 const uint begin=lo;hi=M;
 while(lo<hi){uint mid=(lo+hi)/2;if(ids[mid]<=int(expert))lo=mid+1;else hi=mid;}
 const uint end=lo,row=begin+threadgroup_position_in_grid.y*BM;
 if(row>=end)return;
 const uint col=threadgroup_position_in_grid.x*64;
 constexpr short SM=BM/2,SN=32,TM=SM/16,TN=2,TK=2;
 constexpr uint pad=64+16/sizeof(T);
 threadgroup T Gs[64*pad],Us[64*pad];
 using wl_t=QuantizedBlockLoader<T,64,64,pad,1,128,GS,BITS>;
 const short tm=SM*(sg/2),tn=SN*(sg%2);
 const short nr=short(clamp(int(end-row)-tm,0,int(SM)));
 const size_t base=col;
 wl_t gloader((const device uint8_t*)gate_w+size_t(expert)*gate_w_strides[0]*4+base*gate_w_strides[1]*4,gate_s+size_t(expert)*gate_s_strides[0]+base*gate_s_strides[1],gate_b+size_t(expert)*gate_b_strides[0]+base*gate_b_strides[1],K,Gs,sg,lane,gate_w_strides[1]*4,gate_w_strides[2],gate_s_strides[1],gate_b_strides[1],gate_s_strides[2],gate_b_strides[2]);
 wl_t uloader((const device uint8_t*)up_w+size_t(expert)*up_w_strides[0]*4+base*up_w_strides[1]*4,up_s+size_t(expert)*up_s_strides[0]+base*up_s_strides[1],up_b+size_t(expert)*up_b_strides[0]+base*up_b_strides[1],K,Us,sg,lane,up_w_strides[1]*4,up_w_strides[2],up_s_strides[1],up_b_strides[1],up_s_strides[2],up_b_strides[2]);
 NAXTile<float,TM,TN> g,u;g.clear();u.clear();
 const device T* xn=x+(size_t(row)+tm)*K;
 for(uint k=0;k<K;k+=64){
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gloader.load_unsafe();uloader.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint kk=0;kk<64;kk+=32){
   if(nr>0){
    NAXTile<T,TM,TK> a;
    NAXTile<T,TN,TK> b,c;
    if(nr==SM)a.load(xn+kk,K);else a.load_safe(xn+kk,K,short2(32,nr));
    b.template load<T,pad,1>(Gs+tn*pad+kk);
    c.template load<T,pad,1>(Us+tn*pad+kk);
    tile_matmad_nax(g,a,metal::bool_constant<false>{},b,metal::bool_constant<true>{});
    tile_matmad_nax(u,a,metal::bool_constant<false>{},c,metal::bool_constant<true>{});
   }
  }
  xn+=64;gloader.next();uloader.next();
 }
 for(short i=0;i<TM;++i){for(short j=0;j<TN;++j){
  auto gv=g.frag_at(i,j),uv=u.frag_at(i,j);
  for(short p=0;p<8;++p){
   auto coord=BaseNAXFrag::get_coord(p);
   const uint r=row+tm+i*16+coord.y,c=col+tn+j*16+coord.x;
   if(r>=end)continue;
   T gate=T(gv[p]),up=T(uv[p]);
   if(HAS_LIMIT){gate=T(min(float(gate),float(T(limit[0]))));up=T(clamp(float(up),float(T(-limit[0])),float(T(limit[0]))));}
   auto e=1/(1+metal::precise::exp(metal::abs(gate)));
   const T sigmoid=(gate<0)?e:1-e,silu=T(gate*sigmoid);
   out[size_t(r)*N+c]=T(silu*up);
  }
 }}
"""

_SPARSE_SOURCE = (
    _SOURCE[: _SOURCE.index(" uint task=")]
    + r"""
 const uint expert=threadgroup_position_in_grid.z;
 uint lo=0,hi=M;
 while(lo<hi){uint mid=(lo+hi)/2;if(ids[mid]<int(expert))lo=mid+1;else hi=mid;}
 const uint begin=lo;hi=M;
 while(lo<hi){uint mid=(lo+hi)/2;if(ids[mid]<=int(expert))lo=mid+1;else hi=mid;}
 const uint end=lo,row=begin+threadgroup_position_in_grid.y*BM;
 if(row>=end)return;
"""
    + _SOURCE[_SOURCE.index(" const uint col=") :]
)


@cache
def _sparse_kernel():
    return mx.fast.metal_kernel(
        name="sglang_sorted_qmm_sparse_swiglu",
        input_names=[
            "x",
            "ids",
            "gate_w",
            "gate_s",
            "gate_b",
            "up_w",
            "up_s",
            "up_b",
            "limit",
        ],
        output_names=["out"],
        source=_SPARSE_SOURCE,
        header=Path(__file__).with_suffix(".metal").read_text(),
        ensure_row_contiguous=False,
    )


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_sorted_qmm_swiglu",
        input_names=[
            "x",
            "ids",
            "gate_w",
            "gate_s",
            "gate_b",
            "up_w",
            "up_s",
            "up_b",
            "limit",
        ],
        output_names=["out"],
        source=_SOURCE,
        header=Path(__file__).with_suffix(".metal").read_text(),
        ensure_row_contiguous=False,
    )


@cache
def _nax_kernel():
    return mx.fast.metal_kernel(
        name="sglang_sorted_qmm_swiglu_nax",
        input_names=[
            "x",
            "ids",
            "gate_w",
            "gate_s",
            "gate_b",
            "up_w",
            "up_s",
            "up_b",
            "limit",
        ],
        output_names=["out"],
        source=_NAX_SOURCE,
        header=Path(__file__).with_name("mlx_qmm_swiglu_nax.metal").read_text(),
        ensure_row_contiguous=False,
    )


_QMV_SOURCE = r"""
 const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
 const uint r=threadgroup_position_in_grid.y,expert=uint(ids[r]);
 const uint col=threadgroup_position_in_grid.x*16+sg*4;
 constexpr uint PF=(BITS==5?8:(BITS==6?4:32/BITS));
 constexpr uint BP=(BITS==5?5:(BITS==6?3:4));
 constexpr uint VPT=PF*PPT, BLOCK=VPT*32;
 float gv[4]={0},uv[4]={0}, xt[VPT];
 for(uint k=0;k<K;k+=BLOCK){
  const uint offset=k+lane*VPT;
  if(offset>=K)continue;
  const float sum=load_vector<T,float,VPT,BITS>(x+size_t(r)*K+offset,xt);
  const size_t byte_offset=offset*BP/PF;
  for(uint j=0;j<4;++j){
   if(col+j>=N)continue;
   const size_t gc=size_t(expert)*gate_w_strides[0]+size_t(col+j)*gate_w_strides[1];
   const size_t uc=size_t(expert)*up_w_strides[0]+size_t(col+j)*up_w_strides[1];
   const uint group=offset/GS;
   const float gs=gate_s[size_t(expert)*gate_s_strides[0]+size_t(col+j)*gate_s_strides[1]+group*gate_s_strides[2]];
   const float gb=gate_b[size_t(expert)*gate_b_strides[0]+size_t(col+j)*gate_b_strides[1]+group*gate_b_strides[2]];
   const float us=up_s[size_t(expert)*up_s_strides[0]+size_t(col+j)*up_s_strides[1]+group*up_s_strides[2]];
   const float ub=up_b[size_t(expert)*up_b_strides[0]+size_t(col+j)*up_b_strides[1]+group*up_b_strides[2]];
   if(gate_w_strides[2]==1 && up_w_strides[2]==1){
   gv[j]+=qdot<float,VPT,BITS>((const device uint8_t*)gate_w+gc*4+byte_offset,xt,gs,gb,sum);
   uv[j]+=qdot<float,VPT,BITS>((const device uint8_t*)up_w+uc*4+byte_offset,xt,us,ub,sum);
   }else{
   gv[j]+=qdot<float,VPT,BITS>(StridedPackedBytes{(const device uint8_t*)gate_w+gc*4,size_t(gate_w_strides[2]),byte_offset},xt,gs,gb,sum);
   uv[j]+=qdot<float,VPT,BITS>(StridedPackedBytes{(const device uint8_t*)up_w+uc*4,size_t(up_w_strides[2]),byte_offset},xt,us,ub,sum);
   }
  }
 }
 for(uint j=0;j<4;++j){
  if(col+j>=N)continue;
  const float g=simd_sum(gv[j]),u=simd_sum(uv[j]);
  if(lane==0){
   T gate=T(g),up=T(u);
   if(HAS_LIMIT){gate=T(min(float(gate),float(T(limit[0]))));up=T(clamp(float(up),float(T(-limit[0])),float(T(limit[0]))));}
   auto e=1/(1+metal::precise::exp(metal::abs(gate)));
   const T sigmoid=(gate<0)?e:1-e,silu=T(gate*sigmoid);
   out[size_t(r)*N+col+j]=T(silu*up);
  }
 }
"""


def _qmv_layout_source():
    # The predicate observes Metal's real layout. The common contiguous case
    # uses constant address increments, while independent strided operands keep
    # the original loader without any checkpoint-sized contiguous copies.
    source, conditions = _QMV_SOURCE, []
    for name in ("gate_w", "up_w"):
        for axis, stride in enumerate(("N*K*BITS/32", "K*BITS/32", "1")):
            conditions.append(f"{name}_strides[{axis}]=={stride}")
            source = source.replace(f"{name}_strides[{axis}]", f"({stride})")
    for name in ("gate_s", "up_s", "gate_b", "up_b"):
        for axis, stride in enumerate(("N*K/GS", "K/GS", "1")):
            conditions.append(f"{name}_strides[{axis}]=={stride}")
            source = source.replace(f"{name}_strides[{axis}]", f"({stride})")
    return (
        "if("
        + " && ".join(conditions)
        + "){\n"
        + source
        + "\n}else{\n"
        + _QMV_SOURCE
        + "\n}"
    )


@cache
def _qmv_kernel():
    return mx.fast.metal_kernel(
        name="sglang_paired_qmv_swiglu",
        input_names=[
            "x",
            "ids",
            "gate_w",
            "gate_s",
            "gate_b",
            "up_w",
            "up_s",
            "up_b",
            "limit",
        ],
        output_names=["out"],
        source=_qmv_layout_source(),
        header=Path(__file__).with_name("mlx_qmv_swiglu.metal").read_text(),
        ensure_row_contiguous=False,
    )


def mlx_sorted_qmm_swiglu(x, indices, gate, up, *, bits, group_size, limit=None):
    """Sorted picks [M,1,K] -> [M,1,N], retaining native T rounding."""
    if x.ndim != 3 or x.shape[1] != 1 or x.dtype not in (mx.float16, mx.bfloat16):
        return None
    nax = mx.device_info()["architecture"].startswith("applegpu_g17")
    m, _, k = x.shape
    if m == 0:
        return None
    w, scales, biases = gate
    if bits not in (4, 5, 6, 8) or group_size not in (32, 64, 128):
        return None
    if w.ndim != 3 or k % 32 or k % group_size or w.shape[-1] * 32 != k * bits:
        return None
    e, n, _ = w.shape
    if n == 0 or indices.shape != (m,) or indices.dtype not in (mx.int32, mx.uint32):
        return None
    if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(gate, up)):
        return None
    if (
        scales.shape != (e, n, k // group_size)
        or biases is None
        or biases.shape != scales.shape
    ):
        return None
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    # Pack only small activation/index inputs. Expert operands retain their
    # independent batch, row and packed-column strides in the loader.
    x = mx.contiguous(x)
    indices = mx.contiguous(indices.astype(mx.int32))
    # Match native gather_qmm's sparse-expert GEMV arithmetic while sharing
    # activation loads and fusing gate/up plus the rounded activation epilogue.
    pack_factor = 8 if bits == 5 else (4 if bits == 6 else 32 // bits)
    packs_per_thread = 2 if k % (pack_factor * 2 * 32) == 0 else 1
    if m // e < 4:
        return _qmv_kernel()(
            inputs=[x, indices, *gate, *up, mx.array([limit or 0.0], mx.float32)],
            template=[
                ("T", x.dtype),
                ("N", n),
                ("K", k),
                ("GS", group_size),
                ("BITS", bits),
                ("PPT", packs_per_thread),
                ("HAS_LIMIT", limit is not None),
            ],
            grid=(128 * ((n + 15) // 16), m, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(m, 1, n)],
            output_dtypes=[x.dtype],
        )[0]
    if n % 32:
        return None
    if nax:
        if k % 64 or group_size < 64 or n % 64:
            return None
        bm = 32 if m // e < 64 else 64
        return _nax_kernel()(
            inputs=[
                x,
                indices.astype(mx.int32),
                *gate,
                *up,
                mx.array([limit or 0.0], mx.float32),
            ],
            template=[
                ("T", x.dtype),
                ("M", m),
                ("N", n),
                ("K", k),
                ("GS", group_size),
                ("BITS", bits),
                ("HAS_LIMIT", limit is not None),
                ("BM", bm),
            ],
            grid=(128 * (n // 64), (m + bm - 1) // bm, e),
            threadgroup=(128, 1, 1),
            output_shapes=[(m, 1, n)],
            output_dtypes=[x.dtype],
        )[0]
    bm = 16 if m // e <= 16 else 32
    sparse = m // e <= 16
    return (_sparse_kernel() if sparse else _kernel())(
        inputs=[
            x,
            indices.astype(mx.int32),
            *gate,
            *up,
            mx.array([limit or 0.0], mx.float32),
        ],
        template=[
            ("T", x.dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("GS", group_size),
            ("BITS", bits),
            ("HAS_LIMIT", limit is not None),
            ("E", e),
            ("BM", bm),
        ],
        grid=(
            64 * (n // 32),
            (m + bm - 1) // bm if sparse else (m + bm - 1) // bm + e - 1,
            e if sparse else 1,
        ),
        threadgroup=(64, 1, 1),
        output_shapes=[(m, 1, n)],
        output_dtypes=[x.dtype],
    )[0]
