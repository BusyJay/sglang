# SPDX-License-Identifier: Apache-2.0
"""Sorted expert NAX projection with direct original-pick-order stores.

Retains native projection rounding; the caller owns routing reduction.
M5 ABBA supports four-bit operands. Native five/six-bit retains its original
compact scheduling after measured loader regressions in this implementation.
"""
from functools import cache
from pathlib import Path

import mlx.core as mx

from sglang.kernels.ops.gemm.mlx_qmm_swiglu import _expert_worklist

_SOURCE = r"""
using namespace mlx::steel;
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint task=threadgroup_position_in_grid.y;
if(task>=work_count[0])return;
const uint expert=work[task*3],row=work[task*3+1],end=work[task*3+2];
const uint col=threadgroup_position_in_grid.x*64;
constexpr short SM=BM/2,SN=32,TM=SM/16,TN=2,TK=2;
constexpr uint pad=64+16/sizeof(T);
threadgroup T Ws[64*pad];
using wl_t=QuantizedBlockLoader<T,64,64,pad,1,128,GS,BITS>;
const short tm=SM*(sg/2),tn=SN*(sg%2);
const short nr=short(clamp(int(end-row)-tm,0,int(SM)));
wl_t loader((const device uint8_t*)w+size_t(expert)*w_strides[0]*4+size_t(col)*w_strides[1]*4,s+size_t(expert)*s_strides[0]+size_t(col)*s_strides[1],b+size_t(expert)*b_strides[0]+size_t(col)*b_strides[1],K,Ws,sg,lane,w_strides[1]*4,w_strides[2],s_strides[1],b_strides[1],s_strides[2],b_strides[2]);
NAXTile<float,TM,TN> d;d.clear();
const device T* xn=x+(size_t(row)+tm)*K;
for(uint k=0;k<K;k+=64){
 threadgroup_barrier(mem_flags::mem_threadgroup);
 loader.load_unsafe();
 threadgroup_barrier(mem_flags::mem_threadgroup);
 STEEL_PRAGMA_NO_UNROLL
 for(uint kk=0;kk<64;kk+=32){
  if(nr>0){
   NAXTile<T,TM,TK> a;NAXTile<T,TN,TK> wb;
   volatile int compiler_barrier;
   if(row+tm+SM<=M)a.load(xn+kk,K);
   else a.load_safe(xn+kk,K,short2(32,nr));
   wb.template load<T,pad,1>(Ws+tn*pad+kk);
   tile_matmad_nax(d,a,metal::bool_constant<false>{},wb,metal::bool_constant<true>{});
   (void)compiler_barrier;
  }
 }
 xn+=64;loader.next();
}
for(short i=0;i<TM;++i){for(short j=0;j<TN;++j){
 auto values=d.frag_at(i,j);
 for(short p=0;p<8;++p){
  auto coord=BaseNAXFrag::get_coord(p);
  const uint r=row+tm+i*16+coord.y,c=col+tn+j*16+coord.x;
  if(r<end)out[size_t(order[r])*N+c]=T(values[p]);
 }
}}
"""

@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_compact_sorted_qmm_scatter",
        input_names=["x", "work", "work_count", "order", "w", "s", "b"],
        output_names=["out"],
        source=_SOURCE,
        header=Path(__file__).with_name("mlx_qmm_swiglu_nax.metal").read_text(),
        ensure_row_contiguous=False,
    )


def mlx_sorted_qmm_scatter(
    x, ids, order, operands, *, bits, group_size, tasks=None
):
    """Sorted picks [M,1,K] to original pick-order [M,1,N]."""
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or x.dtype not in (mx.float16, mx.bfloat16)
        or bits != 4
        or group_size not in (64, 128)
        or not mx.device_info()["architecture"].startswith("applegpu_g17")
    ):
        return None
    w, s, b = operands
    if w.ndim != 3 or w.dtype != mx.uint32 or b is None:
        return None
    m, _, k = x.shape
    e, n, _ = w.shape
    if (
        e == 0
        or m // e < 4
        or k % 64
        or n == 0
        or n % 64
        or w.shape[-1] * 32 != k * bits
        or s.shape != (e, n, k // group_size)
        or b.shape != s.shape
        or s.dtype != x.dtype
        or b.dtype != x.dtype
        or ids.shape != (m,)
        or order.shape != (m,)
        or ids.dtype not in (mx.uint32, mx.int32)
        or order.dtype not in (mx.uint32, mx.int32)
    ):
        return None
    bm = 32 if m // e < 64 else 64
    work, count = (
        _expert_worklist(mx.contiguous(ids), e, m, bm) if tasks is None else tasks
    )
    return _kernel()(
        inputs=[mx.contiguous(x), work, count, mx.contiguous(order), w, s, b],
        template=[
            ("T", x.dtype), ("M", m), ("N", n), ("K", k),
            ("GS", group_size), ("BITS", bits), ("BM", bm),
        ],
        grid=(128 * (n // 64), (m + bm - 1) // bm + e - 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, 1, n)],
        output_dtypes=[x.dtype],
    )[0]
