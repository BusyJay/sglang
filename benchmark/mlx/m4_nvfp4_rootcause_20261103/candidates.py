"""Independent scan, accumulator indexing, arithmetic and worklist ablations."""

from functools import cache

import mlx.core as mx

from common import exact, old

_LIST = r"""
uint pick=thread_position_in_grid.x, count=dims[0], rows=dims[1];
if(pick>=count)return;
uint expert=ids[pick],n=0;
for(uint i=0;i<pick;++i)if(ids[i]==expert){counts[pick]=0;return;}
for(uint i=pick;i<count;++i)if(ids[i]==expert)matches[pick*rows+n++]=i;
counts[pick]=n;
"""


@cache
def list_kernel():
    return mx.fast.metal_kernel(
        name="nvfp4_ablation_worklist", input_names=["ids", "dims"],
        output_names=["matches", "counts"], source=_LIST,
    )


def worklist(ids):
    count, rows = ids.size, ids.shape[-2]
    return list_kernel()(
        inputs=[ids.reshape(-1), mx.array([count, rows], mx.uint32)],
        grid=(count, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(count, rows), (count,)], output_dtypes=[mx.uint32, mx.uint32],
    )


@cache
def kernel(kind="scan", static=False, math="fast", fused=False, rounding="normal"):
    source = old._SOURCE
    inputs = ["x", "weight", "scales", "global_scale", "ids", "dims"]
    start, end = source.index("for(uint i=0;i<pick;"), source.index("uint lane=")
    if kind == "list":
        source = source[:start] + r"""
uint nmatch=counts[pick];
if(nmatch==0)return;
uint matches[ROWS];
#pragma unroll
for(uint t=0;t<ROWS;++t)matches[t]=t<nmatch?match_list[pick*ROWS+t]:pick;
""" + source[end:]
        inputs += ["match_list", "counts"]
    elif kind == "single":
        source = source[:start] + "uint matches[ROWS]={pick},nmatch=1;\n" + source[end:]
    if static:
        source = source.replace("for(uint t=0;t<nmatch;++t)",
                                "\n#pragma unroll\nfor(uint t=0;t<ROWS;++t)if(t<nmatch)")
        source = source.replace("for(int r=0;r<RPS;++r)",
                                "\n#pragma unroll\nfor(int r=0;r<RPS;++r)")
    if fused:
        source = source.replace("result[t][r]+=scale*accum;", "result[t][r]=fma(scale,accum,result[t][r]);")
    if rounding == "separate":
        source = source.replace("result[t][r]+=scale*accum;", "volatile float scaled=scale*accum; result[t][r]+=scaled;")
    elif rounding == "contract_off":
        source = "#pragma clang fp contract(off)\n" + source
    return mx.fast.metal_kernel(
        name=f"nvfp4_ablation_{kind}_{int(static)}_{math}_{int(fused)}_{rounding}",
        input_names=inputs, output_names=["out"], source=source, header=exact._HEADER,
        compile_options={"math_mode": math},
    )


def project(x, p, ids, *, kind="scan", static=False, math="fast", fused=False, rps=2, work=None, rounding="normal"):
    _, n, pk = p.weight.shape
    k, count = pk * 2, ids.size
    rows = 1 if kind == "single" else ids.shape[-2]
    inputs = [x.reshape(-1, k), p.weight, p.scales,
              p.weight_scale_2 * 2688, ids.reshape(-1),
              mx.array([k, n, ids.shape[-1], count], mx.int32)]
    if kind == "list":
        inputs += list(worklist(ids) if work is None else work)
    return kernel(kind, static, math, fused, rounding)(
        inputs=inputs, template=[("O", x.dtype), ("ROWS", rows), ("RPS", rps),
                                 ("EXPERT", x.size // k == count)],
        grid=(32, 2 * ((n + 2 * rps - 1) // (2 * rps)), count),
        threadgroup=(32, 2, 1), output_shapes=[(count, n)], output_dtypes=[x.dtype],
    )[0].reshape(*ids.shape, 1, n)
