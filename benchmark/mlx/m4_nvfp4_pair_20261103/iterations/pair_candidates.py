"""Separate dispatch pairing from exact nibble-decode improvements."""

from functools import cache
import re

import mlx.core as mx

from common import oldpair, rows

_DIRECT = r"""
inline float direct_fp4(uchar code) {
    uint m=uint(code)&7;
    uint value=m<2 ? (m==0 ? 0 : 126u<<23) :
        (((126u+(m>>1))<<23)|((m&1u)<<22));
    return as_type<float>(value|((uint(code)&8u)<<28));
}
inline float native_scale(uchar code) {
    ushort magnitude=code&127;
    ushort bits=(magnitude<<7)|(((magnitude+1)>>7)<<14)|((ushort(code&128))<<8);
    half value=as_type<half>(bits);
    return float(value*256.0);
}
"""


def direct_decode(source):
    source = re.sub(r"float\(\*\(thread fp4_e2m1\*\)\(&([a-zA-Z0-9_]+)\)\)", r"direct_fp4(\1)", source)
    source = re.sub(r"float\(\*\(thread fp8_e4m3\*\)\(&([a-zA-Z0-9_]+)\)\)", r"native_scale(\1)", source)
    return source


@cache
def kernel(decode="native", paired=True, rps=4):
    source = oldpair._SOURCE
    source = source.replace("rg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;\n  ru[r]+=float(*(thread fp8_e4m3*)(&us))*au;", r"""
  {
#pragma clang fp contract(off)
  rg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;
  ru[r]+=float(*(thread fp8_e4m3*)(&us))*au;
  }
""")
    source = source.replace("token=pick/PICKS", "token=EXPERT?pick:pick/PICKS")
    header = rows._HEADER
    if decode == "vector":
        a=source.index("   ushort g=wgp[i]");b=source.index("  }\n  uchar gs=",a)
        source=source[:a]+r"""
   ushort g=wgp[i],u=wup[i];
   ushort4 gc=ushort4(g,g>>4,g>>8,g>>12),uc=ushort4(u,u>>4,u>>8,u>>12);
   half4 gh=as_type<half4>(ushort4((gc&ushort4(7))<<9))*half(16384);
   half4 uh=as_type<half4>(ushort4((uc&ushort4(7))<<9))*half(16384);
   float4 gv=float4(select(gh,-gh,(gc&ushort4(8))!=ushort4(0)));
   float4 uv=float4(select(uh,-uh,(uc&ushort4(8))!=ushort4(0)));
   ag+=(values[4*i]*gv.x+values[4*i+1]*gv.y+values[4*i+2]*gv.z+values[4*i+3]*gv.w);
   au+=(values[4*i]*uv.x+values[4*i+1]*uv.y+values[4*i+2]*uv.z+values[4*i+3]*uv.w);
"""+source[b:]
    elif decode != "native":
        source = direct_decode(source)
        header = _DIRECT
        if decode == "branchless":
            a=header.index("    uint value=");b=header.index("    return as_type",a)
            header=header[:a]+"    uint value=((m+252u-uint(m==1))<<22)&(0u-uint(m!=0));\n"+header[b:]
        elif decode == "half":
            a=header.index("    uint value=");b=header.index("}\ninline float native_scale",a)
            header=header[:a]+r"""    ushort value=ushort(((m+28u-uint(m==1))<<9)&(0u-uint(m!=0)));
    return float(as_type<half>(ushort(value|((code&8u)<<12))));
"""+header[b:]
    if not paired:
        # Same arithmetic/geometry, eliminating only the second matrix.
        source = source.replace("ushort g=wgp[i],u=wup[i];", "ushort g=wgp[i],u=0;")
        source = source.replace("au+=", "// au+=")
        source = source.replace("ru[r]+=", "// ru[r]+=")
        source = source.replace("up[pick*N+row+r]=T(u*gu);", "")
    return mx.fast.metal_kernel(
        name=f"nvfp4_pair_ablation_{decode}_{int(paired)}_{rps}",
        input_names=["x", "wg", "sgate", "global_gate", "wu", "sup", "global_up", "ids", "dims"],
        output_names=["gate", "up"], source=source, header=header,
        compile_options={"math_mode": "safe"},
    )


@cache
def fused_kernel(decode="native", rps=2, sgs=2):
    # Extract the established paired arithmetic; outputs still round separately
    # before native-type activation arithmetic in the epilogue.
    # Build from source independently because callable does not expose source.
    source = oldpair._SOURCE.replace("token=pick/PICKS", "token=EXPERT?pick:pick/PICKS")
    source = source.replace("rg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;\n  ru[r]+=float(*(thread fp8_e4m3*)(&us))*au;", "{\n#pragma clang fp contract(off)\nrg[r]+=float(*(thread fp8_e4m3*)(&gs))*ag;\nru[r]+=float(*(thread fp8_e4m3*)(&us))*au;\n}")
    source=source.replace("gate[pick*N+row+r]=T(g*gg);up[pick*N+row+r]=T(u*gu);", r"""
T gate=T(g*gg),up=T(u*gu);
if(CLAMP){gate=T(metal::min(float(gate),limit[0]));up=T(metal::clamp(float(up),-limit[0],limit[0]));}
auto fraction=1/(1+metal::precise::exp(metal::abs(gate)));
T sig=T(gate<T(0)?fraction:1-fraction),silu=T(gate*sig);
out[pick*N+row+r]=T(silu*up);
""")
    source=source.replace("threadgroup_position_in_grid.y*(2*RPS)","threadgroup_position_in_grid.y*(SGS*RPS)")
    return mx.fast.metal_kernel(name=f"nvfp4_fused_swiglu_{decode}_{rps}_{sgs}",
        input_names=["x","wg","sgate","global_gate","wu","sup","global_up","ids","dims","limit"],
        output_names=["out"],source=source,header=rows._HEADER,
        compile_options={"math_mode":"safe"})


def fused(x,g,u,ids,rps=2,limit=7.0,sgs=2):
    _,n,pk=g.weight.shape;k=pk*2
    return fused_kernel("native",rps,sgs)(
        inputs=[x.reshape(-1,k),g.weight,g.scales,g.weight_scale_2*2688,u.weight,u.scales,u.weight_scale_2*2688,ids.reshape(-1),mx.array([k,n,ids.shape[-1]],mx.int32),mx.array([limit or 0],mx.float32)],
        template=[("T",x.dtype),("RPS",rps),("SGS",sgs),("EXPERT",x.size//k==ids.size),("CLAMP",limit is not None)],
        grid=(32,sgs*((n+sgs*rps-1)//(sgs*rps)),ids.size),threadgroup=(32,sgs,1),output_shapes=[(ids.size,n)],output_dtypes=[x.dtype]
    )[0].reshape(*ids.shape,1,n)


def pair(x, g, u, ids, *, decode="native", rps=4, paired=True):
    _, n, pk = g.weight.shape
    k, count = pk * 2, ids.size
    result = kernel(decode, paired, rps)(
        inputs=[x.reshape(-1, k), g.weight, g.scales, g.weight_scale_2*2688,
                u.weight, u.scales, u.weight_scale_2*2688, ids.reshape(-1),
                mx.array([k, n, ids.shape[-1]], mx.int32)],
        template=[("T", x.dtype), ("RPS", rps), ("EXPERT", x.size//k==count)],
        grid=(32, 2*((n+2*rps-1)//(2*rps)), count), threadgroup=(32, 2, 1),
        output_shapes=[(count, n), (count, n)], output_dtypes=[x.dtype, x.dtype],
    )
    return tuple(v.reshape(*ids.shape, 1, n) for v in result)
