"""Keep the archived native qdot and fast-QMV function unchanged."""

from functools import cache

import mlx.core as mx

from common import OUT, exact

native = (OUT / "frozen/native_fp_quantized.h").read_text()
helpers = native[native.index("template <typename T, typename U, int values_per_thread>"):
                 native.index("template <typename U, int values_per_thread, int bits>\ninline U\nqdot_safe")]
dequantize = native[native.index("template <typename T, int group_size>"):
                    native.index("template <typename T, typename U, int values_per_thread>")]
function = native[native.index("template <\n    typename T,\n    int group_size,\n    int bits,\n    bool partial_rows = false"):
                  native.index("template <typename T, int group_size, int bits, bool has_global_scale = false>\nMETAL_FUNC void fp_qmv_impl")]
packing = native[native.index("template <int wsize = 8, int bits = 4>"):
                 native.index("template <typename T, int group_size>")]
header = exact._HEADER + "\nconstant constexpr int SIMD_SIZE=32;\n" + packing + dequantize.replace("fp8_e8m0", "fp8_e4m3") + helpers + function.replace("METAL_FUNC", "inline")


@cache
def kernel(math):
    return mx.fast.metal_kernel(
        name="nvfp4_native_function_" + math,
        input_names=["x", "weight", "scales", "global_scale", "ids", "dims"],
        output_names=["out"], header=header,
        source=r"""
uint pick=threadgroup_position_in_grid.z,expert=ids[pick];
uint K=dims[0],N=dims[1],PICKS=dims[2];
uint vector=EXPERT?pick:pick/PICKS;
fp_qmv_fast_impl<T,16,4,true,true,RPS>(
 (const device uint32_t*)(weight+size_t(expert)*N*(K/2)),
 scales+size_t(expert)*N*(K/16),global_scale+expert,
 x+size_t(vector)*K,out+size_t(pick)*N,
 dims[0],dims[1],uint3(0,threadgroup_position_in_grid.y,0),
 simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
""", compile_options={"math_mode": math},
    )


def project(x, p, ids, math="safe", rps=4):
    _, n, pk = p.weight.shape
    k, count = pk * 2, ids.size
    return kernel(math)(
        inputs=[x.reshape(-1,k),p.weight,p.scales,p.weight_scale_2*2688,
                ids.reshape(-1),mx.array([k,n,ids.shape[-1]],mx.int32)],
        template=[("T",x.dtype),("EXPERT",x.size//k==count),("RPS",rps)],
        grid=(32,2*((n+2*rps-1)//(2*rps)),count),threadgroup=(32,2,1),
        output_shapes=[(count,n)],output_dtypes=[x.dtype],
    )[0].reshape(*ids.shape,1,n)
