# SPDX-License-Identifier: Apache-2.0
"""Diagnostic block-FP8 MPP TensorOps projection; no production adoption.

Adapted from python/sglang/kernels/ops/gemm/mlx_dense_argmax_nax.py.
Consumes unchanged dynamic32 E4M3 activation codes + FP32 scales and unchanged
E4M3 weight bytes + FP32 /128x128 scales. Each SIMD group computes two logical
K64 tiles per K128 block (four K32 MMA), applies its weight scale to that
block's FP32 partial, then combines split-K groups. Activation scales are
applied before matmul.
The descriptor explicitly disables relaxed precision. Reduction and scale
placement differ from mlx_fp8_rows; bit equality is NOT expected. No weights
are reconstructed as BF16 and no dynamic quantization is repeated.

Only 1..4 real rows are supported by this diagnostic. The MPP register tile
is M16/BN32/K32 (two MMA per logical K64), with unused register rows zeroed
inside the kernel; there is no padded input allocation or native padded32
quantized_matmul dispatch.
F32 operands do not prove NAX acceleration on a particular GPU: device timing
and a hardware trace are required, especially on M4 versus M5.

Run the local synthetic contrast (existing environment, no model loading)::

    .venv/mlx-dtype-audit/bin/python benchmark/mlx/ar_optimization_20261103/block_fp8_nax.py --count 1 --split 4

JSON includes source hashes, original/scaled-F32 contrast, BF16 reconstruction
error, cold eval and alternating synchronized raw timings. This is not a
full-model, routing, draft acceptance, or serving/cache correctness test.
Retained as an explicit diagnostic only; nothing installs or selects it in
serving. Adoption requires target-device traces/timings and model-quality/cache
validation. On M4 Pro / MLX 0.32.4, synthetic F32 N129/K1024/count1/split4
gave max error 1.526e-5 versus the original and was slower in synchronized
host timings. BF16 weight reconstruction gave projection max error 0.1603
on that input; it is measured separately and is never used by fn.
"""

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
using diagnostic_left=diagnostic_op::cooperative_tensor_left_input_t<float,float,float>;
using diagnostic_right=diagnostic_op::cooperative_tensor_right_input_t<float,float,float>;
"""

_SOURCE = r"""
const uint K=dims[0], N=dims[1], M=dims[2];
const uint kg=simdgroup_index_in_threadgroup;
const uint col=threadgroup_position_in_grid.x*32;
using ext=metal::dextents<int32_t,2>;
using strides=metal::array<int32_t,2>;
using tt=metal::tensor<threadgroup float,ext,metal::tensor_inline>;
diagnostic_op op;
auto a=op.template get_left_input_cooperative_tensor<float,float,float>();
auto b=op.template get_right_input_cooperative_tensor<float,float,float>();
auto total=op.template get_destination_cooperative_tensor<diagnostic_left,diagnostic_right,float>();
for(uint i=0;i<total.get_capacity();++i)total[i]=0;
for(uint block=kg;block<K/128;block+=SPLIT){
    auto partial=total;
    for(uint i=0;i<partial.get_capacity();++i)partial[i]=0;
    for(uint halfblock=0;halfblock<2;++halfblock){
      for(uint mma=0;mma<2;++mma){
        const uint k=block*128+halfblock*64+mma*32;
        for(uint i=0;i<a.get_capacity();++i){
            auto p=a.get_multidimensional_index(i);
            uint kk=k+p[0], row=p[1];
            a[i]=row<M ? e4m3(xq[size_t(row)*K+kk])*
                xscale[size_t(row)*(K/32)+kk/32] : 0.0f;
        }
        for(uint i=0;i<b.get_capacity();++i){
            auto p=b.get_multidimensional_index(i);
            uint kk=k+p[0], row=col+p[1];
            b[i]=row<N ? e4m3(w[size_t(row)*K+kk]) : 0.0f;
        }
        op.run(a,b,partial);
      }
    }
    // BN32 tiles never straddle a weight scale's N128 boundary.
    float ws=scale[size_t(col/128)*(K/128)+block];
    for(uint i=0;i<total.get_capacity();++i)total[i]+=partial[i]*ws;
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
        name="diagnostic_block_fp8_nax",
        input_names=["w", "scale", "xq", "xscale", "dims"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


def fn(x, w, scale, quantized_input, split=4):
    """Return x-shaped projection using supplied codes, without requantization.

    x provides only output shape/dtype; all activation values come from
    quantized_input. split is a compile-time tuning parameter (1..16).
    K/N/count are runtime dimensions. Validation limits describe this candidate,
    not Metal platform restrictions. Inputs may be views (MLX makes contiguous
    inputs); production callers should measure that cost separately.
    """
    if x.ndim < 1 or x.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        raise ValueError("Expected floating activation rows")
    if w.ndim != 2 or w.dtype != mx.uint8:
        raise ValueError("Expected unchanged E4M3 uint8 [N,K] weight bytes")
    n, k = w.shape
    if n <= 0 or k <= 0 or k % 128 or x.shape[-1] != k:
        raise ValueError("Candidate requires positive N and K divisible by 128")
    count = x.size // k
    if count not in (1, 2, 3, 4) or max(n, k) >= 2**31:
        raise ValueError("Candidate requires 1..4 rows and int32 dimensions")
    if type(split) is not int or not 1 <= split <= 16:
        raise ValueError("Candidate split must be an integer in 1..16")
    if scale.dtype != mx.float32 or scale.size != ((n + 127) // 128) * (k // 128):
        raise ValueError("Expected FP32 weight scales [ceil(N/128),K/128]")
    xq, xs = quantized_input
    if xq.dtype != mx.uint8 or xq.size != count * k:
        raise ValueError("Expected unchanged dynamic32 E4M3 activation codes")
    if xs.dtype != mx.float32 or xs.size != count * (k // 32):
        raise ValueError("Expected FP32 dynamic32 activation scales")
    return _kernel()(
        inputs=[w, scale.reshape(-1), xq.reshape(count, k),
                xs.reshape(count, k // 32), mx.array([k, n, count], mx.int32)],
        template=[("O", x.dtype), ("SPLIT", split)],
        grid=(32 * split * ((n + 31) // 32), 1, 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(count, n)],
        output_dtypes=[x.dtype],
    )[0].reshape(*x.shape[:-1], n)


def _contrast(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    delta = a - b
    return {"max_abs": mx.max(mx.abs(delta)).item(),
            "rms": mx.sqrt(mx.mean(delta * delta)).item(),
            "relative_l2": (mx.sqrt(mx.sum(delta * delta)) /
                            mx.maximum(mx.sqrt(mx.sum(b * b)), 1e-30)).item()}


def main():
    import argparse
    import hashlib
    import importlib.metadata
    import importlib.util
    import json
    from pathlib import Path
    from time import perf_counter_ns

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=129)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--count", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.n <= 0 or args.k <= 0 or args.k % 128 or args.repeats <= 0:
        parser.error("n/repeats must be positive and k divisible by 128")
    if not 1 <= args.split <= 16:
        parser.error("split must be 1..16")
    here = Path(__file__).resolve()
    native_path = here.parent / "remote_source/python/sglang/kernels/ops/gemm/mlx_fp8_rows.py"
    spec = importlib.util.spec_from_file_location("diagnostic_original_fp8", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    mx.random.seed(20261103)
    # Finite original codes, arbitrary non-power-of-two FP32 block scales.
    w = mx.random.randint(0, 127, (args.n, args.k)).astype(mx.uint8)
    signs = mx.random.randint(0, 2, w.shape).astype(mx.uint8) * 128
    w = w | signs
    scale = mx.random.uniform(0.001, 0.02, ((args.n + 127) // 128, args.k // 128))
    x = mx.random.normal((1, args.count, args.k)).astype(mx.float32)
    q = native.mlx_fp8_dynamic_32(x)
    mx.eval(x, w, scale, q)
    calls = {"nax": lambda: fn(x, w, scale, q, args.split),
             "original": lambda: native.mlx_fp8_rows(
                 x, w, scale, block_size=128, quantized_input=q)}
    cold = {}
    for name, call in calls.items():
        start = perf_counter_ns()
        mx.eval(call())
        cold[name] = (perf_counter_ns() - start) / 1000

    def decode(codes):
        mag = (codes & 127).astype(mx.int32)
        exp, mant = mag >> 3, mag & 7
        v = mx.where(exp == 0, mant * (2**-9), (1 + mant / 8) * 2.0**(exp - 7))
        return mx.where((codes & 128) != 0, -v, v).astype(mx.float32)

    ax = decode(q[0]) * mx.repeat(q[1], 32, axis=-1)
    ws = mx.repeat(mx.repeat(scale, 128, axis=0)[:args.n], 128, axis=1)
    weights = decode(w) * ws
    # Diagnostic only: BF16 reconstruction measures a distinct changed operand.
    bfweights = weights.astype(mx.bfloat16).astype(mx.float32)
    ref = ax @ weights.T
    bfref = ax @ bfweights.T
    actual = calls["nax"]().reshape(args.count, args.n)
    original = calls["original"]().reshape(args.count, args.n)
    mx.eval(actual, original, ref, bfref)
    for _ in range(2):
        for call in calls.values():
            mx.eval(call())
    samples = []
    for repeat in range(args.repeats):
        for name in (("nax", "original") if repeat % 2 == 0 else ("original", "nax")):
            start = perf_counter_ns()
            mx.eval(calls[name]())
            samples.append({"repeat": repeat, "arm": name,
                            "us": (perf_counter_ns() - start) / 1000})
    print(json.dumps({
        "device": mx.device_info(), "mlx_version": importlib.metadata.version("mlx"),
        "config": vars(args), "relaxed_precision": False,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (here, native_path)},
        "nax_vs_original": _contrast(actual, original),
        "nax_vs_scaled_f32": _contrast(actual, ref),
        "original_vs_scaled_f32": _contrast(original, ref),
        "bf16_weight_reconstruction": _contrast(bfweights, weights),
        "bf16_projection_vs_scaled_f32": _contrast(bfref, ref),
        "cold_us": cold, "samples": samples,
        "peak_memory_bytes": mx.get_peak_memory(),
        "scope": "synthetic; host submission + synchronized eval; no full-model claims",
    }, indent=2))


if __name__ == "__main__":
    main()
