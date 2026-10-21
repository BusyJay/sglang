# SPDX-License-Identifier: Apache-2.0
"""AVX512-BF16 CPU expert operator; builds lazily at runtime initialization."""

import ctypes
import os
from pathlib import Path


class CpuNvfp4:
    def __init__(self, library, pack, cores, *, gpu_backend="triton"):
        flags = Path("/proc/cpuinfo").read_text()
        required = ("avx512f", "avx512bw", "avx512vl", "avx512_bf16")
        if any(flag not in flags for flag in required):
            raise ValueError("CPU hybrid requires AVX512 BF16/BW/VL")
        available = os.sched_getaffinity(0)
        if not cores:
            chosen, physical = [], set()
            for core in sorted(available):
                siblings = (
                    Path(
                        f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list"
                    )
                    .read_text()
                    .strip()
                )
                if siblings not in physical:
                    chosen.append(core)
                    physical.add(siblings)
                if len(chosen) == 4:
                    break
            cores = tuple(chosen)
        if not 1 <= len(cores) <= 4 or len(set(cores)) != len(cores):
            raise ValueError("CPU hybrid requires 1..4 distinct cores")
        for core in cores:
            if core not in available:
                raise ValueError("CPU hybrid core is outside process affinity")
        siblings = [
            Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
            .read_text()
            .strip()
            for core in cores
        ]
        if len(set(siblings)) != len(siblings):
            raise ValueError("CPU hybrid requires distinct physical cores")
        if pack.manifest.hidden_size % 32 or pack.manifest.intermediate_size % 32:
            raise ValueError("CPU hybrid dimensions must be multiples of 32")
        self.cores = (ctypes.c_int * len(cores))(*cores)
        self.pack = pack
        self.mapping = pack.cuda_source_mapping()
        self.buffer = (ctypes.c_char * len(self.mapping)).from_buffer(self.mapping)
        self.address = ctypes.addressof(self.buffer)
        from sglang.kernels.ops.moe.cpu_nvfp4.build import resolve_library

        self.library = ctypes.CDLL(str(resolve_library(library, cores[0])))
        if (
            not hasattr(self.library, "qwen38_cpu_abi")
            or self.library.qwen38_cpu_abi() != 2
        ):
            raise ValueError(
                "CPU library ABI must be 2 (FP32 output); rebuild the shared library"
            )
        if gpu_backend not in ("triton", "b12x"):
            raise ValueError("unsupported CPU hybrid GPU backend")
        symbol = "qwen38_cpu_run_b12x" if gpu_backend == "b12x" else "qwen38_cpu_run"
        if not hasattr(self.library, symbol):
            raise ValueError("rebuild CPU library for the selected GPU backend")
        self.native_run = getattr(self.library, symbol)
        self.native_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.csf_strides = None
        if pack.manifest.scale_encoding:
            if (
                not hasattr(self.library, "qwen38_cpu_run_csf")
                or not hasattr(self.library, "qwen38_cpu_csf_abi")
                or self.library.qwen38_cpu_csf_abi() != 2
            ):
                raise ValueError("rebuild CPU library for compressed scales")
            self.native_run = self.library.qwen38_cpu_run_csf
            self.native_run.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self.csf_strides = (ctypes.c_uint64 * 2)(
                *(
                    pack.manifest.scale_encoding.get(name, {}).get("stride", 0)
                    for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled")
                )
            )
        self.post_scale = int(gpu_backend == "b12x")
        self.native_run.restype = ctypes.c_int
        self.library.qwen38_cpu_error.restype = ctypes.c_char_p
        names = (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
            "g1_alphas",
            "g2_alphas",
            "w13_input_scale_quant",
            "w2_input_scale_quant",
        )
        self.offsets = {
            layer: (ctypes.c_uint64 * 8)(*(records[n].offset for n in names))
            for layer, records in pack.manifest.tensors.items()
        }

    def run(self, layer, x, ids, weights, output):
        import torch

        m = self.pack.manifest
        if (
            x.device.type != "cpu"
            or x.dtype != torch.bfloat16
            or ids.dtype != torch.int32
            or weights.dtype != torch.float32
            or output.dtype != torch.float32
            or ids.device.type != "cpu"
            or weights.device.type != "cpu"
            or output.device.type != "cpu"
            or x.ndim != 2
            or not 1 <= x.shape[0] <= 3
            or x.shape[1] != m.hidden_size
            or ids.shape != weights.shape
            or ids.shape != (x.shape[0], m.top_k)
            or output.shape != x.shape
            or not all(t.is_contiguous() for t in (x, ids, weights, output))
        ):
            raise ValueError("unsupported CPU hybrid input")
        status = self.native_run(
            self.address,
            len(self.mapping),
            self.offsets[layer],
            m.num_experts,
            m.hidden_size,
            m.intermediate_size,
            x.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            x.shape[0],
            ids.shape[1],
            len(self.cores),
            self.cores,
            len(self.cores),
            output.data_ptr(),
            *(
                (self.csf_strides, self.post_scale)
                if self.csf_strides is not None
                else ()
            ),
        )
        if status:
            # The native ABI checks every route, weight and BF16 activation.
            # Recover the public ValueError diagnostics only on its error path.
            if not bool(torch.isfinite(x).all()) or not bool(
                torch.isfinite(weights).all()
            ):
                raise ValueError("CPU hybrid accepts finite activations/routes only")
            if bool(((ids < -1) | (ids >= m.num_experts)).any()):
                raise ValueError("CPU hybrid route outside expert range")
            raise RuntimeError(self.library.qwen38_cpu_error().decode())

    def release_worker(self):
        self.library.qwen38_cpu_release()

    def close(self):
        del self.buffer
        self.mapping = None


def cpu_nvfp4_moe(operator, layer, x, ids, weights, output):
    operator.run(layer, x, ids, weights, output)
    return output
